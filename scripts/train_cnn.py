"""
Train CNN (ResNet-50 + GAP + 3 heads) on selected index.
Stratified 70/15/15 train/val/test by artist; checkpoints to checkpoints/.
Usage: python scripts/train_cnn.py [--epochs N] [--resume]
  --resume: load last.pt and train --epochs more (default 10).

Persistence (all under checkpoints/):
  - last.pt: latest epoch (model + optimizer) — used by --resume.
  - best.pt: epoch with lowest val_loss — use for eval.
  - train_log.csv: one row per epoch (loss + acc); append-only.
  - results_summary.csv: best checkpoint metrics; written at end of run.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    INDEX_SELECTED,
    WIKIART_ROOT,
    CHECKPOINT_DIR,
    N_STYLE,
    N_ARTIST,
    N_GENRE,
    BATCH_SIZE,
    LR_BACKBONE,
    LR_HEADS,
    MOMENTUM,
    WEIGHT_DECAY,
    EPOCHS,
    COOLDOWN_EPOCHS,
    GRAD_CLIP,
    LOSS_WEIGHT_ARTIST,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from dataset import WikiArtDataset
from model import ResNet50ThreeHeads


def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def stratified_split(df: pd.DataFrame, seed: int = 42):
    """70/15/15 train/val/test stratified by artist_id."""
    idx = df.index.tolist()
    y = df["artist_id"].values
    idx_train, idx_rest = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    y_rest = df.loc[idx_rest, "artist_id"].values
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.5, stratify=y_rest, random_state=seed)
    return idx_train, idx_val, idx_test


def main() -> None:
    if not INDEX_SELECTED.exists():
        print(f"ERROR: {INDEX_SELECTED} not found. Run scripts/build_artgan_index.py first.")
        sys.exit(1)
    if not WIKIART_ROOT.exists():
        print(f"ERROR: {WIKIART_ROOT} not found.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac GPU (Apple Silicon)
    else:
        device = torch.device("cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INDEX_SELECTED)
    idx_train, idx_val, idx_test = stratified_split(df)

    train_ds = WikiArtDataset(INDEX_SELECTED, WIKIART_ROOT, transform=get_transforms(train=True))
    val_ds = WikiArtDataset(INDEX_SELECTED, WIKIART_ROOT, transform=get_transforms(train=False))
    train_subset = Subset(train_ds, idx_train)
    val_subset = Subset(val_ds, idx_val)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs (or extra if --resume)")
    parser.add_argument("--resume", action="store_true", help="Load last.pt and train --epochs more (default 10)")
    args = parser.parse_args()

    start_epoch = 0
    best_val_loss = float("inf")
    resume_path = CHECKPOINT_DIR / "last.pt"

    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        extra = args.epochs if args.epochs is not None else 10
        total_epochs_this_run = extra
        print(f"Resuming from epoch {start_epoch - 1}, training {total_epochs_this_run} more epochs.")
    else:
        if args.resume:
            print("WARNING: --resume but no last.pt found; starting from scratch.")
        extra = None
        total_epochs_this_run = args.epochs if args.epochs is not None else EPOCHS + COOLDOWN_EPOCHS

    model = ResNet50ThreeHeads(n_genre=N_GENRE, n_style=N_STYLE, n_artist=N_ARTIST).to(device)
    backbone_params = list(model.backbone.parameters()) + list(model.pool.parameters())
    head_params = (
        list(model.genre_head.parameters())
        + list(model.style_head.parameters())
        + list(model.artist_head.parameters())
    )
    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEADS},
        ],
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    if args.resume and resume_path.exists():
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    total_steps = total_epochs_this_run * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    def _log_next_steps() -> None:
        print("\n--- Persistence (checkpoints/) ---")
        print("  last.pt     = latest epoch (for --resume)")
        print("  best.pt     = best val_loss (for eval)")
        print("  train_log.csv = per-epoch metrics (append)")
        print("  results_summary.csv = best metrics (written at end)")
        print("To resume training:  python scripts/train_cnn.py --resume --epochs N")
        print("To evaluate best:   python scripts/eval_cnn.py")

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print(f"Saving to: {CHECKPOINT_DIR.resolve()}")
    if start_epoch > 0:
        print(f"Resumed from epoch {start_epoch - 1}; running epochs {start_epoch}..{start_epoch + total_epochs_this_run - 1}")
    else:
        print(f"Starting from scratch; running {total_epochs_this_run} epochs")
    print()

    try:
        for i in range(total_epochs_this_run):
            epoch = start_epoch + i
            model.train()
            train_loss = 0.0
            for images, style_id, artist_id, genre_id in train_loader:
                images = images.to(device)
                style_id = style_id.to(device)
                artist_id = artist_id.to(device)
                genre_id = genre_id.to(device)
                optimizer.zero_grad()
                logits_g, logits_s, logits_a = model(images)
                loss_g = F.cross_entropy(logits_g, genre_id)
                loss_s = F.cross_entropy(logits_s, style_id)
                loss_a = F.cross_entropy(logits_a, artist_id)
                loss = loss_g + loss_s + LOSS_WEIGHT_ARTIST * loss_a
                loss.backward()
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            correct_g = correct_s = correct_a = correct_a5 = total = 0
            with torch.no_grad():
                for images, style_id, artist_id, genre_id in val_loader:
                    images = images.to(device)
                    style_id = style_id.to(device)
                    artist_id = artist_id.to(device)
                    genre_id = genre_id.to(device)
                    logits_g, logits_s, logits_a = model(images)
                    loss = (
                        F.cross_entropy(logits_g, genre_id)
                        + F.cross_entropy(logits_s, style_id)
                        + LOSS_WEIGHT_ARTIST * F.cross_entropy(logits_a, artist_id)
                    )
                    val_loss += loss.item()
                    n = images.size(0)
                    total += n
                    correct_g += (logits_g.argmax(1) == genre_id).sum().item()
                    correct_s += (logits_s.argmax(1) == style_id).sum().item()
                    correct_a += (logits_a.argmax(1) == artist_id).sum().item()
                    _, top5 = logits_a.topk(5, dim=1)
                    correct_a5 += (top5 == artist_id.unsqueeze(1)).any(1).sum().item()
            val_loss /= len(val_loader)
            acc_g = correct_g / total
            acc_s = correct_s / total
            acc_a = correct_a / total
            acc_a5 = correct_a5 / total

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_genre_acc": acc_g,
                "val_style_acc": acc_s,
                "val_artist_acc": acc_a,
                "val_artist_top5_acc": acc_a5,
                "n_genre": N_GENRE,
                "n_style": N_STYLE,
                "n_artist": N_ARTIST,
            }
            torch.save(ckpt, CHECKPOINT_DIR / "last.pt")
            if is_best:
                torch.save(ckpt, CHECKPOINT_DIR / "best.pt")

            log_row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "genre_acc": round(acc_g, 4),
                "style_acc": round(acc_s, 4),
                "artist_acc": round(acc_a, 4),
                "artist_top5_acc": round(acc_a5, 4),
            }
            log_path = CHECKPOINT_DIR / "train_log.csv"
            pd.DataFrame([log_row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

            save_msg = "  [saved last.pt" + (" + best.pt" if is_best else "") + "]"
            print(
                f"epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"genre={acc_g:.2%}  style={acc_s:.2%}  artist={acc_a:.2%}  artist_top5={acc_a5:.2%}  best={is_best}{save_msg}"
            )
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C). Last completed epoch saved in checkpoints/last.pt")
    finally:
        _log_next_steps()

    # Save best-val results summary
    best_ckpt_path = CHECKPOINT_DIR / "best.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
        summary = {
            "best_epoch": best_ckpt.get("epoch"),
            "val_loss": best_ckpt.get("val_loss"),
            "val_genre_acc": best_ckpt.get("val_genre_acc"),
            "val_style_acc": best_ckpt.get("val_style_acc"),
            "val_artist_acc": best_ckpt.get("val_artist_acc"),
            "val_artist_top5_acc": best_ckpt.get("val_artist_top5_acc"),
        }
        pd.DataFrame([summary]).to_csv(CHECKPOINT_DIR / "results_summary.csv", index=False)
        print("Results summary:", CHECKPOINT_DIR / "results_summary.csv")
    print("Done. Best checkpoint:", best_ckpt_path)


if __name__ == "__main__":
    main()
