"""
Evaluate best (or last) checkpoint on the test set.
Reports genre, style, artist top-1 and artist top-5 accuracy.
Usage: python scripts/eval_cnn.py [--last]
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import INDEX_SELECTED, WIKIART_ROOT, CHECKPOINT_DIR, N_STYLE, N_ARTIST, N_GENRE, BATCH_SIZE
from dataset import WikiArtDataset
from model import ResNet50ThreeHeads

# Reuse train split and transforms
import importlib.util
spec = importlib.util.spec_from_file_location("train_cnn", ROOT / "scripts" / "train_cnn.py")
train_cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_cnn)
get_transforms = train_cnn.get_transforms
stratified_split = train_cnn.stratified_split


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--last", action="store_true", help="Evaluate last.pt instead of best.pt")
    args = p.parse_args()
    ckpt_name = "last.pt" if args.last else "best.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found. Train first with scripts/train_cnn.py")
        sys.exit(1)
    if not INDEX_SELECTED.exists() or not WIKIART_ROOT.exists():
        print("ERROR: index or wikiart root missing.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac GPU (Apple Silicon)
    else:
        device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    n_genre = ckpt["n_genre"]
    n_style = ckpt["n_style"]
    n_artist = ckpt["n_artist"]

    import pandas as pd
    df = pd.read_csv(INDEX_SELECTED)
    _, _, idx_test = stratified_split(df)
    ds = WikiArtDataset(INDEX_SELECTED, WIKIART_ROOT, transform=get_transforms(train=False))
    test_loader = DataLoader(Subset(ds, idx_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNet50ThreeHeads(n_genre=n_genre, n_style=n_style, n_artist=n_artist).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    correct_g = correct_s = correct_a = correct_a5 = total = 0
    with torch.no_grad():
        for images, style_id, artist_id, genre_id in test_loader:
            images = images.to(device)
            style_id = style_id.to(device)
            artist_id = artist_id.to(device)
            genre_id = genre_id.to(device)
            logits_g, logits_s, logits_a = model(images)
            n = images.size(0)
            total += n
            correct_g += (logits_g.argmax(1) == genre_id).sum().item()
            correct_s += (logits_s.argmax(1) == style_id).sum().item()
            correct_a += (logits_a.argmax(1) == artist_id).sum().item()
            _, top5 = logits_a.topk(5, dim=1)
            correct_a5 += (top5 == artist_id.unsqueeze(1)).any(1).sum().item()

    print(f"Checkpoint: {ckpt_name}  (epoch {ckpt.get('epoch', '?')})  Test n={total}")
    print(f"  genre acc (top-1): {correct_g / total:.2%}")
    print(f"  style acc (top-1): {correct_s / total:.2%}")
    print(f"  artist acc (top-1): {correct_a / total:.2%}")
    print(f"  artist acc (top-5): {correct_a5 / total:.2%}")


if __name__ == "__main__":
    main()
