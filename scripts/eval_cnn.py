"""
Evaluate best (or last) checkpoint on the test set.
Reports genre, style, artist top-1 and artist top-5 accuracy.
Usage: python scripts/eval_cnn.py [--arch cnn|cnnrnn] [--last]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, TypedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import INDEX_SELECTED, WIKIART_ROOT, checkpoint_dir_for_arch, N_STYLE, N_ARTIST, N_GENRE, BATCH_SIZE
from dataset import WikiArtDataset
from model import ResNet50BiLSTMThreeHeads, ResNet50ThreeHeads

# Reuse train split and transforms
import importlib.util
spec = importlib.util.spec_from_file_location("train_cnn", ROOT / "scripts" / "train_cnn.py")
train_cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_cnn)
get_transforms = train_cnn.get_transforms
stratified_split = train_cnn.stratified_split


class EvalMetrics(TypedDict):
    arch: str
    checkpoint_name: str
    checkpoint_path: str
    epoch: Any
    test_n: int
    genre_top1: float
    style_top1: float
    artist_top1: float
    artist_top5: float


def compute_test_metrics(*, arch: str, last: bool = False) -> EvalMetrics:
    """
    Run the same evaluation as the CLI and return metrics as floats in [0, 1].
    Used by scripts/export_hf_model_card.py for Hub model card YAML.
    """
    ckpt_name = "last.pt" if last else "best.pt"
    ckpt_path = checkpoint_dir_for_arch(arch) / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not INDEX_SELECTED.exists() or not WIKIART_ROOT.exists():
        raise FileNotFoundError("Index or wikiart root missing.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_genre = ckpt["n_genre"]
    n_style = ckpt["n_style"]
    n_artist = ckpt["n_artist"]
    ckpt_arch = ckpt.get("arch", arch)

    import pandas as pd

    df = pd.read_csv(INDEX_SELECTED)
    _, _, idx_test = stratified_split(df)
    ds = WikiArtDataset(INDEX_SELECTED, WIKIART_ROOT, transform=get_transforms(train=False))
    test_loader = DataLoader(Subset(ds, idx_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if ckpt_arch == "cnnrnn":
        model = ResNet50BiLSTMThreeHeads(n_genre=n_genre, n_style=n_style, n_artist=n_artist).to(device)
    else:
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

    assert total > 0
    return {
        "arch": str(ckpt_arch),
        "checkpoint_name": ckpt_name,
        "checkpoint_path": str(ckpt_path),
        "epoch": ckpt.get("epoch", None),
        "test_n": int(total),
        "genre_top1": correct_g / total,
        "style_top1": correct_s / total,
        "artist_top1": correct_a / total,
        "artist_top5": correct_a5 / total,
    }


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="cnn", choices=["cnn", "cnnrnn"], help="Model architecture")
    p.add_argument("--last", action="store_true", help="Evaluate last.pt instead of best.pt")
    args = p.parse_args()
    try:
        m = compute_test_metrics(arch=args.arch, last=args.last)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(
        f"Arch: {m['arch']}  Checkpoint: {m['checkpoint_name']}  "
        f"(epoch {m['epoch']!r})  Test n={m['test_n']}"
    )
    print(f"  genre acc (top-1): {m['genre_top1']:.2%}")
    print(f"  style acc (top-1): {m['style_top1']:.2%}")
    print(f"  artist acc (top-1): {m['artist_top1']:.2%}")
    print(f"  artist acc (top-5): {m['artist_top5']:.2%}")


if __name__ == "__main__":
    main()
