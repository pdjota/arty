"""
Spot-check the CNN baseline on fixed excluded Post_Impressionism images (README table).

Images live under data/wikiart_excluded/Post_Impressionism/ (not in the training index).
Uses the same eval transforms as training (`train_cnn.get_transforms(train=False)`).

Usage (from repo root):

    python scripts/spot_check_excluded_post_impressionism.py
    python scripts/spot_check_excluded_post_impressionism.py --cpu
    python scripts/spot_check_excluded_post_impressionism.py --top-k 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import checkpoint_dir_for_arch
from model import ResNet50ThreeHeads

DEFAULT_REL_PATHS: tuple[str, ...] = (
    "henri-matisse_a-vase-with-oranges.jpg",
    "henri-de-toulouse-lautrec_portrait-of-vincent-van-gogh-1887.jpg",
    "pablo-picasso_seated-monkey-1905.jpg",
    "paul-gauguin_a-seashore-1887.jpg",
    "a.y.-jackson_the-edge-of-the-maple-wood-1910.jpg",
)

LABEL_MAPS_DIR = ROOT / "data" / "label_maps"
EXCLUDED_STYLE_DIR = ROOT / "data" / "wikiart_excluded" / "Post_Impressionism"


def _load_train_cnn():
    spec = importlib.util.spec_from_file_location("train_cnn", ROOT / "scripts" / "train_cnn.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_id2label(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_label_maps() -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    return (
        load_id2label(LABEL_MAPS_DIR / "genre_id2label.json"),
        load_id2label(LABEL_MAPS_DIR / "style_id2label.json"),
        load_id2label(LABEL_MAPS_DIR / "artist_id2label.json"),
    )


def resolve_device(*, force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description="CNN spot-check on excluded Post_Impressionism examples.")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best.pt (default: checkpoints/<cnn>/best.pt from config)",
    )
    p.add_argument("--top-k", type=int, default=3, metavar="K", help="Top-k per head (default: 3)")
    args = p.parse_args()

    if args.top_k < 1:
        print("ERROR: --top-k must be >= 1", file=sys.stderr)
        sys.exit(1)

    device = resolve_device(force_cpu=args.cpu)
    ckpt_path = args.checkpoint if args.checkpoint is not None else checkpoint_dir_for_arch("cnn") / "best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    genre_map, style_map, artist_map = load_label_maps()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_genre = ckpt["n_genre"]
    n_style = ckpt["n_style"]
    n_artist = ckpt["n_artist"]
    model = ResNet50ThreeHeads(n_genre=n_genre, n_style=n_style, n_artist=n_artist, weights=None)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    train_cnn = _load_train_cnn()
    transform = train_cnn.get_transforms(train=False)

    paths = [EXCLUDED_STYLE_DIR / name for name in DEFAULT_REL_PATHS]

    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Top-k: {args.top_k}")
    post_ids = [k for k, v in style_map.items() if v == "Post_Impressionism"]
    print(f"Post_Impressionism style id(s): {post_ids}")
    print()

    for path in paths:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        g, s, a = model.predict_topk_from_path(
            path,
            transform,
            device,
            genre_id2label=genre_map,
            style_id2label=style_map,
            artist_id2label=artist_map,
            k=args.top_k,
        )
        print("=" * 72)
        print(path.name)
        print("  genre (top-%d):" % args.top_k, g)
        print("  style (top-%d):" % args.top_k, s)
        print("  artist (top-%d):" % args.top_k, a)


if __name__ == "__main__":
    main()
