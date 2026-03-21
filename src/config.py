"""Paths and constants for CNN training from selected index."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WIKIART_ROOT = DATA_DIR / "wikiart"
INDEX_SELECTED = DATA_DIR / "wikiart_index_selected.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"

# Default subdirs under checkpoints/ (best.pt, last.pt, train_log.csv, results_summary.csv).
# Override with env: ARTY_CHECKPOINT_CNN_DIR / ARTY_CHECKPOINT_CNNRNN_DIR (relative to checkpoints/, e.g. "cnn/mar17").
_DEFAULT_CKPT_SUBDIR: dict[str, str] = {"cnn": "cnn_baseline", "cnnrnn": "cnnrnn"}
_ENV_CKPT: dict[str, str] = {
    "cnn": "ARTY_CHECKPOINT_CNN_DIR",
    "cnnrnn": "ARTY_CHECKPOINT_CNNRNN_DIR",
}


def checkpoint_dir_for_arch(arch: str) -> Path:
    """Directory for a given architecture's training artifacts."""
    a = arch.lower()
    if a not in _DEFAULT_CKPT_SUBDIR:
        raise ValueError(f"Unknown arch {arch!r}; expected 'cnn' or 'cnnrnn'")
    rel = os.environ.get(_ENV_CKPT[a], "").strip()
    if not rel:
        rel = _DEFAULT_CKPT_SUBDIR[a]
    return (CHECKPOINT_DIR / rel).resolve()

# Class counts (ArtGAN)
N_STYLE = 27
N_ARTIST = 23
N_GENRE = 10

# Training defaults (Zhao et al.–style; batch reduced for small GPU)
BATCH_SIZE = 64
LR_BACKBONE = 0.01
LR_HEADS = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS = 160
COOLDOWN_EPOCHS = 10
GRAD_CLIP = 1.0
# Loss weight for artist (plan: 2.0)
LOSS_WEIGHT_ARTIST = 2.0

# ImageNet normalize
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
