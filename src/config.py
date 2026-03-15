"""Paths and constants for CNN training from selected index."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WIKIART_ROOT = DATA_DIR / "wikiart"
INDEX_SELECTED = DATA_DIR / "wikiart_index_selected.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"

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
