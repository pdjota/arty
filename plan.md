# Art Attribute Classification — CNN + CBAM

## Task

Classify art images on **Style**, **Artist**, **Genre**, and other attributes (general and specific) using the ArtGAN/WikiArt dataset.

## Dataset

- **Source:** ArtGAN WikiArt dataset (cs-chan/ArtGAN).
- **Structure:** Folders by attribute — `Style/`, `Artist/`, `Genre/` (and optionally other attributes).
- **Labels:** Multi-label or multi-task (style, artist, genre, etc.).

## Strategy

- **Architecture:** Simple CNN backbone + **CBAM** (Convolutional Block Attention Module).
- **Why CBAM:** Lightweight channel + spatial attention; improves “what” and “where” without heavy recurrent layers; fits art classification where both texture/style (channel) and composition (spatial) matter.
- **Why not full convolutional–recurrent:** Recurrent (e.g. LSTM) adds cost and complexity; for single-image classification, CNN + attention is simpler and often sufficient; we keep the “attention” part via CBAM.

### High-level design

1. **Backbone:** Small CNN (e.g. 4–5 conv blocks: Conv → BN → ReLU → [CBAM] → Pool).
2. **CBAM:** After selected conv blocks: Channel Attention (pool → MLP → sigmoid) then Spatial Attention (concat avg/max pool → conv → sigmoid); multiply with feature map.
3. **Heads:** One head per attribute type:
   - **General:** Style, Genre (fewer classes, broader).
   - **Specific:** Artist (more classes, finer-grained).
4. **Training:** Multi-task: shared backbone + CBAM, separate classifiers per attribute; loss = sum of per-attribute cross-entropy (or weighted sum).
5. **Data:** Load from ArtGAN folder layout; train/val split; same augmentations (flip, crop, color jitter) for all tasks.

### Deliverables

- Data loader for ArtGAN/WikiArt folder structure.
- CNN + CBAM model with configurable depth and attention placement.
- Multi-head output for Style, Artist, Genre (and extensible to other attributes).
- Training script (single run for all attributes), validation, and basic metrics (accuracy / top-k per head).
- `plan.md` (this file) and `README.md` with setup and run instructions.

## Implementation order

1. Repo setup: git, asdf Python 3.13.2, `requirements.txt`.
2. Dataset: download/prepare ArtGAN WikiArt; folder-based loader and train/val split.
3. Model: CNN blocks + CBAM module; multi-head output layer.
4. Training: multi-task training loop, logging, checkpointing.
5. Eval and README.
