# Checkpoint layout

Training writes to a **single directory per architecture**, under `checkpoints/`:

| Arch   | Default folder        | Env override              |
|--------|------------------------|---------------------------|
| `cnn`  | `checkpoints/cnn_baseline/` | `ARTY_CHECKPOINT_CNN_DIR` |
| `cnnrnn` | `checkpoints/cnnrnn/`   | `ARTY_CHECKPOINT_CNNRNN_DIR` |

Each folder contains:

- `best.pt` — best val loss (for eval / upload)
- `last.pt` — latest epoch (for `--resume`)
- `train_log.csv` — per-epoch metrics
- `results_summary.csv` — summary written at end of training

`ARTY_CHECKPOINT_*` values are **paths relative to `checkpoints/`** (e.g. `cnn/mar17` for nested runs).
