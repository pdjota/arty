# Monorepo Gradio Space

The Arty repo keeps **`gradio/`** as normal files (not a Git submodule). The Hugging Face Space should clone **this entire repository** so `src/model.py` is available next to `gradio/app.py`.

## Space settings

| Setting | Value |
|--------|--------|
| Repository | Your Arty monorepo (e.g. GitHub mirror or HF Git) |
| SDK | Gradio |
| `app_file` | `gradio/app.py` |
| Requirements | `requirements_gradio_space.txt` (includes [`gradio/requirements.txt`](../gradio/requirements.txt)) or equivalent |

## Environment

- `BASELINE_MODEL_REPO_ID` — default `pdjota/cnn-baseline`
- `CNNRNN_MODEL_REPO_ID` — default `pdjota/arty-cnn-rnn`
- `HF_TOKEN` — if model repos are gated

## Migrating from the old submodule Space

1. Push the monorepo (with `gradio/` as plain files) to GitHub / HF.
2. In the Space, **change the linked repository** from `spaces/pdjota/artydemo` to the **monorepo** URL (or create a new Space from the monorepo).
3. Set **`app_file`** to **`gradio/app.py`** and requirements as above.
4. Remove the old submodule-based workflow.

See [gradio/README.md](../gradio/README.md).
