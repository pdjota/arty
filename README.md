---
title: Artydemo
emoji: 🦀
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.11"
app_file: gradio/app.py
pinned: false
---

# Arty

WikiArt **genre / style / artist** classifiers (CNN baseline and CNN-RNN). This Hugging Face **Space** runs the **Gradio** app under [`gradio/app.py`](gradio/app.py); weights load from Hub model repos and architecture from [`src/model.py`](src/model.py).

**Training** (Docker GPU job) lives in [`Dockerfile`](Dockerfile) — use a **separate** Space pointed at the same repo if you only want training, or run locally. Do not set `sdk: docker` on this Space if you want the Gradio UI.

## Env (optional)

- `BASELINE_MODEL_REPO_ID` — default `pdjota/cnn-baseline`
- `CNNRNN_MODEL_REPO_ID` — default `pdjota/arty-cnn-rnn`
- `HF_TOKEN` — if model repos are gated

More detail: [`gradio/README.md`](gradio/README.md), [`docs/monorepo_gradio_space.md`](docs/monorepo_gradio_space.md).
