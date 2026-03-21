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

# Arty Gradio demo (monorepo)

- Set **`app_file`** to **`gradio/app.py`** (see YAML above).
- The build must include **`src/model.py`** (architecture) while **weights** load from Hub model repos.

## Local run

```bash
cd gradio
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python app.py
```

## Environment variables (optional)

- `BASELINE_MODEL_REPO_ID` — default `pdjota/cnn-baseline`
- `CNNRNN_MODEL_REPO_ID` — default `pdjota/arty-cnn-rnn`
- `HF_TOKEN` — required if model repos are gated / private

## Hugging Face Space checklist

1. Space **linked repo** = your Arty **monorepo** (GitHub or HF Git).
2. **Root** `requirements.txt` is optional; if missing, Space uses **`gradio/requirements.txt`** when the app is under `gradio/` (or add a root `requirements.txt` that `-r gradio/requirements.txt`).
3. Migrate off the old Space submodule: use **one repo** + `app_file: gradio/app.py`.

See also **[docs/gradio_space_fix_options.md](../docs/gradio_space_fix_options.md)**.
