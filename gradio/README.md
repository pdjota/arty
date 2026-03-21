# Arty Gradio demo (monorepo)

Hugging Face reads **Space SDK settings from the repo root** [`README.md`](../README.md) (`sdk: gradio`, `app_file: gradio/app.py`). If the root README used `sdk: docker`, the Space would **not** run this app.

The build must include **`src/model.py`** (architecture) while **weights** load from Hub model repos.

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
2. **Root** [`requirements.txt`](../requirements.txt) must include Gradio deps (this repo adds `gradio` there for the Space build).
3. Migrate off the old Space submodule: use **one repo** + `app_file: gradio/app.py`.

See also **[docs/gradio_space_fix_options.md](../docs/gradio_space_fix_options.md)**.
