# Documentation index

| Doc | Purpose |
|-----|--------|
| [setup.md](setup.md) | Development environment: asdf, venv, `pip install -r requirements.txt`. |
| [data_preparation.md](data_preparation.md) | WikiArt / ArtGAN sources, `data/wikiart`, and building CSV indexes. |
| [checkpoints_layout.md](checkpoints_layout.md) | Where `best.pt` / `last.pt` and logs live; env overrides. |
| [hf_model_cards.md](hf_model_cards.md) | Generating Hub model card markdown from local eval. |
| [hf_model_card_cnn_baseline.md](hf_model_card_cnn_baseline.md) | Exported card template (CNN baseline). |
| [hf_model_card_arty_cnn_rnn.md](hf_model_card_arty_cnn_rnn.md) | Exported card template (CNN–RNN). |
| [monorepo_gradio_space.md](monorepo_gradio_space.md) | Hugging Face Gradio Space: repo layout, `app_file`, env vars. |

Training: `scripts/train_cnn.py` (CPU / MPS / CUDA).
