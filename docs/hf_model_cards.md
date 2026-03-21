# Hugging Face model cards (eval metrics)

To generate a **`README.md` for your Hub model repo** with filled `model-index` metrics (see [Evaluation results](https://huggingface.co/docs/hub/model-cards#evaluation-results)):

```bash
# From repo root, with data + checkpoint present
python scripts/export_hf_model_card.py --arch cnn --output docs/hf_model_card_cnn_baseline.md
```

Defaults:

- Reads `checkpoints/cnn_baseline/best.pt` by default (use `--last` for `last.pt`; override dir with `ARTY_CHECKPOINT_CNN_DIR`)
- Writes `docs/hf_model_card_cnn_baseline.md`

Then copy the file to your model repository as **`README.md`** (or merge the YAML into an existing card).

Options: `--model-name`, `--dataset`, `--source-url`, `--license`, `--extra-body`.

CNN-RNN example (default output `docs/hf_model_card_arty_cnn_rnn.md`):

```bash
python scripts/export_hf_model_card.py --arch cnnrnn --model-name "Arty CNN-RNN"
```

Reads `checkpoints/cnnrnn/best.pt` by default (override with `ARTY_CHECKPOINT_CNNRNN_DIR`).
