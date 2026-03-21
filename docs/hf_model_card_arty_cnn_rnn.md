---
license: apache-2.0
language: en
pipeline_tag: image-classification
library_name: pytorch
datasets:
  - pdjota/artyset
tags:
  - pytorch
  - torchvision
  - resnet-50
  - image-classification
  - multi-task
  - wikiart
  - art
  - lstm
  - bilstm

model-index:
  - name: Arty CNN-RNN
    results:
      - task:
          type: image-classification
        dataset:
          name: pdjota/artyset
          type: pdjota/artyset
        metrics:
          - name: Genre accuracy (top-1)
            type: accuracy
            value: 0.678015
          - name: Style accuracy (top-1)
            type: accuracy
            value: 0.650533
          - name: Artist accuracy (top-1)
            type: accuracy
            value: 0.577112
          - name: Artist accuracy (top-5)
            type: accuracy
            value: 0.884742
        source:
          name: Local eval (scripts/eval_cnn.py)
          url: https://huggingface.co/docs/hub/model-cards
---

# Arty CNN-RNN

Multi-head **ResNet-50** backbone with **column pooling**, a **bidirectional LSTM** over spatial strips, and linear heads for **genre**, **style**, and **artist** on a WikiArt subset ([pdjota/artyset](https://huggingface.co/datasets/pdjota/artyset)).

## Evaluation (test split)

| Metric | Value |
|--------|------:|
| Genre (top-1) | 67.80% |
| Style (top-1) | 65.05% |
| Artist (top-1) | 57.71% |
| Artist (top-5) | 88.47% |

- **Checkpoint** (local eval): `checkpoints/cnnrnn/best.pt` — on Hub this repo typically ships as **`best_model.pt`**.  
- **Arch**: `cnnrnn`  
- **Epoch** (from checkpoint): 0  
- **Test images**: 2438

## Files on this model repo

Typical layout after upload:

- `best_model.pt` — PyTorch checkpoint (`model_state_dict`, `n_genre` / `n_style` / `n_artist`, optional `arch`)
- `genre_id2label.json`, `style_id2label.json`, `artist_id2label.json` — class index → label for demos

## Limitations

Not for production attribution or forensic ID; academic / demo use.
