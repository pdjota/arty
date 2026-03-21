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

model-index:
  - name: Arty CNN baseline
    results:
      - task:
          type: image-classification
        dataset:
          name: pdjota/artyset
          type: pdjota/artyset
        metrics:
          - name: Genre accuracy (top-1)
            type: accuracy
            value: 0.796144
          - name: Style accuracy (top-1)
            type: accuracy
            value: 0.806809
          - name: Artist accuracy (top-1)
            type: accuracy
            value: 0.844955
          - name: Artist accuracy (top-5)
            type: accuracy
            value: 0.974159
        source:
          name: Local eval (scripts/eval_cnn.py)
          url: https://huggingface.co/docs/hub/model-cards
---

# Arty CNN baseline

Multi-head **ResNet-50** classifier: **genre**, **style**, and **artist** on a WikiArt subset ([pdjota/artyset](https://huggingface.co/datasets/pdjota/artyset)).

## Evaluation (test split)

| Metric | Value |
|--------|------:|
| Genre (top-1) | 79.61% |
| Style (top-1) | 80.68% |
| Artist (top-1) | 84.50% |
| Artist (top-5) | 97.42% |

- **Checkpoint**: `best.pt`  
- **Arch**: `cnn`  
- **Epoch** (from checkpoint): 8  
- **Test images**: 2438

## Limitations

Not for production attribution or forensic ID; academic / demo use.
