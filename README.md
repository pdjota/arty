---
title: Artydemo
emoji: 🦀
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: "5.50.0"
python_version: "3.11"
app_file: gradio/app.py
pinned: false
---

# Arty

**Arty** is an example solution of the [Painting in a Painting](https://humanai.foundation/gsoc/2024/proposal_PaintingInAPainting.html) proposal for Google Summer of Code 2026. It is a limited multi-task WikiArt classifier: **genre**, **style**, and **artist** in one model with two architectures — a **CNN baseline** (ResNet-50 + global pooling + three heads) and a **CNN–RNN** (same backbone, **bidirectional long short-term memory (BiLSTM)** over spatial features + three heads). This Hugging Face **Space** runs the **Gradio** app in [`gradio/app.py`](gradio/app.py); **weights** load from Hub model repos and **architecture** from [`src/model.py`](src/model.py). Each architecture corresponds to a model: [pdjota/cnn-baseline](https://huggingface.co/pdjota/cnn-baseline) or [pdjota/arty-cnn-rnn](https://huggingface.co/pdjota/arty-cnn-rnn).

**Training:** run [`scripts/train_cnn.py`](scripts/train_cnn.py) locally (CPU / MPS / CUDA). Optional GPU **Docker** workflow: see comments in [`Dockerfile`](Dockerfile). This Space contains a **`sdk: gradio`** for the demo of the model.

## Limitations

- **Not for production attribution, forensic identification, or legal evidence** — research and demo use only; outputs are not certified provenance.
- **Closed label set** — genre, style, and artist heads are trained on a fixed **ArtGAN-aligned** taxonomy ([`pdjota/artyset`](https://huggingface.co/datasets/pdjota/artyset)). Artists, styles, or genres outside that set are **not represented**; predictions are always one of the trained classes.
- **Domain** — models are tuned for **catalogue-style paintings** in the index. Scans, photos of art, sketches, digital pieces, or strong domain shift may behave unpredictably.
- **Metrics** — reported accuracies are on a **held-out test split** from the same distribution as training; real-world performance varies. Top-k scores are **not** calibrated probabilities of “being authentic.”
- **Hardware** — large batches or the CNN–RNN can **OOM** on small GPUs or Apple Silicon; reduce `--batch-size` or use `--cpu` as documented in `train_cnn.py`. The models uploaded to Hugging Face have minimal training below the recommendations of Zhao et al at this time.

## Documentation

- **[`docs/README.md`](docs/README.md)** — index of project docs (checkpoints, Hub model cards, Space layout).
- **[`gradio/README.md`](gradio/README.md)** — running and configuring the Gradio demo.

## About this project

### Classification with three labels together

**Genre** and **style** are relatively **generic**: many paintings share the same movement or subject category, and the model learns broad visual patterns that match those labels. **Artist** is **more specific** — we usually think of *who* painted it **within** a stylistic movement. Distinguishing painters who share a movement like Sisley and Monet in the impressionist movement or identifying Picasso across different periods in symbolism or cubism becomes challenging.

We use a **shared convolutional trunk** (one set of image features) and **three separate heads** (genre, style, artist). The trunk carries **generic** painting features; the heads split **coarse** (genre, style) vs **fine** (artist) decisions so the artist task can specialize without forcing a single output to encode everything at once.

### ResNet-50 and short fine-tuning

**ResNet-50** is a deep convolutional network built from **residual (skip) connections** so very deep stacks train without vanishing gradients ([He et al., 2016](https://arxiv.org/abs/1512.03385); trained on **ImageNet** in that work). **Transfer learning** is standard: features from early conv layers tend to transfer across related visual domains better than random initialization ([Yosinski et al., 2014](https://arxiv.org/abs/1411.1792)). For **paintings**, [Zhao et al. (2021)](https://doi.org/10.1371/journal.pone.0248414) compare the same model families **with and without** ImageNet-based transfer on WikiArt (genre / style / artist) and report strong results with pretraining — so we **fine-tune** the backbone and heads for a **limited number of epochs** instead of training from scratch at ImageNet-scale cost. 
We use an **ArtGAN-aligned** WikiArt-style index: images catalogued with consistent **genre, style, and 
artist** labels; a few broken paths are excluded. A curated dataset is on the Hub (e.g. [`pdjota/artyset`]
(https://huggingface.co/datasets/pdjota/artyset)) for reproducible training.

The resulting classification is already good for some examples. **Spot check (CNN baseline `best.pt` with [`scripts/spot_check_excluded_post_impressionism.py`](scripts/spot_check_excluded_post_impressionism.py)):**  five **Post_Impressionism** images under `data/wikiart_excluded/Post_Impressionism/` (not in the training index) — style top-1 and a short note:

| Painting | Style (top-1) | Comment |
| -------- | ------------- | ------- |
| `henri-matisse_a-vase-with-oranges.jpg` | Post_Impressionism (~80%) | Still life; confident style match. |
| `henri-de-toulouse-lautrec_portrait-of-vincent-van-gogh-1887.jpg` | Impressionism (~99%) | Sketchy handling reads as Impressionist; Post_Impressionism far behind. |
| `pablo-picasso_seated-monkey-1905.jpg` | Post_Impressionism (~42%) | Close with Expressionism; **artist** top-1 Picasso (~93%). |
| `paul-gauguin_a-seashore-1887.jpg` | Impressionism (~75%) | Post_Impressionism second (~16%). |
| `a.y.-jackson_the-edge-of-the-maple-wood-1910.jpg` | Impressionism (~94%) | Landscape; artist head has no A.Y. Jackson class (23 ArtGAN artists). |

### Bidirectional long short-term memory (BiLSTM) on top of the CNN

Zhao et al. note that their setup uses **colour** information heavily and that **spatial** information could still improve classification. Standard **global average pooling (GAP)** after the last conv map **throws away layout**: it averages each channel over all spatial positions, so the classifier sees a **single vector per channel** with **no remaining (x, y) structure** ([Lin et al., 2014](https://arxiv.org/abs/1312.4400); ResNet-50 uses this pattern before its **fully connected (FC)** layer [He et al., 2016](https://arxiv.org/abs/1512.03385)). That answers “what is present” but not “how it is arranged.” We keep the same ResNet backbone, then **turn the spatial grid into a sequence** (e.g. column-wise strips), run a **bidirectional long short-term memory (BiLSTM)**, then classify. The **reasoning** is: composition, figure–ground balance, and brushstroke patterns often have **left–right (or strip-wise) structure**; a sequence model can integrate **context** along that axis **bidirectionally**, which GAP does not model. The CNN–RNN is a **minimal, comparable** upgrade: same heads and training loop, different pooling.

### Data: ArtGAN-aligned index and Hugging Face

We align with the **ArtGAN / WikiArt** lineage so labels are **catalogue-consistent** for genre, style, and artist. We **trim** the index (drop broken/missing files, validate paths) and publish a curated dataset on the Hub (e.g. [`pdjota/artyset`](https://huggingface.co/datasets/pdjota/artyset)) so **training and demos are reproducible**.

`scripts/train_cnn.py` uses **70% / 15% / 15%** train / val / test, **stratified by `artist_id`**. **Reasoning:** if we split randomly by image, we might put **almost all works of a rare artist** in one fold; **artist** is also the label that would most easily “leak” structurally (same brushwork in train vs test). Stratifying by artist keeps **each split’s artist mix** more representative, so validation/test **accuracy and loss** are comparable across runs and less dominated by **which artists** landed in which fold.

### Training artifacts and this Space

Runs save **PyTorch** checkpoints (`best.pt`, `last.pt`), **CSV** logs (`train_log.csv`, `results_summary.csv`), and we upload reference models with **`id2label` JSON** to Hub model repos. Training works on **CPU**, **Apple Silicon (MPS)**, or a **GPU Space**.

**Gradio (this Space):** upload a painting and compare **CNN baseline** vs **CNN–RNN** top-k predictions. Env vars below select which Hub checkpoints to load.

## Env (optional)

- `BASELINE_MODEL_REPO_ID` — default `pdjota/cnn-baseline`
- `CNNRNN_MODEL_REPO_ID` — default `pdjota/arty-cnn-rnn`
- `HF_TOKEN` — if model repos are gated

### References (ResNet / transfer / WikiArt)

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
2. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NeurIPS. [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
3. Zhao, W., Zhou, D., Qiu, X., & Jiang, W. (2021). *Compare the performance of the models in art classification.* PLOS ONE 16(3): e0248414. [DOI:10.1371/journal.pone.0248414](https://doi.org/10.1371/journal.pone.0248414)
4. Lin, M., Chen, Q., & Yan, S. (2014). *Network in network.* ICLR. [arXiv:1312.4400](https://arxiv.org/abs/1312.4400) — global average pooling to aggregate conv feature maps before classification.
