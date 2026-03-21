#!/usr/bin/env python3
"""
Write a Hugging Face model card README (YAML + markdown) with evaluation metrics.

Runs the same test-set evaluation as eval_cnn.py and fills `model-index` metrics.

Usage:
  python scripts/export_hf_model_card.py --arch cnn --output docs/hf_model_card_cnn_baseline.md

See: https://huggingface.co/docs/hub/model-cards#evaluation-results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_cnn import EvalMetrics, compute_test_metrics  # noqa: E402


def render_model_card(
    m: EvalMetrics,
    *,
    model_index_name: str,
    dataset_id: str,
    source_name: str,
    source_url: str,
    license_id: str,
    extra_body: str,
) -> str:
    """Build full README.md content for a Hub model repo."""
    g, s, a, a5 = m["genre_top1"], m["style_top1"], m["artist_top1"], m["artist_top5"]
    epoch = m["epoch"]
    epoch_str = repr(epoch) if epoch is not None else "unknown"

    yaml = f"""---
license: {license_id}
language: en
pipeline_tag: image-classification
library_name: pytorch
datasets:
  - {dataset_id}
tags:
  - pytorch
  - torchvision
  - resnet-50
  - image-classification
  - multi-task
  - wikiart
  - art

model-index:
  - name: {model_index_name}
    results:
      - task:
          type: image-classification
        dataset:
          name: {dataset_id}
          type: {dataset_id}
        metrics:
          - name: Genre accuracy (top-1)
            type: accuracy
            value: {g:.6f}
          - name: Style accuracy (top-1)
            type: accuracy
            value: {s:.6f}
          - name: Artist accuracy (top-1)
            type: accuracy
            value: {a:.6f}
          - name: Artist accuracy (top-5)
            type: accuracy
            value: {a5:.6f}
        source:
          name: {source_name}
          url: {source_url}
---

# {model_index_name}

Multi-head **ResNet-50** classifier: **genre**, **style**, and **artist** on a WikiArt subset ([{dataset_id}](https://huggingface.co/datasets/{dataset_id})).

## Evaluation (test split)

| Metric | Value |
|--------|------:|
| Genre (top-1) | {g:.2%} |
| Style (top-1) | {s:.2%} |
| Artist (top-1) | {a:.2%} |
| Artist (top-5) | {a5:.2%} |

- **Checkpoint**: `{m["checkpoint_name"]}`  
- **Arch**: `{m["arch"]}`  
- **Epoch** (from checkpoint): {epoch_str}  
- **Test images**: {m["test_n"]}

{extra_body}
"""
    return yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Export HF model card README with eval metrics.")
    p.add_argument("--arch", choices=["cnn", "cnnrnn"], default="cnn")
    p.add_argument("--last", action="store_true", help="Use last.pt instead of best.pt")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "hf_model_card_cnn_baseline.md",
        help="Where to write README content (copy to Hub model repo as README.md)",
    )
    p.add_argument("--model-name", default="Arty CNN baseline", help="model-index name + H1 title")
    p.add_argument("--dataset", default="pdjota/artyset", help="Hub dataset id")
    p.add_argument("--source-name", default="Local eval (scripts/eval_cnn.py)")
    p.add_argument("--source-url", default="", help="Optional URL (e.g. GitHub repo)")
    p.add_argument("--license", default="apache-2.0", dest="license_id")
    p.add_argument(
        "--extra-body",
        default="## Limitations\n\nNot for production attribution or forensic ID; academic / demo use.",
        help="Markdown appended after the eval table",
    )
    args = p.parse_args()

    try:
        m = compute_test_metrics(arch=args.arch, last=args.last)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    source_url = args.source_url.strip() or "https://huggingface.co/docs/hub/model-cards"
    text = render_model_card(
        m,
        model_index_name=args.model_name,
        dataset_id=args.dataset,
        source_name=args.source_name,
        source_url=source_url,
        license_id=args.license_id,
        extra_body=args.extra_body,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
