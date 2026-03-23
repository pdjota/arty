"""Format `model.predict_topk` tuples for Gradio summaries and JSON."""
from __future__ import annotations

from typing import Any


def topk_tuples_to_ui_items(rows: list[tuple[str, float]]) -> list[dict[str, Any]]:
    """Match legacy Gradio `_topk` shape: `label` + `prob` rounded to 4 decimals."""
    return [{"label": label, "prob": round(float(prob), 4)} for label, prob in rows]
