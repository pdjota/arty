from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def test_topk_tuples_to_ui_items_rounding() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from predict_format import topk_tuples_to_ui_items

    out = topk_tuples_to_ui_items([("a", 0.123456789), ("b", 0.5)])
    assert out == [{"label": "a", "prob": 0.1235}, {"label": "b", "prob": 0.5}]
