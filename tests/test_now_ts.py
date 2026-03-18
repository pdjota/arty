import re


def test_now_ts_format() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "train_cnn.py"
    spec = importlib.util.spec_from_file_location("train_cnn", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["train_cnn"] = mod
    spec.loader.exec_module(mod)

    now_ts = mod.now_ts

    s = now_ts()
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", s)

