"""
Copy src/model.py → gradio/model.py for submodule-only Hugging Face Spaces
(that clone only the gradio repo and do not have ../src).

Run after changing the model:

    python scripts/sync_model_to_gradio.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "model.py"
DST = ROOT / "gradio" / "model.py"


def sync_model(src: Path = SRC, dst: Path = DST) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text())


def main() -> None:
    sync_model()
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
