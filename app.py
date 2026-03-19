import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import ResNet50BiLSTMThreeHeads  # type: ignore
from model import ResNet50ThreeHeads  # type: ignore

BASELINE_REPO = os.environ.get("BASELINE_MODEL_REPO_ID", "pdjota/arty-cnn-baseline")
CNNRNN_REPO = os.environ.get("CNNRNN_MODEL_REPO_ID", "pdjota/arty-cnn-rnn")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.environ.get("HF_TOKEN")

transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# --- model cache -----------------------------------------------------------

_cache: Dict[str, Any] = {}  # repo_id -> {model, genre, style, artist}


def _load(repo_id: str) -> Dict[str, Any]:
    if repo_id in _cache:
        return _cache[repo_id]

    ckpt_path = hf_hub_download(repo_id, "best_model.pt", repo_type="model", token=HF_TOKEN)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    n_genre = ckpt["n_genre"]
    n_style = ckpt["n_style"]
    n_artist = ckpt["n_artist"]
    state = ckpt["model_state_dict"]

    arch = ckpt.get("arch") or ("cnnrnn" if any(k.startswith("lstm.") for k in state) else "cnn")

    if arch == "cnnrnn":
        model = ResNet50BiLSTMThreeHeads(n_genre=n_genre, n_style=n_style, n_artist=n_artist, weights=None)
    else:
        model = ResNet50ThreeHeads(n_genre=n_genre, n_style=n_style, n_artist=n_artist, weights=None)

    model.load_state_dict(state)
    model.to(DEVICE).eval()

    def _id2label(filename: str) -> Dict[int, str]:
        p = hf_hub_download(repo_id, filename, repo_type="model", token=HF_TOKEN)
        with open(p, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}

    entry = {
        "model": model,
        "arch": arch,
        "genre": _id2label("genre_id2label.json"),
        "style": _id2label("style_id2label.json"),
        "artist": _id2label("artist_id2label.json"),
    }
    _cache[repo_id] = entry
    return entry


# --- prediction helpers ----------------------------------------------------

def _topk(logits: torch.Tensor, id2label: Dict[int, str], k: int = 3) -> List[Dict[str, Any]]:
    probs = F.softmax(logits, dim=-1)[0]
    vals, idxs = probs.topk(k)
    return [{"label": id2label.get(int(i), str(int(i))), "prob": round(float(v), 4)} for v, i in zip(vals, idxs)]


def _bucket(pct: float) -> str:
    if pct >= 80:
        return "very likely"
    if pct >= 60:
        return "possible"
    if pct >= 40:
        return "unlikely"
    return "low confidence"


def _summarize(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "Unknown"
    top = items[0]
    pct = top["prob"] * 100
    label = top["label"] if pct >= 40 else "Unknown"
    bucket = _bucket(pct)
    rest = [x["label"] for x in items[1:]]
    tail = f" (others: {', '.join(rest)})" if rest else ""
    return f"{label} — {pct:.0f}% ({bucket}){tail}"


# --- main predict ----------------------------------------------------------

def predict(model_choice: str, image: Optional[Image.Image]) -> Tuple[str, str]:
    if image is None:
        return "", ""

    repo_id = CNNRNN_REPO if model_choice == "CNN-RNN (BiLSTM)" else BASELINE_REPO

    try:
        image = image.convert("RGB")
        assets = _load(repo_id)
        model = assets["model"]

        x = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            lg, ls, la = model(x)

        g3 = _topk(lg, assets["genre"])
        s3 = _topk(ls, assets["style"])
        a3 = _topk(la, assets["artist"])

        summary = "\n".join([
            f"**Genre**: {_summarize(g3)}",
            f"**Style**: {_summarize(s3)}",
            f"**Artist**: {_summarize(a3)}",
        ])
        details = json.dumps({
            "model": repo_id,
            "arch": assets["arch"],
            "genre_top3": g3,
            "style_top3": s3,
            "artist_top3": a3,
        }, indent=2)
        return summary, details

    except Exception as exc:
        err = json.dumps({
            "error": str(exc),
            "repo_id": repo_id,
            "hf_token_present": bool(HF_TOKEN),
            "hint": "Check that the model repo exists and HF_TOKEN secret is set if it is private.",
        }, indent=2)
        return "", err


# --- UI --------------------------------------------------------------------

with gr.Blocks(title="Arty: WikiArt Classifier") as demo:
    gr.Markdown(
        "## Arty — WikiArt painting classifier\n"
        "Upload a painting to get genre, style and artist predictions."
    )
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["CNN baseline", "CNN-RNN (BiLSTM)"],
            value="CNN baseline",
            label="Model",
        )
    img = gr.Image(type="pil", label="Painting")
    btn = gr.Button("Predict")
    summary_md = gr.Markdown(label="Predictions")
    with gr.Accordion("Details (top-3 JSON)", open=True):
        details_box = gr.Textbox(label="", lines=15, interactive=False)

    btn.click(fn=predict, inputs=[model_choice, img], outputs=[summary_md, details_box])

if __name__ == "__main__":
    demo.launch()
