import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import ResNet50BiLSTMThreeHeads  # type: ignore


MODEL_REPO_ID = "pdjota/arty-cnnrnn"  # adjust if different
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_id2label(filename: str) -> Dict[int, str]:
    path = hf_hub_download(MODEL_REPO_ID, filename, repo_type="model")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_model() -> Tuple[torch.nn.Module, Dict[int, str], Dict[int, str], Dict[int, str]]:
    ckpt_path = hf_hub_download(MODEL_REPO_ID, "best_model.pt", repo_type="model")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    n_genre = ckpt["n_genre"]
    n_style = ckpt["n_style"]
    n_artist = ckpt["n_artist"]

    model = ResNet50BiLSTMThreeHeads(
        n_genre=n_genre,
        n_style=n_style,
        n_artist=n_artist,
        weights=None,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    genre_id2label = load_id2label("genre_id2label.json")
    style_id2label = load_id2label("style_id2label.json")
    artist_id2label = load_id2label("artist_id2label.json")

    return model, genre_id2label, style_id2label, artist_id2label


model, GENRE_ID2LABEL, STYLE_ID2LABEL, ARTIST_ID2LABEL = load_model()

transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def topk_to_readable(
    logits: torch.Tensor,
    id2label: Dict[int, str],
    k: int = 3,
) -> List[Dict[str, Any]]:
    probs = F.softmax(logits, dim=-1)
    values, indices = probs.topk(k, dim=-1)
    values = values[0].tolist()
    indices = indices[0].tolist()
    out: List[Dict[str, Any]] = []
    for p, idx in zip(values, indices):
        label = id2label.get(idx, str(idx))
        out.append({"label": label, "id": int(idx), "prob": float(p)})
    return out


def predict(image: Image.Image) -> Dict[str, Any]:
    if image is None:
        return {}
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_g, logits_s, logits_a = model(x)

    return {
        "genre_top3": topk_to_readable(logits_g, GENRE_ID2LABEL, k=3),
        "style_top3": topk_to_readable(logits_s, STYLE_ID2LABEL, k=3),
        "artist_top3": topk_to_readable(logits_a, ARTIST_ID2LABEL, k=3),
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a painting"),
    outputs=gr.JSON(label="Predictions"),
    title="Arty: CNN-RNN WikiArt Classifier",
    description=(
        "Upload a painting to get top-3 predictions for genre, style, and artist "
        "from the CNN-RNN (ResNet-50 + BiLSTM) model."
    ),
)


if __name__ == "__main__":
    demo.launch()

