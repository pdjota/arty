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


DEFAULT_MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "pdjota/arty-cnnrnn")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.environ.get("HF_TOKEN")


def load_id2label(repo_id: str, filename: str) -> Dict[int, str]:
    path = hf_hub_download(repo_id, filename, repo_type="model", token=HF_TOKEN)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_model(repo_id: str) -> Tuple[torch.nn.Module, Dict[int, str], Dict[int, str], Dict[int, str]]:
    ckpt_path = hf_hub_download(repo_id, "best_model.pt", repo_type="model", token=HF_TOKEN)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    n_genre = ckpt["n_genre"]
    n_style = ckpt["n_style"]
    n_artist = ckpt["n_artist"]

    state = ckpt["model_state_dict"]
    arch = ckpt.get("arch")
    if arch is None:
        # Heuristic: CNN checkpoints have heads on 2048-dim features.
        # (artist_head.1.weight shape is [n_artist, feat_dim]).
        feat_dim = state.get("artist_head.1.weight").shape[1] if "artist_head.1.weight" in state else None
        arch = "cnn" if feat_dim == 2048 else "cnnrnn"

    if arch == "cnnrnn":
        model = ResNet50BiLSTMThreeHeads(
            n_genre=n_genre,
            n_style=n_style,
            n_artist=n_artist,
            weights=None,
        ).to(DEVICE)
    else:
        model = ResNet50ThreeHeads(
            n_genre=n_genre,
            n_style=n_style,
            n_artist=n_artist,
            weights=None,
        ).to(DEVICE)

    model.load_state_dict(state)
    model.eval()

    genre_id2label = load_id2label(repo_id, "genre_id2label.json")
    style_id2label = load_id2label(repo_id, "style_id2label.json")
    artist_id2label = load_id2label(repo_id, "artist_id2label.json")

    return model, genre_id2label, style_id2label, artist_id2label


_CACHED_REPO_ID: Optional[str] = None
_CACHED_MODEL: Optional[torch.nn.Module] = None
_CACHED_LABELS: Optional[Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]] = None

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


def _get_assets(repo_id: str) -> Tuple[torch.nn.Module, Dict[int, str], Dict[int, str], Dict[int, str]]:
    global _CACHED_REPO_ID, _CACHED_MODEL, _CACHED_LABELS
    repo_id = repo_id.strip()
    if _CACHED_MODEL is not None and _CACHED_LABELS is not None and _CACHED_REPO_ID == repo_id:
        g, s, a = _CACHED_LABELS
        return _CACHED_MODEL, g, s, a

    model, g, s, a = load_model(repo_id)
    _CACHED_REPO_ID = repo_id
    _CACHED_MODEL = model
    _CACHED_LABELS = (g, s, a)
    return model, g, s, a


def predict(repo_id: str, image: Image.Image) -> Dict[str, Any]:
    if image is None:
        return {}
    try:
        model, gmap, smap, amap = _get_assets(repo_id or DEFAULT_MODEL_REPO_ID)
        x = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits_g, logits_s, logits_a = model(x)
        return {
            "repo_id": repo_id or DEFAULT_MODEL_REPO_ID,
            "genre_top3": topk_to_readable(logits_g, gmap, k=3),
            "style_top3": topk_to_readable(logits_s, smap, k=3),
            "artist_top3": topk_to_readable(logits_a, amap, k=3),
        }
    except Exception as e:
        return {
            "error": str(e),
            "hint": "Set Space secret HF_TOKEN if repo is private, and set MODEL_REPO_ID (or type it here) to the correct model repo containing best_model.pt + *_id2label.json.",
            "repo_id_tried": (repo_id or DEFAULT_MODEL_REPO_ID),
            "hf_token_present": bool(HF_TOKEN),
        }


with gr.Blocks(title="Arty: CNN-RNN WikiArt Classifier") as demo:
    gr.Markdown(
        "Upload a painting to get top-3 predictions for **genre**, **style**, and **artist**.\n\n"
        "If the model repo is private or the repo id is wrong, you’ll see an error JSON instead of the Space crashing."
    )
    repo = gr.Textbox(label="Model repo id (HF)", value=DEFAULT_MODEL_REPO_ID)
    img = gr.Image(type="pil", label="Upload a painting")
    out = gr.JSON(label="Predictions")
    btn = gr.Button("Predict")
    btn.click(fn=predict, inputs=[repo, img], outputs=[out])


if __name__ == "__main__":
    demo.launch()

