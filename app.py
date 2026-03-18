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


DEFAULT_MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "pdjota/arty-cnn-baseline")
BASELINE_MODEL_REPO_ID = os.environ.get("BASELINE_MODEL_REPO_ID", "pdjota/arty-cnn-baseline")
CNNRNN_MODEL_REPO_ID = os.environ.get("CNNRNN_MODEL_REPO_ID", "pdjota/arty-cnnrnn")
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
        # Robust inference: CNN-RNN checkpoints contain LSTM parameters.
        has_lstm = any(k.startswith("lstm.") for k in state.keys())
        arch = "cnnrnn" if has_lstm else "cnn"

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

    # Return model + labels; arch is re-derived by caller if needed.
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


def confidence_bucket(prob_pct: float) -> str:
    if prob_pct >= 80.0:
        return "very likely"
    if prob_pct >= 60.0:
        return "possible"
    if prob_pct >= 40.0:
        return "unlikely"
    return "low confidence / Unknown"


def summarize_topk(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "Unknown (low confidence / Unknown)"
    top = items[0]
    label = str(top.get("label", "Unknown"))
    prob = float(top.get("prob", 0.0)) * 100.0
    bucket = confidence_bucket(prob)
    if prob < 40.0:
        label = "Unknown"
    others = [str(x.get("label", "")) for x in items[1:] if x.get("label")]
    others_txt = f" (others: {', '.join(others)})" if others else ""
    return f"{label} — {prob:.0f}% ({bucket}){others_txt}"


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


def _resolve_repo_id(choice: str, override_repo_id: str) -> str:
    override_repo_id = (override_repo_id or "").strip()
    if override_repo_id:
        return override_repo_id
    if choice == "CNN-RNN (BiLSTM)":
        return CNNRNN_MODEL_REPO_ID
    return BASELINE_MODEL_REPO_ID


def predict(choice: str, override_repo_id: str, image: Image.Image) -> Tuple[str, Dict[str, Any]]:
    if image is None:
        return "", {}
    try:
        rid = _resolve_repo_id(choice, override_repo_id) or DEFAULT_MODEL_REPO_ID
        # Detect arch for debugging (same logic as load_model, but without downloading twice)
        ckpt_path = hf_hub_download(rid, "best_model.pt", repo_type="model", token=HF_TOKEN)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", {})
        detected_arch = ckpt.get("arch") or ("cnnrnn" if any(str(k).startswith("lstm.") for k in state.keys()) else "cnn")

        model, gmap, smap, amap = _get_assets(rid)
        x = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits_g, logits_s, logits_a = model(x)

        genre_top3 = topk_to_readable(logits_g, gmap, k=3)
        style_top3 = topk_to_readable(logits_s, smap, k=3)
        artist_top3 = topk_to_readable(logits_a, amap, k=3)

        summary_md = "\n".join(
            [
                f"**Most likely: Genre**: {summarize_topk(genre_top3)}",
                f"**Most likely: Style**: {summarize_topk(style_top3)}",
                f"**Most likely: Artist**: {summarize_topk(artist_top3)}",
            ]
        )

        details = {
            "repo_id": rid,
            "selection": choice,
            "override_repo_id": (override_repo_id or "").strip(),
            "detected_arch": detected_arch,
            "genre_top3": genre_top3,
            "style_top3": style_top3,
            "artist_top3": artist_top3,
        }
        return summary_md, details
    except Exception as e:
        return "", {
            "error": str(e),
            "hint": "Set Space secret HF_TOKEN if repo is private, and set MODEL_REPO_ID (or type it here) to the correct model repo containing best_model.pt + *_id2label.json.",
            "repo_id_tried": rid,
            "hf_token_present": bool(HF_TOKEN),
        }


with gr.Blocks(title="Arty: CNN-RNN WikiArt Classifier") as demo:
    gr.Markdown(
        "Upload a painting to get top-3 predictions for **genre**, **style**, and **artist**.\n\n"
        "If the model repo is private or the repo id is wrong, you’ll see an error JSON instead of the Space crashing."
    )
    model_choice = gr.Dropdown(
        label="Model",
        choices=["CNN baseline", "CNN-RNN (BiLSTM)"],
        value="CNN baseline",
    )
    repo = gr.Textbox(
        label="Override model repo id (optional)",
        value="",
        placeholder="Leave blank to use the selected model above (e.g. pdjota/arty-cnn-baseline)",
    )
    img = gr.Image(type="pil", label="Upload a painting")
    summary = gr.Markdown()
    with gr.Accordion("Details (top-3 + debug)", open=False):
        out = gr.JSON(label="Predictions")
    btn = gr.Button("Predict")
    btn.click(fn=predict, inputs=[model_choice, repo, img], outputs=[summary, out])


if __name__ == "__main__":
    # Spaces currently runs Gradio with SSR enabled by default; disable to avoid
    # asyncio event-loop teardown warnings in some runtimes.
    demo.launch(ssr_mode=False)

