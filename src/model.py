"""ResNet-50 backbone variants for multi-task classification."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


def _topk_from_logits(
    logits: torch.Tensor, id2label: dict[int, str], k: int
) -> list[tuple[str, float]]:
    """Map top-k softmax probabilities to label names (batch size 1)."""
    n = logits.size(-1)
    k = min(k, n)
    probs = F.softmax(logits, dim=-1)[0]
    top = probs.topk(k)
    return [(id2label[int(i)], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]


class _ThreeHeadPredictMixin:
    """Top-k decoding for genre / style / artist heads (shared by CNN and CNN–RNN)."""

    def predict_topk(
        self,
        x: torch.Tensor,
        *,
        genre_id2label: dict[int, str],
        style_id2label: dict[int, str],
        artist_id2label: dict[int, str],
        k: int = 3,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float]]]:
        self.eval()
        with torch.no_grad():
            lg, ls, la = self(x)
        return (
            _topk_from_logits(lg, genre_id2label, k),
            _topk_from_logits(ls, style_id2label, k),
            _topk_from_logits(la, artist_id2label, k),
        )

    def predict_topk_from_path(
        self,
        path: Path | str,
        transform: torch.nn.Module,
        device: torch.device,
        *,
        genre_id2label: dict[int, str],
        style_id2label: dict[int, str],
        artist_id2label: dict[int, str],
        k: int = 3,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float]]]:
        from PIL import Image

        p = Path(path)
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        return self.predict_topk(
            x,
            genre_id2label=genre_id2label,
            style_id2label=style_id2label,
            artist_id2label=artist_id2label,
            k=k,
        )


class ResNet50ThreeHeads(_ThreeHeadPredictMixin, nn.Module):
    """ResNet-50 (ImageNet pretrained), GAP, then three linear heads: genre, style, artist."""

    def __init__(
        self,
        n_genre: int,
        n_style: int,
        n_artist: int,
        weights: ResNet50_Weights | None = ResNet50_Weights.IMAGENET1K_V2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.n_genre = n_genre
        self.n_style = n_style
        self.n_artist = n_artist
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 2048
        self.genre_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, n_genre),
        )
        self.style_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, n_style),
        )
        self.artist_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, n_artist),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return (
            self.genre_head(pooled),
            self.style_head(pooled),
            self.artist_head(pooled),
        )


class ResNet50BiLSTMThreeHeads(_ThreeHeadPredictMixin, nn.Module):
    """
    ResNet-50 (ImageNet pretrained) feature map -> column pooling -> BiLSTM -> mean pool -> three heads.

    Minimal-change upgrade over `ResNet50ThreeHeads` to keep the training loop comparable:
    same backbone, same head style (dropout + linear), only the pooling/aggregation is replaced.
    """

    def __init__(
        self,
        n_genre: int,
        n_style: int,
        n_artist: int,
        weights: ResNet50_Weights | None = ResNet50_Weights.IMAGENET1K_V2,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.n_genre = n_genre
        self.n_style = n_style
        self.n_artist = n_artist

        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        feat_dim = 2 * lstm_hidden

        self.genre_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, n_genre),
        )
        self.style_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, n_style),
        )
        self.artist_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, n_artist),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ResNet feature map: [B, 2048, 7, 7] for 224x224 input
        features = self.backbone(x)
        # Column pooling: avg over height -> [B, 2048, 7]
        seq = features.mean(dim=2)
        # [B, 7, 2048]
        seq = seq.permute(0, 2, 1).contiguous()
        # BiLSTM over 7 steps -> [B, 7, 2*h]
        out, _ = self.lstm(seq)
        # Mean pool over sequence -> [B, 2*h]
        pooled = out.mean(dim=1)
        return (
            self.genre_head(pooled),
            self.style_head(pooled),
            self.artist_head(pooled),
        )
