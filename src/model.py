"""ResNet-50 backbone + global average pooling + three classification heads."""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50ThreeHeads(nn.Module):
    """ResNet-50 (ImageNet pretrained), GAP, then three linear heads: genre, style, artist."""

    def __init__(
        self,
        n_genre: int,
        n_style: int,
        n_artist: int,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.n_genre = n_genre
        self.n_style = n_style
        self.n_artist = n_artist
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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
