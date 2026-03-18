import sys
from pathlib import Path

import torch


def test_resnet50_three_heads_forward_shapes_no_weights() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from model import ResNet50ThreeHeads

    model = ResNet50ThreeHeads(n_genre=10, n_style=27, n_artist=23, weights=None)
    x = torch.randn(2, 3, 224, 224)
    g, s, a = model(x)
    assert g.shape == (2, 10)
    assert s.shape == (2, 27)
    assert a.shape == (2, 23)


def test_resnet50_bilstm_three_heads_forward_shapes_no_weights() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from model import ResNet50BiLSTMThreeHeads

    model = ResNet50BiLSTMThreeHeads(n_genre=10, n_style=27, n_artist=23, weights=None)
    x = torch.randn(2, 3, 224, 224)
    g, s, a = model(x)
    assert g.shape == (2, 10)
    assert s.shape == (2, 27)
    assert a.shape == (2, 23)

