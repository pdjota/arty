import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T


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


def test_resnet50_three_heads_predict_topk() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from model import ResNet50ThreeHeads

    model = ResNet50ThreeHeads(n_genre=10, n_style=27, n_artist=23, weights=None)
    x = torch.randn(1, 3, 224, 224)
    gmap = {i: f"g{i}" for i in range(10)}
    smap = {i: f"s{i}" for i in range(27)}
    amap = {i: f"a{i}" for i in range(23)}
    g, s, a = model.predict_topk(
        x,
        genre_id2label=gmap,
        style_id2label=smap,
        artist_id2label=amap,
        k=3,
    )
    assert len(g) == len(s) == len(a) == 3
    assert all(isinstance(name, str) and 0.0 <= p <= 1.0 for name, p in g + s + a)


def test_resnet50_three_heads_predict_topk_from_path(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from model import ResNet50ThreeHeads

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (256, 256), color=(120, 80, 40)).save(img_path, format="JPEG")

    model = ResNet50ThreeHeads(n_genre=10, n_style=27, n_artist=23, weights=None)
    device = torch.device("cpu")
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    gmap = {i: f"g{i}" for i in range(10)}
    smap = {i: f"s{i}" for i in range(27)}
    amap = {i: f"a{i}" for i in range(23)}
    g, s, a = model.predict_topk_from_path(
        img_path,
        transform,
        device,
        genre_id2label=gmap,
        style_id2label=smap,
        artist_id2label=amap,
    )
    assert len(g) == len(s) == len(a) == 3


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

