"""Tests for train_cnn cosine scheduler resume helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "train_cnn.py"


def _load_train_cnn():
    spec = importlib.util.spec_from_file_location("train_cnn", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["train_cnn"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scheduler_legacy_end_of_epoch() -> None:
    mod = _load_train_cnn()
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    t_max = 1000
    ckpt = {"epoch": 2, "interrupted": False}
    sched = mod._scheduler_for_resume_legacy(opt, ckpt, t_max, batches_per_epoch=100)
    # Finished epochs 0,1,2 → 300 optimizer steps; PyTorch reports last_epoch as one past the ctor arg.
    assert sched.last_epoch == 300


def test_scheduler_legacy_interrupted() -> None:
    mod = _load_train_cnn()
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    t_max = 10000
    ckpt = {"epoch": 5, "interrupted": True, "batch_in_epoch": 10}
    sched = mod._scheduler_for_resume_legacy(opt, ckpt, t_max, batches_per_epoch=100)
    assert sched.last_epoch == 5 * 100 + 10


def test_scheduler_state_roundtrip() -> None:
    mod = _load_train_cnn()
    opt = torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 0.01}],
        momentum=0.9,
    )
    t_max = 50
    s1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
    for _ in range(12):
        opt.step()
        s1.step()
    state = s1.state_dict()
    opt2 = torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 0.01}],
        momentum=0.9,
    )
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=t_max)
    s2.load_state_dict(state)
    assert s2.last_epoch == s1.last_epoch
    assert abs(s2.get_last_lr()[0] - s1.get_last_lr()[0]) < 1e-12
