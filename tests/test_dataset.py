"""FrameStateDataset shape + split behavior."""

from __future__ import annotations

import torch


def test_dataset_returns_correct_shapes(tiny_capture):
    from src.config import FRAME_H, FRAME_W, STATE_DIM
    from src.dataset import FrameStateDataset

    ds = FrameStateDataset(str(tiny_capture), split="train", val_frac=0.25)
    state, image = ds[0]
    assert state.shape == (STATE_DIM,)
    assert image.shape == (3, FRAME_H, FRAME_W)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0


def test_split_sizes_sum_to_total(tiny_capture):
    from src.dataset import FrameStateDataset

    val_frac = 0.25
    train = FrameStateDataset(str(tiny_capture), split="train", val_frac=val_frac)
    val = FrameStateDataset(str(tiny_capture), split="val", val_frac=val_frac)
    assert len(train) + len(val) == 16
    assert len(val) >= 1


def test_invalid_split_raises(tiny_capture):
    from src.dataset import FrameStateDataset
    import pytest as _pytest

    with _pytest.raises(ValueError):
        FrameStateDataset(str(tiny_capture), split="bogus")
