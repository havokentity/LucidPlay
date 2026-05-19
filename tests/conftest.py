"""Shared fixtures.

Forces pygame headless before any test imports the scene module. Keeps
tests fast and pure-CPU — no MPS / CUDA dependency.
"""

from __future__ import annotations

import os

# Must be set before pygame is imported anywhere downstream.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import sys
from pathlib import Path

import pytest

# Make the repo's `src/` importable as `from src.foo import ...` exactly the
# way the scripts/ wrappers do at runtime.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def cpu_device():
    import torch
    return torch.device("cpu")


@pytest.fixture
def tiny_capture(tmp_path):
    """Run capture for a handful of frames into a tmp dir; yield its path."""
    from src.capture import run
    from src.config import CaptureConfig

    out = tmp_path / "tiny"
    cfg = CaptureConfig(out_dir=str(out), n=16, seed=0)
    run(cfg)
    return out


@pytest.fixture
def tiny_checkpoint(tmp_path, cpu_device):
    """Build a fresh untrained ConditionalRenderer and save to a checkpoint path."""
    import torch
    from src.model import ConditionalRenderer

    model = ConditionalRenderer().to(cpu_device)
    ckpt_path = tmp_path / "untrained.pt"
    torch.save({"model": model.state_dict(), "step": 0, "best_val": float("inf")}, ckpt_path)
    return ckpt_path
