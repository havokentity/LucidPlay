"""ConditionalRenderer forward + shape invariants."""

from __future__ import annotations

import torch


def test_forward_output_shape(cpu_device):
    from src.config import FRAME_H, FRAME_W, STATE_DIM
    from src.model import ConditionalRenderer

    m = ConditionalRenderer().to(cpu_device).eval()
    x = torch.zeros(4, STATE_DIM, device=cpu_device)
    with torch.inference_mode():
        y = m(x)
    assert y.shape == (4, 3, FRAME_H, FRAME_W)
    assert y.min() >= 0.0 and y.max() <= 1.0  # sigmoid range


def test_param_count_within_expected_band(cpu_device):
    """Spec §5: ~2–4M params. Catches a runaway capacity regression."""
    from src.model import ConditionalRenderer, num_params

    m = ConditionalRenderer().to(cpu_device)
    n = num_params(m)
    assert 1_500_000 < n < 5_000_000, f"unexpected param count: {n:,}"


def test_deterministic_for_same_input(cpu_device):
    from src.config import STATE_DIM
    from src.model import ConditionalRenderer

    m = ConditionalRenderer().to(cpu_device).eval()
    x = torch.full((1, STATE_DIM), 0.3, device=cpu_device)
    with torch.inference_mode():
        y1 = m(x)
        y2 = m(x)
    assert torch.equal(y1, y2)


def test_render_one_drops_batch_dim(cpu_device):
    from src.config import STATE_DIM
    from src.model import ConditionalRenderer

    m = ConditionalRenderer().to(cpu_device).eval()
    x = torch.zeros(STATE_DIM, device=cpu_device)
    y = m.render_one(x)
    assert y.dim() == 3 and y.size(0) == 3
