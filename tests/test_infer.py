"""Renderer round-trip: loads a checkpoint and returns valid JPEG bytes."""

from __future__ import annotations


def test_render_returns_jpeg_bytes(tiny_checkpoint, cpu_device):
    from src.infer import Renderer
    from src.scene import WorldState

    r = Renderer(str(tiny_checkpoint), device=cpu_device)
    state = WorldState(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    blob = r.render(state)

    assert isinstance(blob, (bytes, bytearray))
    assert len(blob) > 100
    # JPEG magic: starts with FFD8 FFE0/E1, ends with FFD9.
    assert blob[:2] == b"\xff\xd8"
    assert blob[-2:] == b"\xff\xd9"
