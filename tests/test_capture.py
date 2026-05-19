"""Capture pipeline: writes the right files in the right shape."""

from __future__ import annotations

import json
import os


def test_capture_writes_frames_and_states(tiny_capture):
    frames_dir = tiny_capture / "frames"
    states_path = tiny_capture / "states.jsonl"
    assert frames_dir.is_dir()
    jpgs = sorted(p for p in frames_dir.iterdir() if p.suffix == ".jpg")
    assert len(jpgs) == 16
    # File names are zero-padded indices.
    assert jpgs[0].name == "000000.jpg"
    assert jpgs[-1].name == "000015.jpg"

    lines = [ln for ln in states_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 16
    rec = json.loads(lines[0])
    assert rec["i"] == 0
    assert set(rec["state"].keys()) == {
        "player_x", "player_y", "vx", "vy",
        "on_ground", "facing", "anim_phase", "t",
    }


def test_capture_states_are_normalized(tiny_capture):
    lines = (tiny_capture / "states.jsonl").read_text().splitlines()
    for ln in lines:
        rec = json.loads(ln)
        s = rec["state"]
        # Positions normalized to [-1, 1].
        assert -1.0 - 1e-6 <= s["player_x"] <= 1.0 + 1e-6
        assert -1.0 - 1e-6 <= s["player_y"] <= 1.0 + 1e-6
        # Flags are clean.
        assert s["on_ground"] in (0.0, 1.0)
        assert s["facing"] in (-1.0, 1.0)
        # t is mod-period normalized.
        assert 0.0 <= s["t"] < 1.0
