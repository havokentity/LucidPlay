"""Scene + WorldState invariants."""

from __future__ import annotations

import math


def test_world_state_to_vec_length():
    from src.scene import WorldState
    s = WorldState(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    assert len(s.to_vec()) == 8


def test_world_state_roundtrip():
    from src.scene import WorldState
    s = WorldState(0.1, -0.2, 0.3, -0.4, 1.0, -1.0, 0.5, 0.6)
    s2 = WorldState.from_vec(s.to_vec())
    assert s.to_vec() == s2.to_vec()


def test_scene_renders_target_size():
    from src.config import FRAME_H, FRAME_W
    from src.scene import Scene, WorldState
    scene = Scene()
    surf = scene.render(WorldState(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0))
    assert surf.get_size() == (FRAME_W, FRAME_H)


def test_make_state_normalizes_to_unit_range():
    from src.scene import LEVEL_H, LEVEL_W, GROUND_H, make_state
    # Player at far right of level should map to player_x ≈ +1.
    s = make_state(player_x_px=LEVEL_W, player_y_px=LEVEL_H - GROUND_H)
    assert math.isclose(s.player_x, 1.0, abs_tol=1e-6)
    # On-ground / facing produce {-1,0,1}-like flags.
    assert s.on_ground in (0.0, 1.0)
    assert s.facing in (-1.0, 1.0)


def test_facing_negative_when_left():
    from src.scene import make_state
    s = make_state(player_x_px=0.0, player_y_px=0.0, facing=-1.0)
    assert s.facing == -1.0


def test_anim_phase_wraps_to_unit_interval():
    from src.scene import make_state
    s = make_state(player_x_px=0.0, player_y_px=0.0, anim_phase=2.75)
    assert 0.0 <= s.anim_phase < 1.0
