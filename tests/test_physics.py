"""Physics in the game server — pure logic, no WebSocket."""

from __future__ import annotations


def test_idle_player_stays_on_ground():
    from src.game_server import _PlayerSim
    from src.scene import GROUND_H, LEVEL_H

    sim = _PlayerSim()
    for _ in range(60):
        sim.step(1 / 60)
    assert sim.on_ground
    assert abs(sim.py - (LEVEL_H - GROUND_H)) < 1e-3


def test_right_input_moves_player_right():
    from src.game_server import _PlayerSim

    sim = _PlayerSim()
    start = sim.px
    sim.keys = {"right"}
    for _ in range(60):
        sim.step(1 / 60)
    assert sim.px > start
    assert sim.facing == 1.0


def test_jump_lifts_player_off_ground():
    from src.game_server import _PlayerSim

    sim = _PlayerSim()
    sim.keys = {"up"}
    sim.step(1 / 60)  # jump impulse fires
    sim.keys = set()  # release
    sim.step(1 / 60)
    assert not sim.on_ground


def test_player_clamped_to_level_bounds():
    from src.game_server import _PlayerSim
    from src.scene import LEVEL_W, PLAYER_W

    sim = _PlayerSim()
    sim.keys = {"right"}
    for _ in range(60 * 60):  # 60 sim-seconds; plenty to slam into the wall
        sim.step(1 / 60)
    assert sim.px <= LEVEL_W - PLAYER_W * 0.5 + 1e-6
