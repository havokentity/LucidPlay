"""Capture (state, frame) pairs from the scripted pygame scene.

This is the only place the ground-truth scene runs in bulk. The output feeds the
training loop. At play time it's gone — the model is the renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Iterator, Tuple

import pygame

from .config import CaptureConfig, FRAME_W, FRAME_H
from .scene import (
    GRAVITY,
    GROUND_H,
    JUMP_VELOCITY,
    LEVEL_H,
    LEVEL_W,
    MOVE_SPEED,
    PLATFORMS,
    PLAYER_H,
    PLAYER_W,
    Scene,
    T_PERIOD,
    WorldState,
    make_state,
    time_to_norm,
    vel_to_norm,
    world_to_norm,
)


# --- Physics shared with game_server ---------------------------------------
def _platform_collide(prev_y: float, new_y: float, x: float, vy: float) -> Tuple[float, float, bool]:
    """Resolve falling onto platform tops. Returns (clamped_y, vy, on_platform)."""
    if vy < 0:
        return new_y, vy, False
    foot_prev = prev_y
    foot_new = new_y
    for (px, py, pw, ph) in PLATFORMS:
        # Player horizontally overlapping platform?
        if x + PLAYER_W * 0.5 < px or x - PLAYER_W * 0.5 > px + pw:
            continue
        # Crossing the top edge this step?
        if foot_prev <= py and foot_new >= py:
            return float(py), 0.0, True
    return new_y, vy, False


def _step_physics(
    px: float, py: float, vx: float, vy: float,
    on_ground: bool, ax_input: float, jump: bool, dt: float,
) -> Tuple[float, float, float, float, bool]:
    """One physics step. Mirrors game_server logic so capture covers the same manifold."""
    # Horizontal velocity is set directly from input (arcade feel).
    vx = ax_input * MOVE_SPEED
    if jump and on_ground:
        vy = JUMP_VELOCITY

    # Gravity
    vy += GRAVITY * dt

    new_x = px + vx * dt
    new_y = py + vy * dt

    # Walls
    half = PLAYER_W * 0.5
    new_x = max(half, min(LEVEL_W - half, new_x))

    # Ground
    ground_y = LEVEL_H - GROUND_H
    next_on_ground = False
    if new_y >= ground_y:
        new_y = ground_y
        vy = 0.0
        next_on_ground = True

    # Platforms (only when falling)
    new_y, vy, on_plat = _platform_collide(py, new_y, new_x, vy)
    if on_plat:
        next_on_ground = True

    return new_x, new_y, vx, vy, next_on_ground


# --- Scripted exploration ---------------------------------------------------
def _scripted_states(n: int, seed: int) -> Iterator[WorldState]:
    """Yield `n` WorldStates from a scripted random agent.

    The agent picks a horizontal target, walks toward it, occasionally jumps
    or idles. This biases the data toward states the player can actually reach,
    which covers the manifold better than uniform random.
    """
    rng = random.Random(seed)

    px = LEVEL_W * 0.5
    py = LEVEL_H - GROUND_H
    vx = 0.0
    vy = 0.0
    on_ground = True
    facing = 1.0
    anim_phase = 0.0
    t = 0.0
    dt = 1.0 / 60.0

    target_x = rng.uniform(0, LEVEL_W)
    plan_ticks = 0
    state_mode = "walk"  # "walk" | "idle" | "jump_arc"

    for _ in range(n):
        # Replan periodically.
        plan_ticks -= 1
        if plan_ticks <= 0:
            choice = rng.random()
            if choice < 0.7:
                state_mode = "walk"
                target_x = rng.uniform(0, LEVEL_W)
                plan_ticks = rng.randint(30, 180)
            elif choice < 0.9:
                state_mode = "idle"
                plan_ticks = rng.randint(15, 90)
            else:
                state_mode = "jump_arc"
                target_x = rng.uniform(0, LEVEL_W)
                plan_ticks = rng.randint(40, 120)

        # Decide input from mode.
        ax = 0.0
        jump = False
        if state_mode == "walk":
            ax = 1.0 if target_x > px else -1.0
            if abs(target_x - px) < 4.0:
                ax = 0.0
        elif state_mode == "jump_arc":
            ax = 1.0 if target_x > px else -1.0
            if on_ground and rng.random() < 0.05:
                jump = True
        # idle: ax stays 0

        if abs(ax) > 0.1:
            facing = 1.0 if ax > 0 else -1.0

        px, py, vx, vy, on_ground = _step_physics(px, py, vx, vy, on_ground, ax, jump, dt)

        # Anim phase advances proportionally to horizontal speed while on ground.
        if on_ground and abs(vx) > 1.0:
            anim_phase = (anim_phase + abs(vx) / MOVE_SPEED * dt * 4.0) % 1.0
        # t marches forward.
        t = (t + dt) % T_PERIOD

        yield make_state(
            player_x_px=px,
            player_y_px=py,
            vx=vx,
            vy=vy,
            on_ground=on_ground,
            facing=facing,
            anim_phase=anim_phase,
            t_seconds=t,
        )


# --- Capture driver ---------------------------------------------------------
def run(cfg: CaptureConfig) -> None:
    out = cfg.out_dir
    frames_dir = os.path.join(out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    states_path = os.path.join(out, "states.jsonl")

    scene = Scene(frame_w=cfg.frame_w, frame_h=cfg.frame_h)

    t_start = time.time()
    with open(states_path, "w", encoding="utf-8") as fs:
        for i, state in enumerate(_scripted_states(cfg.n, cfg.seed)):
            surf = scene.render(state)
            if (cfg.frame_w, cfg.frame_h) != surf.get_size():
                # Spec says 160x96; render to that natively, no resize needed.
                surf = pygame.transform.smoothscale(surf, (cfg.frame_w, cfg.frame_h))
            path = os.path.join(frames_dir, f"{i:06d}.jpg")
            pygame.image.save(surf, path)
            # JPEG quality: pygame doesn't expose quality; re-encode via Pillow
            # only if a user really cares. For the POC pygame's default JPEG is fine.

            fs.write(json.dumps({"i": i, "state": state.to_dict()}) + "\n")

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(1e-3, elapsed)
                print(f"  {i + 1}/{cfg.n}  ({rate:.1f} fps)", file=sys.stderr)

    print(f"wrote {cfg.n} frames + states to {out} in {time.time() - t_start:.1f}s")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Capture (state, frame) pairs for LucidPlay.")
    p.add_argument("--out", default=CaptureConfig.out_dir)
    p.add_argument("--n", type=int, default=CaptureConfig.n)
    p.add_argument("--seed", type=int, default=CaptureConfig.seed)
    p.add_argument("--frame-w", type=int, default=FRAME_W)
    p.add_argument("--frame-h", type=int, default=FRAME_H)
    return p


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = CaptureConfig(
        out_dir=args.out,
        n=args.n,
        seed=args.seed,
        frame_w=args.frame_w,
        frame_h=args.frame_h,
    )
    run(cfg)


if __name__ == "__main__":
    main()
