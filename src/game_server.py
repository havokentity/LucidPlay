"""WebSocket game server.

Runs a 60Hz physics loop, calls the neural renderer each tick, and streams
JPEG frames to one viewer. The pygame scene is *not* used here — at play time
the model is the renderer. We only share geometry constants for collision.
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol

from .config import ServeConfig
from .device import pick_device
from .infer import Renderer
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
    T_PERIOD,
    WorldState,
    make_state,
)


@dataclass
class _PlayerSim:
    px: float = LEVEL_W * 0.5
    py: float = LEVEL_H - GROUND_H
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = True
    facing: float = 1.0
    anim_phase: float = 0.0
    t: float = 0.0
    keys: Set[str] = field(default_factory=set)

    def step(self, dt: float) -> None:
        ax = 0.0
        if "left" in self.keys:
            ax -= 1.0
        if "right" in self.keys:
            ax += 1.0
        jump = "up" in self.keys

        self.vx = ax * MOVE_SPEED
        if abs(ax) > 0.1:
            self.facing = 1.0 if ax > 0 else -1.0
        if jump and self.on_ground:
            self.vy = JUMP_VELOCITY
            self.on_ground = False

        self.vy += GRAVITY * dt

        new_x = self.px + self.vx * dt
        new_y = self.py + self.vy * dt

        half = PLAYER_W * 0.5
        new_x = max(half, min(LEVEL_W - half, new_x))

        ground_y = LEVEL_H - GROUND_H
        next_on_ground = False
        if new_y >= ground_y:
            new_y = ground_y
            self.vy = 0.0
            next_on_ground = True

        if self.vy >= 0:
            foot_prev = self.py
            foot_new = new_y
            for (plx, ply, plw, _plh) in PLATFORMS:
                if new_x + half < plx or new_x - half > plx + plw:
                    continue
                if foot_prev <= ply and foot_new >= ply:
                    new_y = float(ply)
                    self.vy = 0.0
                    next_on_ground = True
                    break

        self.px = new_x
        self.py = new_y
        self.on_ground = next_on_ground

        if self.on_ground and abs(self.vx) > 1.0:
            self.anim_phase = (self.anim_phase + abs(self.vx) / MOVE_SPEED * dt * 4.0) % 1.0
        self.t = (self.t + dt) % T_PERIOD

    def to_state(self) -> WorldState:
        return make_state(
            player_x_px=self.px,
            player_y_px=self.py,
            vx=self.vx,
            vy=self.vy,
            on_ground=self.on_ground,
            facing=self.facing,
            anim_phase=self.anim_phase,
            t_seconds=self.t,
        )


class _StaticServer(threading.Thread):
    """Stdlib HTTP server in a background thread, serving src/viewer/."""

    def __init__(self, port: int, directory: str):
        super().__init__(daemon=True)
        self.port = port
        self.directory = directory
        self._httpd = None

    def run(self) -> None:
        directory = self.directory

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def log_message(self, fmt, *args):  # quiet
                pass

        with socketserver.TCPServer(("", self.port), Handler) as httpd:
            self._httpd = httpd
            print(f"[static] serving {directory} on http://localhost:{self.port}", file=sys.stderr)
            httpd.serve_forever()


async def _client_loop(ws: WebSocketServerProtocol, sim: _PlayerSim, renderer: Renderer, cfg: ServeConfig) -> None:
    print("[ws] client connected", file=sys.stderr)
    tick = 1.0 / cfg.physics_hz
    last = time.perf_counter()

    async def reader():
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("type") == "input":
                    keys = msg.get("keys", {})
                    sim.keys = {k for k, v in keys.items() if v}
        except websockets.ConnectionClosed:
            pass

    reader_task = asyncio.create_task(reader())

    try:
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now
            # Clamp dt to keep physics stable if a render took long.
            dt = min(dt, 1.0 / 20.0)
            sim.step(dt)

            state = sim.to_state()
            # Renderer call can be heavy on CPU/MPS — run in a thread so we
            # don't starve the event loop.
            jpeg = await asyncio.to_thread(renderer.render, state)
            try:
                await ws.send(jpeg)
            except websockets.ConnectionClosed:
                break

            if cfg.debug_state:
                try:
                    await ws.send(json.dumps({"type": "state", "state": state.to_dict()}))
                except websockets.ConnectionClosed:
                    break

            # Sleep until next tick. Latest-state-wins: if render overran,
            # skip the sleep and immediately render the next state.
            sleep_for = tick - (time.perf_counter() - now)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
    finally:
        reader_task.cancel()
        print("[ws] client disconnected", file=sys.stderr)


def _make_handler(renderer: Renderer, cfg: ServeConfig):
    async def handler(ws: WebSocketServerProtocol):
        # One player per connection. POC: refuse extras.
        sim = _PlayerSim()
        await _client_loop(ws, sim, renderer, cfg)
    return handler


async def _run_async(cfg: ServeConfig) -> None:
    device = pick_device()
    print(f"[infer] loading {cfg.ckpt} on {device}", file=sys.stderr)
    renderer = Renderer(cfg.ckpt, device=device, jpeg_quality=cfg.jpeg_quality)

    handler = _make_handler(renderer, cfg)
    print(f"[ws] listening on ws://localhost:{cfg.ws_port}", file=sys.stderr)
    async with websockets.serve(handler, "localhost", cfg.ws_port, max_size=2 ** 22):
        await asyncio.Future()  # run forever


def _build_argparser() -> argparse.ArgumentParser:
    d = ServeConfig()
    p = argparse.ArgumentParser(description="Serve the LucidPlay neural renderer.")
    p.add_argument("--ckpt", default=d.ckpt)
    p.add_argument("--port", dest="ws_port", type=int, default=d.ws_port)
    p.add_argument("--static-port", type=int, default=d.static_port)
    p.add_argument("--debug-state", action="store_true")
    p.add_argument("--no-static", action="store_true", help="Don't serve the viewer HTML.")
    p.add_argument("--jpeg-quality", type=int, default=d.jpeg_quality)
    return p


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = ServeConfig(
        ckpt=args.ckpt,
        ws_port=args.ws_port,
        static_port=args.static_port,
        debug_state=args.debug_state,
        jpeg_quality=args.jpeg_quality,
    )

    if not args.no_static:
        viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")
        _StaticServer(cfg.static_port, viewer_dir).start()
        print(f"[viewer] open http://localhost:{cfg.static_port}/", file=sys.stderr)

    try:
        asyncio.run(_run_async(cfg))
    except KeyboardInterrupt:
        print("\n[serve] bye", file=sys.stderr)


if __name__ == "__main__":
    main()
