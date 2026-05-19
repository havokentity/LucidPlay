"""WebSocket game server.

Runs a configurable tick loop (default 60 Hz, override with --tick-hz) —
each tick advances simple physics, calls the neural renderer, and streams a
JPEG frame to one viewer.
The pygame scene is *not* used here — at play time the model is the renderer.
We only share geometry constants for collision.
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
    tick = 1.0 / cfg.tick_hz
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

    # Rolling 1-second timing window. One log line per second so cap behavior
    # is observable without spamming.
    render_total = 0.0
    send_total = 0.0
    sleep_req_total = 0.0
    sleep_actual_total = 0.0
    tick_count = 0
    t_window_start = time.perf_counter()

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
            t_render_start = time.perf_counter()
            jpeg = await asyncio.to_thread(renderer.render, state)
            t_after_render = time.perf_counter()
            render_total += t_after_render - t_render_start
            try:
                await ws.send(jpeg)
            except websockets.ConnectionClosed:
                break
            send_total += time.perf_counter() - t_after_render

            if cfg.debug_state:
                try:
                    await ws.send(json.dumps({"type": "state", "state": state.to_dict()}))
                except websockets.ConnectionClosed:
                    break

            # Wait until next tick boundary. Latest-state-wins: if render
            # overran, fall through immediately.
            #
            # On Windows + ProactorEventLoop, asyncio.sleep(x) ignores x for
            # sub-15 ms requests unless system timer resolution has been
            # raised (timeBeginPeriod in main()). The yield-loop after
            # asyncio.sleep is a safety net that holds the cap even without
            # that — at worst we yield through the event loop a few times,
            # which lets the reader task run too.
            target = now + tick
            sleep_for = target - time.perf_counter()
            sleep_req_total += max(0.0, sleep_for)
            t_sleep_start = time.perf_counter()
            if sleep_for > 0.001:
                await asyncio.sleep(sleep_for)
            while time.perf_counter() < target:
                await asyncio.sleep(0)
            sleep_actual_total += time.perf_counter() - t_sleep_start

            tick_count += 1
            elapsed = time.perf_counter() - t_window_start
            if elapsed >= 1.0:
                ms = 1000.0 / tick_count
                print(
                    f"[timing] {tick_count:4d} ticks/s  "
                    f"render {render_total * ms:5.2f}ms  "
                    f"send {send_total * ms:5.2f}ms  "
                    f"sleep_req {sleep_req_total * ms:5.2f}ms  "
                    f"sleep_actual {sleep_actual_total * ms:5.2f}ms",
                    file=sys.stderr,
                    flush=True,
                )
                render_total = send_total = sleep_req_total = sleep_actual_total = 0.0
                tick_count = 0
                t_window_start = time.perf_counter()
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
    p.add_argument("--tick-hz", type=int, default=d.tick_hz,
                   help="Loop target rate in Hz — one physics step + one render per tick. Default 60.")
    return p


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    if args.tick_hz <= 0:
        print("error: --tick-hz must be > 0", file=sys.stderr)
        sys.exit(2)
    cfg = ServeConfig(
        ckpt=args.ckpt,
        ws_port=args.ws_port,
        static_port=args.static_port,
        debug_state=args.debug_state,
        jpeg_quality=args.jpeg_quality,
        tick_hz=args.tick_hz,
    )

    # On Windows the default kernel timer resolution (~15 ms) makes
    # asyncio.sleep ignore sub-15 ms requests, which lets the tick rate cap
    # leak. timeBeginPeriod(1) requests 1 ms resolution for the duration of
    # this process; the loop's yield-safety-net handles the residual.
    winmm = None
    if sys.platform == "win32":
        try:
            import ctypes
            winmm = ctypes.windll.winmm
            winmm.timeBeginPeriod(1)
        except Exception as exc:
            print(f"[serve] warning: couldn't raise Windows timer resolution ({exc})", file=sys.stderr)
            winmm = None

    try:
        if not args.no_static:
            viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")
            _StaticServer(cfg.static_port, viewer_dir).start()
            print(f"[viewer] open http://localhost:{cfg.static_port}/", file=sys.stderr)

        try:
            asyncio.run(_run_async(cfg))
        except KeyboardInterrupt:
            print("\n[serve] bye", file=sys.stderr)
    finally:
        if winmm is not None:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass


if __name__ == "__main__":
    main()
