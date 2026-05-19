"""Headless pygame side-scroller scene.

Used at capture time to produce ground-truth frames. At play time the neural
renderer replaces this entirely; the geometry constants below are still imported
by the game server for physics/collision.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple

# Force pygame headless — must be set before `import pygame`. Capture and
# training paths must never open a window.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from .config import FRAME_W, FRAME_H  # noqa: E402

# --- Level geometry (world coordinates, in pixels) --------------------------
LEVEL_W = FRAME_W * 4  # ~4 screens wide
LEVEL_H = FRAME_H      # one screen tall
GROUND_H = 16          # pixels of ground at the bottom
PLAYER_W = 10
PLAYER_H = 14

# Platforms: (x, y, w, h) where y is top edge (smaller y == higher on screen).
PLATFORMS: List[Tuple[int, int, int, int]] = [
    (140, FRAME_H - GROUND_H - 22, 60, 4),
    (300, FRAME_H - GROUND_H - 38, 70, 4),
    (470, FRAME_H - GROUND_H - 24, 60, 4),
]

# Physics constants used by the online game server. Capture also reuses them
# when scripting motion that should look natural.
GRAVITY = 900.0          # px/s^2
MOVE_SPEED = 80.0        # px/s
JUMP_VELOCITY = -260.0   # px/s (negative = up)
T_PERIOD = 8.0           # seconds; t wraps within this period


# --- World state ------------------------------------------------------------
@dataclass
class WorldState:
    """8-float conditioning vector. Normalized into roughly [-1, 1]."""
    player_x: float       # normalized over LEVEL_W
    player_y: float       # normalized over LEVEL_H
    vx: float             # normalized over MOVE_SPEED
    vy: float             # normalized over |JUMP_VELOCITY|
    on_ground: float      # 0.0 or 1.0
    facing: float         # -1.0 (left) or +1.0 (right)
    anim_phase: float     # [0, 1), cycles while running
    t: float              # normalized over T_PERIOD

    def to_vec(self) -> List[float]:
        return [
            self.player_x, self.player_y, self.vx, self.vy,
            self.on_ground, self.facing, self.anim_phase, self.t,
        ]

    @classmethod
    def from_vec(cls, v) -> "WorldState":
        return cls(
            player_x=float(v[0]), player_y=float(v[1]),
            vx=float(v[2]),       vy=float(v[3]),
            on_ground=float(v[4]), facing=float(v[5]),
            anim_phase=float(v[6]), t=float(v[7]),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# --- Normalization helpers --------------------------------------------------
def world_to_norm(px: float, py: float) -> Tuple[float, float]:
    """Pixel coords → normalized [-1, 1]."""
    nx = (px / LEVEL_W) * 2.0 - 1.0
    ny = (py / LEVEL_H) * 2.0 - 1.0
    return nx, ny


def norm_to_world(nx: float, ny: float) -> Tuple[float, float]:
    px = (nx + 1.0) * 0.5 * LEVEL_W
    py = (ny + 1.0) * 0.5 * LEVEL_H
    return px, py


def vel_to_norm(vx: float, vy: float) -> Tuple[float, float]:
    return vx / MOVE_SPEED, vy / abs(JUMP_VELOCITY)


def norm_to_vel(nvx: float, nvy: float) -> Tuple[float, float]:
    return nvx * MOVE_SPEED, nvy * abs(JUMP_VELOCITY)


def time_to_norm(t_seconds: float) -> float:
    return (t_seconds % T_PERIOD) / T_PERIOD


# --- Scene ------------------------------------------------------------------
class Scene:
    """Headless renderer of the ground-truth side-scroller.

    `render(state)` returns a `pygame.Surface` of size (FRAME_W, FRAME_H).
    Camera is anchored to the player horizontally, clamped to level bounds.
    """

    SKY_TOP = (110, 170, 230)
    SKY_BOT = (200, 220, 240)
    MOUNTAIN = (90, 110, 140)
    MOUNTAIN_NEAR = (70, 90, 120)
    CLOUD = (245, 245, 250)
    GROUND_TOP = (110, 80, 50)
    GROUND_BODY = (80, 55, 35)
    PLATFORM = (160, 110, 70)

    def __init__(self, frame_w: int = FRAME_W, frame_h: int = FRAME_H):
        pygame.init()
        self.frame_w = frame_w
        self.frame_h = frame_h
        # Persistent offscreen surface — never displayed.
        self._surf = pygame.Surface((frame_w, frame_h))
        self._sky = self._build_sky_surface(frame_w, frame_h)

    def _build_sky_surface(self, w: int, h: int) -> pygame.Surface:
        s = pygame.Surface((w, h))
        for y in range(h):
            a = y / max(1, h - 1)
            r = int(self.SKY_TOP[0] * (1 - a) + self.SKY_BOT[0] * a)
            g = int(self.SKY_TOP[1] * (1 - a) + self.SKY_BOT[1] * a)
            b = int(self.SKY_TOP[2] * (1 - a) + self.SKY_BOT[2] * a)
            pygame.draw.line(s, (r, g, b), (0, y), (w, y))
        return s

    def _camera_x(self, player_world_x: float) -> float:
        """Horizontal camera offset so player sits near screen center."""
        cam = player_world_x - self.frame_w * 0.5
        cam = max(0.0, min(cam, LEVEL_W - self.frame_w))
        return cam

    def render(self, state: WorldState) -> pygame.Surface:
        s = self._surf
        s.blit(self._sky, (0, 0))

        # Player world position
        px, py = norm_to_world(state.player_x, state.player_y)
        cam_x = self._camera_x(px)

        t = state.t * T_PERIOD  # back to seconds-like for drift

        # Parallax: far mountains (slow). Big triangles.
        self._draw_mountains(s, cam_x * 0.2, t, far=True)
        # Mid mountains
        self._draw_mountains(s, cam_x * 0.4, t, far=False)
        # Clouds drift independent of camera but also with t
        self._draw_clouds(s, cam_x * 0.3 + t * 6.0)

        # Ground
        ground_y = self.frame_h - GROUND_H
        pygame.draw.rect(s, self.GROUND_BODY, (0, ground_y, self.frame_w, GROUND_H))
        pygame.draw.rect(s, self.GROUND_TOP, (0, ground_y, self.frame_w, 3))

        # Platforms (world-space, then offset by camera)
        for (x, y, w, h) in PLATFORMS:
            sx = x - cam_x
            if sx + w < 0 or sx > self.frame_w:
                continue
            pygame.draw.rect(s, self.PLATFORM, (int(sx), y, w, h))
            pygame.draw.rect(s, self.GROUND_TOP, (int(sx), y, w, 1))

        # Player
        self._draw_player(s, px - cam_x, py, state)

        return s

    def _draw_mountains(self, s: pygame.Surface, cam: float, t: float, far: bool) -> None:
        color = self.MOUNTAIN if far else self.MOUNTAIN_NEAR
        base_y = self.frame_h - GROUND_H
        # Repeating triangle silhouette.
        spacing = 60 if far else 50
        height = 30 if far else 38
        # Slight vertical bob to imply distant atmosphere (helps the model learn `t`).
        bob = math.sin(t * 0.6 + (0.0 if far else 1.0)) * (1.0 if far else 0.5)
        offset = int(-cam) % spacing
        x = -offset - spacing
        while x < self.frame_w + spacing:
            tip = (x + spacing // 2, base_y - height + bob)
            left = (x, base_y)
            right = (x + spacing, base_y)
            pygame.draw.polygon(s, color, [left, tip, right])
            x += spacing

    def _draw_clouds(self, s: pygame.Surface, drift: float) -> None:
        # Three cloud "lanes" at different heights and speeds.
        lanes = [(12, 1.0, 70), (24, 0.7, 100), (8, 1.3, 130)]
        for (y, speed, spacing) in lanes:
            offset = int(drift * speed) % spacing
            x = -offset - spacing
            while x < self.frame_w + spacing:
                pygame.draw.ellipse(s, self.CLOUD, (x, y, 24, 8))
                pygame.draw.ellipse(s, self.CLOUD, (x + 10, y - 3, 18, 8))
                pygame.draw.ellipse(s, self.CLOUD, (x + 20, y, 20, 8))
                x += spacing

    def _draw_player(self, s: pygame.Surface, sx: float, sy: float, state: WorldState) -> None:
        # Body color shifts with anim_phase to give the model a clear signal.
        phase = state.anim_phase * 2.0 * math.pi
        r = int(220 + 20 * math.sin(phase))
        g = int(80 + 60 * math.sin(phase + 2.0))
        b = int(80 + 60 * math.sin(phase + 4.0))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        body = (r, g, b)

        # Body: feet at (sx, sy), top at (sx, sy - PLAYER_H).
        rect_x = int(sx - PLAYER_W * 0.5)
        rect_y = int(sy - PLAYER_H)
        pygame.draw.rect(s, body, (rect_x, rect_y, PLAYER_W, PLAYER_H))

        # Face direction marker: bright pixel on the leading edge.
        eye_x = rect_x + PLAYER_W - 2 if state.facing > 0 else rect_x + 1
        eye_y = rect_y + 3
        pygame.draw.rect(s, (255, 255, 255), (eye_x, eye_y, 2, 2))

        # Simple legs: alternate based on anim_phase when on ground.
        if state.on_ground > 0.5:
            step = math.sin(phase) > 0
            leg_y = rect_y + PLAYER_H
            if step:
                pygame.draw.rect(s, body, (rect_x + 1, leg_y, 3, 2))
            else:
                pygame.draw.rect(s, body, (rect_x + PLAYER_W - 4, leg_y, 3, 2))


def make_state(
    player_x_px: float,
    player_y_px: float,
    vx: float = 0.0,
    vy: float = 0.0,
    on_ground: bool = True,
    facing: float = 1.0,
    anim_phase: float = 0.0,
    t_seconds: float = 0.0,
) -> WorldState:
    """Convenience constructor: takes world-pixel coords, returns a normalized WorldState."""
    nx, ny = world_to_norm(player_x_px, player_y_px)
    nvx, nvy = vel_to_norm(vx, vy)
    return WorldState(
        player_x=nx, player_y=ny, vx=nvx, vy=nvy,
        on_ground=1.0 if on_ground else 0.0,
        facing=1.0 if facing >= 0 else -1.0,
        anim_phase=float(anim_phase) % 1.0,
        t=time_to_norm(t_seconds),
    )
