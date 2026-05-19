"""ConditionalRenderer: state vector -> RGB image.

Deterministic conditional generator. No noise input. fp32, plain eager.
Per spec §5: 5x3 base feature at 256ch, five upsamples → 160x96 RGB.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import BASE_H, BASE_W, FRAME_H, FRAME_W, STATE_DIM, UPSAMPLE_CHANNELS


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class ConditionalRenderer(nn.Module):
    """state (B, STATE_DIM) → image (B, 3, FRAME_H, FRAME_W) in [0, 1]."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        base_w: int = BASE_W,
        base_h: int = BASE_H,
        channels: tuple = UPSAMPLE_CHANNELS,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("Need at least one upsample stage.")
        c0 = channels[0]

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(),
            nn.Linear(128, 256),       nn.SiLU(),
            nn.Linear(256, 256),
        )
        # z (B, 256) → base feature (B, c0, base_h, base_w)
        self.base_w = base_w
        self.base_h = base_h
        self.proj = nn.Linear(256, c0 * base_h * base_w)

        blocks = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(_UpBlock(in_ch, out_ch))
        self.blocks = nn.ModuleList(blocks)
        n_up = len(blocks)
        expected_h = base_h * (2 ** n_up)
        expected_w = base_w * (2 ** n_up)
        if (expected_h, expected_w) != (FRAME_H, FRAME_W):
            raise ValueError(
                f"Channel ladder produces {expected_w}x{expected_h}, "
                f"but config asks for {FRAME_W}x{FRAME_H}."
            )

        self.to_rgb = nn.Conv2d(channels[-1], 3, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        z = self.state_mlp(state)
        x = self.proj(z).view(z.size(0), -1, self.base_h, self.base_w)
        for blk in self.blocks:
            x = blk(x)
        x = self.to_rgb(x)
        return torch.sigmoid(x)

    @torch.inference_mode()
    def render_one(self, state_vec: torch.Tensor) -> torch.Tensor:
        """Convenience for inference: takes (STATE_DIM,) → (3, H, W)."""
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        out = self.forward(state_vec)
        return out.squeeze(0)


def num_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
