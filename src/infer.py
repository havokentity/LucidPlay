"""Inference wrapper: load checkpoint, render(state) -> JPEG bytes."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .config import FRAME_H, FRAME_W
from .model import ConditionalRenderer
from .scene import WorldState


class Renderer:
    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None, jpeg_quality: int = 85):
        if device is None:
            from .device import pick_device
            device = pick_device()
        self.device = device
        self.jpeg_quality = jpeg_quality
        self.model = ConditionalRenderer().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        # Warm-up forward to avoid first-call latency.
        dummy = torch.zeros(1, 8, device=device)
        with torch.inference_mode():
            self.model(dummy)

    @torch.inference_mode()
    def render(self, state: WorldState) -> bytes:
        vec = torch.tensor([state.to_vec()], dtype=torch.float32, device=self.device)
        out = self.model(vec)[0].clamp(0, 1)
        # (3, H, W) → (H, W, 3) uint8
        arr = (out.cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()
