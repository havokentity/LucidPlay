"""Torch Dataset for (state, frame) pairs captured by src.capture."""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import FRAME_H, FRAME_W, STATE_DIM
from .scene import WorldState


def _load_states_jsonl(path: str) -> List[List[float]]:
    states: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = rec["state"]
            states.append([
                s["player_x"], s["player_y"], s["vx"], s["vy"],
                s["on_ground"], s["facing"], s["anim_phase"], s["t"],
            ])
    return states


class FrameStateDataset(Dataset):
    """Returns `(state[STATE_DIM], image[3, FRAME_H, FRAME_W])` tensors in [0, 1].

    Split: deterministic by line index. The last `val_frac` of the dataset is
    used for validation (mirrors the temporal/scripted order — fine for POC).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_frac: float = 0.05,
        frame_w: int = FRAME_W,
        frame_h: int = FRAME_H,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.root = root
        self.frame_w = frame_w
        self.frame_h = frame_h

        all_states = _load_states_jsonl(os.path.join(root, "states.jsonl"))
        n = len(all_states)
        if n == 0:
            raise RuntimeError(f"No states found in {root}/states.jsonl")
        val_n = max(1, int(n * val_frac))
        train_n = n - val_n
        if split == "train":
            self._indices = list(range(0, train_n))
        else:
            self._indices = list(range(train_n, n))
        self._states = [all_states[i] for i in self._indices]
        self._frames_dir = os.path.join(root, "frames")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        global_i = self._indices[idx]
        frame_path = os.path.join(self._frames_dir, f"{global_i:06d}.jpg")
        img = Image.open(frame_path).convert("RGB")
        if img.size != (self.frame_w, self.frame_h):
            img = img.resize((self.frame_w, self.frame_h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        tens = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)

        state_vec = torch.tensor(self._states[idx], dtype=torch.float32)
        assert state_vec.numel() == STATE_DIM, f"state dim mismatch: {state_vec.numel()} != {STATE_DIM}"
        return state_vec, tens
