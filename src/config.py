from dataclasses import dataclass, field
import sys

FRAME_W = 320
FRAME_H = 192
FRAME_C = 3
STATE_DIM = 8

# Seven channels = six upsample stages: 5×2^6 = 320, 3×2^6 = 192.
UPSAMPLE_CHANNELS = (256, 192, 160, 128, 96, 64, 32)
BASE_W = 5
BASE_H = 3


def _default_workers() -> int:
    return 4 if sys.platform.startswith("win") else 2


@dataclass
class CaptureConfig:
    out_dir: str = "data/sidescroller_v1"
    n: int = 20000
    seed: int = 0
    jpeg_quality: int = 90
    frame_w: int = FRAME_W
    frame_h: int = FRAME_H


@dataclass
class TrainConfig:
    data_dir: str = "data/sidescroller_v1"
    out_ckpt: str = "checkpoints/sidescroller_v1.pt"
    steps: int = 50000
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    val_frac: float = 0.05
    log_every: int = 100
    val_every: int = 1000
    ckpt_every: int = 5000
    preview_every: int = 1000
    num_workers: int = field(default_factory=_default_workers)
    msssim_weight: float = 0.1
    seed: int = 0
    resume: str = ""
    # Perf knobs. Defaults preserve spec-compliant fp32 behavior on every device.
    cache_data: bool = False        # pre-decode entire dataset into device memory; bypasses DataLoader workers
    amp: bool = False               # bf16 autocast on forward+loss. CUDA only; ignored on MPS/CPU.
    torch_compile: bool = False     # torch.compile the model. CUDA only; ignored on MPS/CPU.
    channels_last: bool = False     # NHWC memory format. Opt-in; can regress on MPS.


@dataclass
class ServeConfig:
    ckpt: str = "checkpoints/sidescroller_v1.pt"
    ws_port: int = 8765
    static_port: int = 8000
    debug_state: bool = False
    tick_hz: int = 60
    jpeg_quality: int = 85
