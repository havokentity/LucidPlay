"""Training loop for ConditionalRenderer."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import FRAME_H, FRAME_W, TrainConfig
from .dataset import FrameStateDataset
from .device import pick_device, pin_memory_for
from .model import ConditionalRenderer, num_params


def _try_import_msssim():
    try:
        from pytorch_msssim import MS_SSIM  # type: ignore
        return MS_SSIM
    except Exception as exc:  # pragma: no cover
        print(f"[train] pytorch-msssim unavailable ({exc}); falling back to pure L1.", file=sys.stderr)
        return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int = 500) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _enable_cuda_perf() -> None:
    # TF32 + cuDNN autotune. fp32-compatible numerically; free perf at fixed shapes.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class _DeviceBatchIter:
    """Yields fixed-size batches over device-resident tensors. Used with --cache-data."""

    def __init__(self, states: torch.Tensor, frames: torch.Tensor, batch_size: int, shuffle: bool):
        assert states.size(0) == frames.size(0)
        self.states = states
        self.frames = frames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = states.size(0)

    def __len__(self) -> int:
        return self.n // self.batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.states.device)
        else:
            idx = torch.arange(self.n, device=self.states.device)
        for start in range(0, self.n - self.batch_size + 1, self.batch_size):
            sl = idx[start:start + self.batch_size]
            yield self.states[sl], self.frames[sl]


def _materialize(ds: FrameStateDataset, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    states = []
    frames = []
    for i in range(len(ds)):
        s, f = ds[i]
        states.append(s)
        frames.append(f)
    states_t = torch.stack(states).to(device, non_blocking=True)
    frames_t = torch.stack(frames).to(device, non_blocking=True)
    return states_t, frames_t


def _save_preview(model, val_loader, device: torch.device, out_path: str) -> None:
    from PIL import Image
    model.eval()
    states, gts = next(iter(val_loader))
    states = states[:16].to(device, non_blocking=True)
    gts = gts[:16].detach().cpu()
    with torch.inference_mode():
        preds = model(states).clamp(0, 1).detach().float().cpu()
    n = preds.size(0)
    rows, cols = 4, 4
    cell_w, cell_h = FRAME_W, FRAME_H * 2
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h))
    for i in range(min(n, rows * cols)):
        r, c = divmod(i, cols)
        pred_arr = (preds[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        gt_arr = (gts[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        cell = Image.new("RGB", (cell_w, cell_h))
        cell.paste(Image.fromarray(pred_arr), (0, 0))
        cell.paste(Image.fromarray(gt_arr), (0, FRAME_H))
        grid.paste(cell, (c * cell_w, r * cell_h))
    grid.save(out_path, quality=90)
    model.train()


def _validate(model, val_loader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.inference_mode():
        for states, gts in val_loader:
            states = states.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            preds = model(states).float()
            total += F.l1_loss(preds, gts.float(), reduction="sum").item()
            n += gts.numel()
    model.train()
    return total / max(1, n)


def run(cfg: TrainConfig) -> None:
    _set_seed(cfg.seed)
    device = pick_device()
    if device.type == "cuda":
        _enable_cuda_perf()
    pin = pin_memory_for(device)

    use_amp = cfg.amp and device.type == "cuda"
    use_compile = cfg.torch_compile and device.type == "cuda"
    if cfg.amp and not use_amp:
        print(f"[train] --amp ignored on device={device.type}; bf16 autocast is CUDA-only here.", file=sys.stderr)
    if cfg.torch_compile and not use_compile:
        print(f"[train] --compile ignored on device={device.type}; torch.compile path is CUDA-only here.", file=sys.stderr)

    print(
        f"[train] device={device}  pin_memory={pin}  cache={cfg.cache_data}  "
        f"amp={use_amp}  compile={use_compile}  channels_last={cfg.channels_last}",
        file=sys.stderr,
    )

    train_ds = FrameStateDataset(cfg.data_dir, split="train", val_frac=cfg.val_frac)
    val_ds = FrameStateDataset(cfg.data_dir, split="val", val_frac=cfg.val_frac)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}", file=sys.stderr)

    train_loader: Iterable
    val_loader: Iterable
    if cfg.cache_data:
        t0 = time.time()
        train_states, train_frames = _materialize(train_ds, device)
        val_states, val_frames = _materialize(val_ds, device)
        if cfg.channels_last:
            train_frames = train_frames.to(memory_format=torch.channels_last)
            val_frames = val_frames.to(memory_format=torch.channels_last)
        bytes_total = sum(
            t.element_size() * t.nelement()
            for t in (train_states, train_frames, val_states, val_frames)
        )
        print(
            f"[train] cached {len(train_ds) + len(val_ds)} samples on {device} "
            f"({bytes_total / 1e6:.1f} MB) in {time.time() - t0:.1f}s",
            file=sys.stderr,
        )
        train_loader = _DeviceBatchIter(train_states, train_frames, cfg.batch_size, shuffle=True)
        val_loader = _DeviceBatchIter(val_states, val_frames, cfg.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=pin, drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=max(0, cfg.num_workers // 2), pin_memory=pin,
        )

    raw_model = ConditionalRenderer().to(device)
    if cfg.channels_last:
        raw_model = raw_model.to(memory_format=torch.channels_last)
    print(f"[train] params={num_params(raw_model):,}", file=sys.stderr)

    opt = torch.optim.AdamW(raw_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    MS_SSIM_cls = _try_import_msssim()
    ms_ssim_module = None
    if MS_SSIM_cls is not None:
        # pytorch-msssim hardcodes a (win_size-1)*2^4 size check regardless of
        # how many `weights` you pass. With FRAME_H=96, win_size=11 needs >160
        # (fails) and win_size=7 needs >96 (fails — equal). win_size=5 needs
        # >64, which passes for our 160×96 frames.
        ms_ssim_module = MS_SSIM_cls(
            data_range=1.0, size_average=True, channel=3, win_size=5,
        ).to(device)

    start_step = 0
    best_val = float("inf")
    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[train] resumed from {cfg.resume} at step {start_step}", file=sys.stderr)

    # Compile after resume load so the saved state_dict keys match raw_model.
    model = torch.compile(raw_model) if use_compile else raw_model
    if use_compile:
        print("[train] torch.compile(model) enabled — first ~10 steps will be slow.", file=sys.stderr)

    os.makedirs(os.path.dirname(cfg.out_ckpt) or ".", exist_ok=True)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else nullcontext()
    )

    model.train()
    step = start_step
    iterator = iter(train_loader)
    t_start = time.time()
    running = 0.0
    running_n = 0

    while step < cfg.steps:
        try:
            states, gts = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            states, gts = next(iterator)

        states = states.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        if cfg.channels_last and gts.dim() == 4:
            gts = gts.to(memory_format=torch.channels_last)

        lr_now = _cosine_lr(step, cfg.steps, cfg.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        with amp_ctx:
            preds = model(states)
            l1 = F.l1_loss(preds, gts)
        if ms_ssim_module is not None:
            # MS-SSIM uses ops that don't reliably autocast; compute in fp32 explicitly.
            ssim_loss = 1.0 - ms_ssim_module(preds.float(), gts.float())
            loss = l1.float() + cfg.msssim_weight * ssim_loss
        else:
            loss = l1

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running += float(l1.detach().float().item())
        running_n += 1
        step += 1

        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            ips = (step - start_step) * cfg.batch_size / max(1e-3, elapsed)
            print(
                f"step {step:6d} | l1 {running / running_n:.4f} | lr {lr_now:.2e} | "
                f"{ips:5.0f} img/s | {elapsed:5.0f}s",
                file=sys.stderr,
            )
            running = 0.0
            running_n = 0

        if step % cfg.val_every == 0:
            val_l1 = _validate(model, val_loader, device)
            print(f"[val ] step {step:6d}  val_l1 {val_l1:.4f}", file=sys.stderr)
            if val_l1 < best_val:
                best_val = val_l1
                _save_checkpoint(cfg.out_ckpt + ".best", raw_model, opt, step, best_val)

        if step % cfg.preview_every == 0:
            preview_path = os.path.join(
                os.path.dirname(cfg.out_ckpt) or ".",
                f"preview_{step:06d}.jpg",
            )
            try:
                _save_preview(model, val_loader, device, preview_path)
            except Exception as exc:
                print(f"[train] preview failed: {exc}", file=sys.stderr)

        if step % cfg.ckpt_every == 0:
            _save_checkpoint(cfg.out_ckpt, raw_model, opt, step, best_val)

    _save_checkpoint(cfg.out_ckpt, raw_model, opt, step, best_val)
    print(f"[train] done. final step {step}  best_val_l1 {best_val:.4f}", file=sys.stderr)


def _save_checkpoint(path: str, model, opt, step: int, best_val: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "best_val": best_val,
            "frame_w": FRAME_W,
            "frame_h": FRAME_H,
        },
        path,
    )


def _build_argparser() -> argparse.ArgumentParser:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train LucidPlay's ConditionalRenderer.")
    p.add_argument("--data", dest="data_dir", default=d.data_dir)
    p.add_argument("--out", dest="out_ckpt", default=d.out_ckpt)
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--val-frac", type=float, default=d.val_frac)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--val-every", type=int, default=d.val_every)
    p.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    p.add_argument("--preview-every", type=int, default=d.preview_every)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--resume", default=d.resume)
    # Perf flags. Defaults preserve spec-compliant fp32 behavior.
    p.add_argument("--cache-data", action="store_true", default=d.cache_data,
                   help="Pre-decode entire dataset into device memory. Bypasses DataLoader workers.")
    p.add_argument("--amp", action="store_true", default=d.amp,
                   help="bf16 autocast on forward+L1 (CUDA only; ignored on MPS/CPU).")
    p.add_argument("--compile", dest="torch_compile", action="store_true", default=d.torch_compile,
                   help="torch.compile the model (CUDA only; ignored on MPS/CPU).")
    p.add_argument("--channels-last", action="store_true", default=d.channels_last,
                   help="NHWC memory format. Opt-in; can regress on MPS.")
    p.add_argument("--fast", action="store_true", default=False,
                   help="Shortcut: --cache-data + --amp + --compile (amp/compile auto-disabled on non-CUDA).")
    return p


def main(argv: Optional[list] = None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_ckpt=args.out_ckpt,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        log_every=args.log_every,
        val_every=args.val_every,
        ckpt_every=args.ckpt_every,
        preview_every=args.preview_every,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        cache_data=args.cache_data or args.fast,
        amp=args.amp or args.fast,
        torch_compile=args.torch_compile or args.fast,
        channels_last=args.channels_last,
    )
    run(cfg)


if __name__ == "__main__":
    main()
