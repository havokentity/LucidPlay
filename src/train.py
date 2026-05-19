"""Training loop for ConditionalRenderer."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import Optional

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


def _save_preview(model: ConditionalRenderer, val_loader: DataLoader, device: torch.device, out_path: str) -> None:
    from PIL import Image  # local to avoid hard dep at top
    model.eval()
    states, gts = next(iter(val_loader))
    states = states[:16].to(device)
    gts = gts[:16]
    with torch.inference_mode():
        preds = model(states).cpu().clamp(0, 1)
    n = preds.size(0)
    # 4x4 grid of (pred above, gt below) — stack vertically per cell.
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


def _validate(model: ConditionalRenderer, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.inference_mode():
        for states, gts in val_loader:
            states = states.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            preds = model(states)
            total += F.l1_loss(preds, gts, reduction="sum").item()
            n += gts.numel()
    model.train()
    return total / max(1, n)


def run(cfg: TrainConfig) -> None:
    _set_seed(cfg.seed)
    device = pick_device()
    pin = pin_memory_for(device)
    print(f"[train] device={device}  pin_memory={pin}", file=sys.stderr)

    train_ds = FrameStateDataset(cfg.data_dir, split="train", val_frac=cfg.val_frac)
    val_ds = FrameStateDataset(cfg.data_dir, split="val", val_frac=cfg.val_frac)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}", file=sys.stderr)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(0, cfg.num_workers // 2), pin_memory=pin,
    )

    model = ConditionalRenderer().to(device)
    print(f"[train] params={num_params(model):,}", file=sys.stderr)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

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
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[train] resumed from {cfg.resume} at step {start_step}", file=sys.stderr)

    os.makedirs(os.path.dirname(cfg.out_ckpt) or ".", exist_ok=True)

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

        # LR schedule (cosine with linear warmup).
        lr_now = _cosine_lr(step, cfg.steps, cfg.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        preds = model(states)
        l1 = F.l1_loss(preds, gts)
        if ms_ssim_module is not None:
            # MS_SSIM returns similarity ∈ [0,1]; loss = 1 - similarity.
            ssim_loss = 1.0 - ms_ssim_module(preds, gts)
            loss = l1 + cfg.msssim_weight * ssim_loss
        else:
            loss = l1

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running += float(l1.item())
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
                _save_checkpoint(cfg.out_ckpt + ".best", model, opt, step, best_val)

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
            _save_checkpoint(cfg.out_ckpt, model, opt, step, best_val)

    _save_checkpoint(cfg.out_ckpt, model, opt, step, best_val)
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
    )
    run(cfg)


if __name__ == "__main__":
    main()
