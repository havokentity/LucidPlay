# LucidPlay

A tiny **neural game engine**. Game logic owns the world state; a learned PyTorch
model is the renderer. A WebSocket bridges Python physics and a `<canvas>` viewer
that takes keyboard input and shows the model's frames.

POC target: a 2D side-scroller. The same `ConditionalRenderer` interface will
later swap to a 3D first-person scene without architectural change. The pygame
scene exists **only to generate training data** — at play time the model is the
renderer.

```
state → ConditionalRenderer → image → WebSocket → <canvas>
                                                    ↑
                                                  keys
```

## What's in the box

| Path | Purpose |
|------|---------|
| `src/scene.py` | Headless pygame side-scroller (ground-truth, capture-only). |
| `src/capture.py` | Scripted-motion data dump: writes `frames/*.jpg` + `states.jsonl`. |
| `src/model.py`   | `ConditionalRenderer`: 8-float state → 320×192 RGB. |
| `src/train.py`   | L1 (+ MS-SSIM if available) trainer, AdamW, cosine LR, preview grids. |
| `src/game_server.py` | asyncio `websockets` server: 60Hz physics + neural render. |
| `src/viewer/index.html` | Single-file `<canvas>` client with keyboard input + FPS. |

## Quick start

### Mac (Apple Silicon, MPS)

```bash
python3.11 -m venv .venv     # or python3.12 — both work
source .venv/bin/activate
pip install -r requirements.txt
python scripts/capture.py --out data/sidescroller_v1 --n 20000
python scripts/train.py  --data data/sidescroller_v1 --out checkpoints/sidescroller_v1.pt --cache-data
python scripts/serve.py  --ckpt checkpoints/sidescroller_v1.pt
# open http://localhost:8000 in any browser
```

`--cache-data` preloads the dataset into unified memory and skips DataLoader
overhead per step — typically 10–20 % faster on MPS. The CUDA-only flags
(`--amp`, `--compile`) are no-ops here.

### Windows (NVIDIA CUDA)

```powershell
Set-ExecutionPolicy -Scope Process Bypass    # so activate.ps1 can run this session
py -3 -m venv .venv                          # Python 3.11+ (3.13 tested)
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts\capture.py --out data\sidescroller_v1 --n 20000
python scripts\train.py  --data data\sidescroller_v1 --out checkpoints\sidescroller_v1.pt --fast --batch-size 128
python scripts\serve.py  --ckpt checkpoints\sidescroller_v1.pt
```

> PyTorch on Windows installs from the CUDA wheel index. Use **cu128** —
> required for RTX 50-series (Blackwell, sm_120). On Python 3.13 you
> additionally need `pip install triton-windows` for `--compile`; cleanest is
> Python 3.12 where the wheel resolves automatically. The Mac default wheel
> already includes MPS. Pygame renders headlessly on both via offscreen
> `Surface` — no window pops up.
>
> Verify CUDA was picked up:
> ```powershell
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> ```

The `--fast` flag bundles `--cache-data --amp --compile`. At the current
**320×192** output resolution the cached dataset is ~15 GB and per-step
compute is ~10× heavier than the original 160×96 POC, so a 50k-step run on a
5090 lands around **~90 minutes** (vs ~hours on Apple Silicon MPS — Mac is
inference-only at this resolution). `--batch-size 128` is the recommended
sweet spot; bumping to 256 fits with ~10 GB headroom if you want faster
walk-clock per step.

### Smoke test on tiny data

```bash
python scripts/capture.py --out data/sidescroller_v1 --n 200 --seed 0
python scripts/train.py  --data data/sidescroller_v1 --steps 200
python scripts/serve.py  --ckpt checkpoints/sidescroller_v1.pt
```

200 frames at 200 steps won't look like much, but it proves the pipeline runs
end-to-end on whatever device you've got.

## Training perf flags

Defaults preserve the spec's fp32 / no-compile recipe and run on every device.
The flags below opt into bigger wins. CUDA-only flags warn and skip on MPS/CPU.

| Flag              | Effect                                                                                          | Devices       |
|-------------------|-------------------------------------------------------------------------------------------------|---------------|
| `--cache-data`    | Pre-decodes the whole dataset into device memory; skips DataLoader workers and per-step H2D.    | Any           |
| `--amp`           | bf16 autocast on the model forward + L1 loss. MS-SSIM stays fp32.                               | CUDA only     |
| `--compile`       | `torch.compile(model)`. First ~10 steps slow, then fused kernels.                               | CUDA only     |
| `--channels-last` | NHWC memory format on model + frame tensors. Can regress on MPS — measure before keeping.       | Any (opt-in)  |
| `--fast`          | Shortcut: `--cache-data --amp --compile` (amp/compile auto-disabled on non-CUDA).               | Any           |

TF32 matmul + `cudnn.benchmark` are turned on automatically when CUDA is the
active device — they're fp32-compatible and free at our fixed 320×192 shape.
MS-SSIM `win_size` is also picked dynamically from the smaller frame dim.

On a 5090: start with `--fast --batch-size 128`. The default batch (32) leaves
the card mostly idle. On M4 Max: `--cache-data` alone is the main win;
the cached dataset now costs ~15 GB of unified memory at the default 20 k
frames + 320×192 resolution, so drop `--cache-data` if your Mac doesn't have
the room.

## How it works

1. **Capture** (`scripts/capture.py`). A scripted random agent walks the level,
   occasionally jumps and idles. For each tick we render the ground-truth
   pygame scene to an offscreen surface and dump `(state_vec, frame.jpg)` pairs.
2. **Train** (`scripts/train.py`). A small conditional generator
   (~3M params, 5×3 base → 6 upsamples → 320×192) is fit on those pairs with
   `L1 + 0.1·MS-SSIM`. Preview grids of (pred / ground truth) land in
   `checkpoints/preview_*.jpg`.
3. **Play** (`scripts/serve.py`). The game server runs 60Hz physics, calls
   `renderer.render(state)` each tick, and streams JPEG bytes over a
   WebSocket. The viewer draws every binary message — latest state wins.

The state vector (`src/scene.py:WorldState`) is 8 floats normalized to
roughly `[-1, 1]`: position, velocity, on-ground flag, facing, animation phase,
and a global time channel for parallax/clouds.

## Controls

- `←` / `→` or `A` / `D` to move
- `Space` or `↑` or `W` to jump

## Acceptance for v1

- Capture ≥20k frames + matching `states.jsonl` in under 30 minutes.
- Train end-to-end without errors on CUDA **or** MPS **or** CPU.
- Open the viewer; the player visibly moves in response to keys; FPS ≥30 on a
  mid-range GPU.

Val L1 < ~0.05 on the held-out 5% split is the rough "looks like a side-scroller"
threshold. Eyeball the preview grids — this is a POC, not SOTA.

## Convergence is stochastic — don't be surprised by run-to-run differences

Training the renderer is not a deterministic process across machines, even with
seeds:

- **fp32 isn't bit-exact across devices.** CUDA, MPS, and CPU use different
  matmul / reduction kernels. Over 50k steps the tiny per-op differences
  accumulate into very different final weights.
- **DataLoader workers differ by OS** (default 4 on Windows, 2 on Mac), which
  changes batch ordering even with the same seed.
- **The loss landscape has many local minima.** Two runs with different
  initializations can land in basins whose visible quality differs noticeably.

In practice this means the same code + same data + same seed on Windows vs Mac
can produce one model that renders jumps perfectly and one that drops the
player mid-air. With v1's mode mix (~35% airborne frames) it's much less
common, but if your run looks broken on some state, **re-run with
`--seed 1`**. It's the standard cheap fix.

```powershell
python scripts\train.py --data data\sidescroller_v1 --out checkpoints\sidescroller_v1.pt --fast --seed 1
```

The `checkpoints/preview_*.jpg` grids saved every 1000 steps are the fastest
sanity check — each cell shows (prediction top / ground truth bottom). If the
top row is missing the player while the bottom shows it, that's the failure
mode this section is talking about.

## Out of scope for v1 (v2 ideas)

- Enemies, projectiles, full collision.
- Audio.
- Multi-client / shared world.
- Recurrent renderer (current frame conditioned on previous frame). Stateless
  `state → frame` is enough for the POC.
- 3D / first-person variant — the renderer interface stays the same; only the
  state schema and scene change.

## Notes on portability

- No `tiny-cuda-nn`, no custom CUDA kernels, no `gsplat`. Plain PyTorch ops.
- Default training is fp32 on every device — spec-compliant and portable.
  CUDA-only perf flags (`--amp`, `--compile`) are opt-in and gate themselves
  off on MPS / CPU so the same command works on every machine.
- Pygame **never** calls `pygame.display.set_mode`. `SDL_VIDEODRIVER=dummy` is
  forced in `src/scene.py` before `import pygame`.

### Re-training after a code update

If you've already trained on an older `main` and pulled new changes, the
capture distribution may have changed too. The simplest reset:

```powershell
# Windows
Remove-Item -Recurse -Force data\sidescroller_v1, checkpoints
python scripts\capture.py --out data\sidescroller_v1 --n 20000
python scripts\train.py  --data data\sidescroller_v1 --out checkpoints\sidescroller_v1.pt --fast --batch-size 128
```

```bash
# Mac
rm -rf data/sidescroller_v1 checkpoints
python scripts/capture.py --out data/sidescroller_v1 --n 20000
python scripts/train.py  --data data/sidescroller_v1 --out checkpoints/sidescroller_v1.pt --cache-data
```

## License

MIT — see [LICENSE](LICENSE).
