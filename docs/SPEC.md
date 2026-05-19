# LucidPlay — POC Spec & Claude Code Handoff

A lightweight "neural game engine" POC. Game logic owns the world state; a learned neural renderer turns state into pixels. A WebSocket bridges them; a browser viewer takes keyboard input and displays the model's frames.

Target: a 2D side-scroller first. Same renderer class will later swap to a 3D first-person scene without architectural change.

---

## 1. High-level architecture

```
┌──────────────────────┐    state    ┌─────────────────────┐
│  Browser viewer      │ ──────────► │  Game server (WS)   │
│  - <canvas>          │   keys      │  - holds world      │
│  - keydown/keyup     │             │  - physics @60Hz    │
│  - draws PNG frames  │ ◄────────── │  - calls renderer   │
└──────────────────────┘    frame    │  - emits PNG/JPEG   │
                                     └──────────┬──────────┘
                                                │  (in-process)
                                                ▼
                                     ┌─────────────────────┐
                                     │  Neural renderer    │
                                     │  state → image      │
                                     │  PyTorch model      │
                                     └─────────────────────┘

Offline:
   pygame scene ──► capture loop ──► (state, frame) pairs ──► train ──► checkpoint
```

Two phases:
1. **Offline**: a pygame scene renders the "ground truth" world. A capture loop sweeps the player through valid states and dumps `(state_vec, frame)` pairs to disk. A PyTorch training loop fits a conditional generator `f(state) → image`.
2. **Online**: the game server loads the checkpoint, runs physics, calls `f(state)` each tick, streams frames over WebSocket. The viewer renders frames and forwards input.

The pygame scene is **only used for data generation**. At play-time the model is the renderer. This is the whole point of the POC.

---

## 2. Tech stack

| Layer            | Choice                              | Why                                                                        |
|------------------|-------------------------------------|----------------------------------------------------------------------------|
| Language         | Python 3.10+                        | Single language across data/train/serve. Cross-platform.                   |
| Scene + capture  | `pygame` (offscreen `Surface`)      | Pure Python, headless capable, identical on Mac + Windows.                 |
| ML framework     | PyTorch ≥ 2.2                       | Native MPS (Apple Silicon) + CUDA. No CUDA-only ops in the model.          |
| WebSocket server | `websockets` (asyncio)              | Tiny, no framework, plays nicely with an asyncio physics loop.             |
| Static viewer    | `http.server` (stdlib)              | Serves one HTML file. No build step.                                       |
| Viewer           | Plain HTML + JS + `<canvas>`        | Zero deps. Opens in any browser.                                           |
| Image transport  | JPEG bytes over WS binary frames    | Smaller than PNG, fine for POC quality.                                    |

Avoid: tiny-cuda-nn, gsplat custom CUDA kernels, anything that won't build on macOS.

---

## 3. Repo layout

```
lucidplay/
├── README.md
├── pyproject.toml                 # or requirements.txt — pick one
├── .gitignore
├── data/                          # generated; gitignored
│   └── sidescroller_v1/
│       ├── frames/000000.jpg ...
│       └── states.jsonl           # one JSON per line, matches frame index
├── checkpoints/                   # gitignored
│   └── sidescroller_v1.pt
├── src/
│   ├── __init__.py
│   ├── device.py                  # pick cuda / mps / cpu
│   ├── config.py                  # dataclasses for run config
│   ├── scene.py                   # pygame side-scroller world + state schema
│   ├── capture.py                 # CLI: dump (state, frame) pairs
│   ├── dataset.py                 # torch Dataset reading data/<name>/
│   ├── model.py                   # ConditionalRenderer (state vec → image)
│   ├── train.py                   # training loop
│   ├── infer.py                   # load checkpoint, expose render(state) -> bytes
│   ├── game_server.py             # asyncio WS server: physics + render
│   └── viewer/
│       └── index.html             # canvas + WS client + input
└── scripts/
    ├── capture.py                 # thin wrapper that calls src.capture
    ├── train.py                   # thin wrapper that calls src.train
    └── serve.py                   # starts game_server + static file server
```

`scripts/*.py` are just CLI entry points so users don't need `python -m src.foo`.

---

## 4. State schema (side-scroller v1)

The state vector is the model's only conditioning input. Keep it small, normalized to roughly `[-1, 1]`, and stable.

```python
# src/scene.py
@dataclass
class WorldState:
    player_x: float         # world units, normalized to [-1, 1] over level width
    player_y: float         # normalized over level height
    vx: float               # normalized
    vy: float
    on_ground: float        # 0.0 / 1.0
    facing: float           # -1.0 left, +1.0 right
    anim_phase: float       # [0, 1), cycles while running
    t: float                # global time mod some period, for parallax/clouds
```

Total: **8 floats**. Serialize to JSON as a dict; the model concatenates them in fixed order via `WorldState.to_vec()`.

Frame resolution: **160 × 96 RGB** (5:3-ish, side-scroller feel, small enough to train fast). Make it a config constant.

Level: 1 horizontal level, ~4 screens wide. Hand-authored in code: ground tiles, 2–3 platforms, a parallax mountain layer, a parallax cloud layer that drifts with `t`. Player is a simple animated rectangle/blob (color shift on `anim_phase`). No enemies in v1.

---

## 5. Model — `ConditionalRenderer`

A small conditional generator. Takes the state vector, produces an image. No noise input (deterministic renderer).

```
state (8) ──► MLP ──► z (256) ──► Linear ──► 4x4x256 feature
                                              │
                                              ▼  upsample blocks (each: Upsample x2 + Conv3x3 + GroupNorm + SiLU)
                                              4x4  → 8x8 → 16x16 (24x12 cropped later? — see below)
                                              ...
                                              ▼
                                              Conv1x1 → 3 channels → sigmoid → RGB
```

Concrete spatial path for 160×96 output: build a 5×3 base feature, upsample five times → 160×96.

- Base spatial: **5 × 3** at 256 channels (FC reshape from `z`).
- Upsample blocks (channels): 256 → 192 → 128 → 96 → 64 → 32.
- Each block: `nn.Upsample(scale_factor=2, mode='nearest')` → `Conv2d(in, out, 3, padding=1)` → `GroupNorm(8, out)` → `SiLU()` → `Conv2d(out, out, 3, padding=1)` → `GroupNorm(8, out)` → `SiLU()`.
- Final: `Conv2d(32, 3, 1)` → `sigmoid`.
- Final size: 5·2⁵ × 3·2⁵ = 160 × 96. ✓

State MLP: `Linear(8, 128) → SiLU → Linear(128, 256) → SiLU → Linear(256, 256)`.

Parameters: rough estimate ≈ 2–4M. Trains fast on a single GPU; runs at >60Hz on inference.

Loss: `L1 + 0.1 * MS-SSIM` (use `pytorch-msssim`). If that's a hassle to install, fall back to pure L1 for v1 — call it out in the README.

Optimizer: `AdamW`, lr `2e-4`, cosine decay. Batch 32. ~50k steps. Early stop on val L1.

---

## 6. Device selection

```python
# src/device.py
import torch

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Do **not** call `.half()` or use `torch.compile` in v1 — MPS support is patchy. Stick to fp32, plain eager mode. Once it works, the user can add `torch.compile` on CUDA only via a flag.

Pin memory only when device is `cuda`. Use `num_workers=2` on macOS, `num_workers=4` on Windows — make it a config knob.

---

## 7. Data capture — `src/capture.py`

CLI:
```
python scripts/capture.py --out data/sidescroller_v1 --n 20000 --seed 0
```

Steps:
1. Build the scene (`scene.build()`).
2. Generate `n` states by simulating random scripted motion: random horizontal targets, occasional jumps, occasional idles. This covers state space better than uniform random.
3. For each state: set the scene to that state, render to an offscreen `pygame.Surface`, resize to 160×96, save as `frames/{idx:06d}.jpg` (quality 90).
4. Append `{"i": idx, "state": {...}}` to `states.jsonl`.
5. Print progress every 500 frames.

Important: rendering must be **headless** — use `pygame.Surface((W, H))` without ever calling `pygame.display.set_mode`. This is the path that works on Windows + macOS without a window popping up.

Split 95/5 train/val via `states.jsonl` line indices; the dataset class reads a `split` arg.

---

## 8. Training — `src/train.py`

CLI:
```
python scripts/train.py --data data/sidescroller_v1 --out checkpoints/sidescroller_v1.pt --steps 50000
```

- Standard PyTorch loop. Log loss every 100 steps, validate every 1000.
- Save checkpoint every 5000 steps and on best val.
- Every 1000 steps also write a 4x4 grid of (state → predicted, ground truth) pairs to `checkpoints/preview_{step}.jpg` for sanity.
- Resume from `--resume` if given.
- All hyperparams live in `src/config.py` as a dataclass; CLI args override.

Acceptance for v1: val L1 < 0.05 on a held-out 5% split (eyeball — the user wants a POC, not SOTA).

---

## 9. Inference — `src/infer.py`

```python
class Renderer:
    def __init__(self, ckpt_path: str, device: torch.device): ...
    @torch.inference_mode()
    def render(self, state: WorldState) -> bytes:
        """Returns JPEG bytes."""
```

Use `torch.inference_mode()`, fp32, move state to device, run model, denormalize, convert to `uint8`, encode JPEG via Pillow. Target: <16ms per call on a mid-range GPU; <50ms on Apple Silicon.

---

## 10. Game server — `src/game_server.py`

`asyncio` + `websockets`. One client at a time is fine for POC.

Protocol (JSON for control, binary for frames):

Client → server (text, JSON):
```json
{"type": "input", "keys": {"left": true, "right": false, "up": false}}
```

Server → client:
- Binary frame: raw JPEG bytes. Viewer treats every binary message as the latest frame.
- Optional text `{"type": "state", "state": {...}}` for debugging (toggle with `--debug-state`).

Loop:
- 60Hz physics tick (`asyncio.sleep(1/60)` minus elapsed).
- Apply current input to player (move left/right, jump if `on_ground`).
- Compute new `WorldState`.
- `renderer.render(state)` → JPEG bytes → `await ws.send(bytes)`.
- If render is slower than 1/60, drop physics ticks rather than queue frames (latest-state wins).

Entry point:
```
python scripts/serve.py --ckpt checkpoints/sidescroller_v1.pt --port 8765 --static-port 8000
```

Starts both the WS server (`8765`) and a stdlib HTTP server (`8000`) for `src/viewer/`.

---

## 11. Viewer — `src/viewer/index.html`

One file. No build step. Layout:

- `<canvas>` sized to 160×96, CSS-scaled to fill window (image-rendering: pixelated).
- Open `ws://localhost:8765`, set `binaryType = "arraybuffer"`.
- On binary message: create a `Blob`, `createImageBitmap`, draw to canvas. Keep a single bitmap rolling.
- Keyboard: track `left/right/up` via keydown/keyup, send the **set** of currently-held keys whenever it changes (not every frame).
- Show FPS in a corner.

Controls: ←/→ or A/D to move, Space or ↑ to jump.

---

## 12. Cross-platform notes (README)

The README must include:

**Mac (Apple Silicon)**:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/capture.py --out data/sidescroller_v1 --n 20000
python scripts/train.py --data data/sidescroller_v1 --out checkpoints/sidescroller_v1.pt
python scripts/serve.py --ckpt checkpoints/sidescroller_v1.pt
# open http://localhost:8000 in browser
```

**Windows (CUDA)**:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts\capture.py --out data\sidescroller_v1 --n 20000
python scripts\train.py --data data\sidescroller_v1 --out checkpoints\sidescroller_v1.pt
python scripts\serve.py --ckpt checkpoints\sidescroller_v1.pt
```

Call out: PyTorch on Windows installs from the CUDA index URL; on Mac the default wheel includes MPS. Pygame works headless on both via offscreen Surfaces.

---

## 13. requirements.txt

```
torch>=2.2
numpy
pillow
pygame>=2.5
websockets>=12
pytorch-msssim     # optional; train.py degrades to pure L1 if import fails
tqdm
```

No `tiny-cuda-nn`, no `nerfacc`, no `gsplat`, no `xformers`.

---

## 14. Out of scope for v1 (capture as TODOs in README)

- Enemies, projectiles, collisions beyond ground/platforms.
- Audio.
- Multi-client.
- Recurrent renderer (current frame conditioned on previous frame). Stateless `state → frame` is fine for the POC.
- 3D / FPS variant — but keep the renderer interface generic so swapping in a new state schema + new scene is the only change.

---

## 15. Acceptance criteria

The POC is "done" when, on both Mac and Windows:
1. `scripts/capture.py` produces ≥20k frames + matching `states.jsonl` in under 30 minutes.
2. `scripts/train.py` runs end-to-end without errors on the picked device, writes checkpoints + preview grids.
3. After training, `scripts/serve.py` runs; opening the viewer shows a recognizable side-scroller, the player visibly moves in response to keys, and FPS is ≥30 on a mid-range GPU.
4. No CUDA-only code paths; the same commands run on Mac (MPS) with at most a perf hit.

---

# 16. Claude Code handoff prompt

Copy everything below this line into Claude Code. It is self-contained; it references this spec by file.

---

> **You are building a Python POC called "LucidPlay" — a tiny neural game engine where game logic runs in Python and the renderer is a learned PyTorch model. The full spec is in `lucidplay-spec.md` (sections 1–15 above this prompt). Read it first; treat it as authoritative.**
>
> **Constraints:**
> - Cross-platform: must run on macOS (Apple Silicon, PyTorch MPS) and Windows (NVIDIA CUDA). No CUDA-only ops. No `tiny-cuda-nn`, no custom CUDA kernels.
> - Python 3.10+. PyTorch ≥ 2.2. fp32 only in v1.
> - Pygame must render headless via offscreen `Surface` — never call `pygame.display.set_mode` in capture or training.
> - Frame size: 160 × 96 RGB. State vector: 8 floats per the schema in spec §4.
>
> **Build order (do not skip ahead):**
>
> 1. **Scaffold the repo** per spec §3. Create `pyproject.toml` (or `requirements.txt` — pick one and stick with it), `.gitignore` (ignore `data/`, `checkpoints/`, `.venv/`, `__pycache__`), empty `README.md` placeholder.
> 2. **`src/device.py`** — implement `pick_device()` per §6.
> 3. **`src/config.py`** — `@dataclass` with frame size, state dim, channels-per-block, lr, batch, steps, num_workers. One dataclass for capture, one for train.
> 4. **`src/scene.py`** — pygame headless side-scroller per §4. Implement `WorldState` dataclass with `to_vec()` and `from_vec()`. Implement `Scene` class with `render(state) -> pygame.Surface` (offscreen). Author the level in code: ground row, 2–3 platforms, parallax mountains + clouds tied to `state.t`, player rectangle whose color shifts with `anim_phase`. No display.
> 5. **`src/capture.py`** — per §7. Scripted-motion state generator (random targets, occasional jumps). Save JPEGs + `states.jsonl`. Print progress.
> 6. **`src/dataset.py`** — `torch.utils.data.Dataset` reading `data/<name>/`. Args: `root`, `split` ("train"|"val"), `val_frac=0.05`. Returns `(state_tensor[8], image_tensor[3,96,160])` in `[0,1]`.
> 7. **`src/model.py`** — `ConditionalRenderer` per §5. Exactly the block structure and channel ladder in the spec.
> 8. **`src/train.py`** — per §8. Loss = `L1 + 0.1 * MS-SSIM` if import succeeds, else pure L1 with a printed warning. AdamW, cosine schedule, checkpointing, preview grid every 1000 steps.
> 9. **`src/infer.py`** — `Renderer` class per §9. Returns JPEG bytes.
> 10. **`src/game_server.py`** — `asyncio` + `websockets` per §10. 60Hz physics, latest-state-wins rendering. Simple player physics: horizontal velocity from input, gravity, jump impulse when `on_ground`. Collide against the same ground/platforms used in the scene (factor that geometry into a shared module if cleanest).
> 11. **`src/viewer/index.html`** — single-file viewer per §11.
> 12. **`scripts/capture.py`, `scripts/train.py`, `scripts/serve.py`** — thin wrappers.
> 13. **`README.md`** — quick-start for Mac + Windows per §12, plus a "what this is" paragraph.
>
> **Verify before finishing:**
> - `python -c "from src.scene import Scene, WorldState; s=Scene(); surf=s.render(WorldState(0,0,0,0,1,1,0,0)); print(surf.get_size())"` prints `(160, 96)` (or whatever you set — match spec).
> - `python scripts/capture.py --out data/sidescroller_v1 --n 200 --seed 0` produces 200 jpgs and a 200-line `states.jsonl`.
> - `python scripts/train.py --data data/sidescroller_v1 --steps 200` completes without error on whatever device is available and writes a checkpoint.
> - `python scripts/serve.py --ckpt checkpoints/sidescroller_v1.pt` boots both servers and accepts a WS connection (test with a one-line `websockets` client).
>
> **Do NOT:**
> - Add multiplayer, audio, enemies, or a recurrent frame predictor — those are v2.
> - Pull in extra ML libraries beyond what's in spec §13.
> - Reach for `torch.compile`, fp16, or AOT graph capture in v1.
> - Use `pygame.display` for capture or training paths.
>
> **When done, summarize:**
> 1. Files created (paths only).
> 2. Any deviations from the spec and why.
> 3. The exact commands the user should run on Mac and Windows, copy-pasteable.
