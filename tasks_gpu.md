# GPU handoff tasks for Codex

Goal: reproduce the canonical Landau-de Gennes elastic-relaxation benchmark on a
GPU machine, measure true CUDA inference speed, and regenerate the rebuttal
figure with CPU and GPU caption numbers that match stdout.

## 0. Start from the pushed repository

```bash
git clone https://github.com/GangulyArkava/PhysicsNeMo_LiquidCrystals.git
cd PhysicsNeMo_LiquidCrystals
```

If the repo already exists:

```bash
git pull origin main
```

Activate the local environment where NVIDIA PhysicsNeMo, PyTorch with CUDA, NumPy,
and Matplotlib are installed.

## 1. Verify the backend and CUDA

```bash
python - <<'PY'
import torch
import physicsnemo
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("physicsnemo import: ok")
PY
```

Stop if `cuda_available` is false. The rebuttal figure must not report GPU
speedup from a CPU-only run.

## 2. Use the canonical benchmark configuration

Use this exact physical/numerical setup unless intentionally retraining:

- grid: `64 x 64`
- time step: `dt = 2e-4`
- steps: `500`
- final time: `t = 0.1`
- initialization: `canonical`
- canonical mode: `(3, 2)`
- canonical amplitude: `0.8`
- parameters: `A = -0.1`, `C = 1.0`, `L = 0.02`, `gamma = 1.0`
- evaluation samples: `64`

Existing CPU-trained checkpoint:

```bash
checkpoints/lc64_canonical_cpu.pt
```

PDE-loss checkpoint, if present:

```bash
checkpoints/lc64_canonical_pde_cpu.pt
```

## 3. Reproduce CPU response-figure metrics

Run this once to establish the CPU number on the GPU workstation:

```bash
python scripts/plot_response_figure.py \
  --checkpoint checkpoints/lc64_canonical_cpu.pt \
  --fixed-canonical-params \
  --device cpu \
  --eval-samples 64 \
  --out figures/response_field_comparison_cpu.png
```

Record:

- `mean_relative_l2`
- `median_relative_l2`
- `single_trajectory_speedup_x`
- `spectral_time_s`
- `fno_time_s`
- `backend`

Expected sanity check: `mean_relative_l2` should be about `7e-3` to `8e-3`.
The backend must be `PhysicsNeMo FNO`.

## 4. Measure GPU inference speed

Run the same script on CUDA:

```bash
python scripts/plot_response_figure.py \
  --checkpoint checkpoints/lc64_canonical_cpu.pt \
  --fixed-canonical-params \
  --device cuda \
  --eval-samples 64 \
  --out figures/response_field_comparison_gpu_draft.png
```

Record the CUDA `single_trajectory_speedup_x`. If the first CUDA timing looks
unusually slow, rerun once after warmup and keep the second result.

## 5. Regenerate the final rebuttal figure with both speedups

Replace `<CPU_SPEEDUP>` and `<GPU_SPEEDUP>` with the measured values from steps 3
and 4:

```bash
python scripts/plot_response_figure.py \
  --checkpoint checkpoints/lc64_canonical_cpu.pt \
  --fixed-canonical-params \
  --device cuda \
  --eval-samples 64 \
  --caption-cpu-speedup-x <CPU_SPEEDUP> \
  --caption-gpu-speedup-x <GPU_SPEEDUP> \
  --dataset-size 512 \
  --out figures/response_field_comparison.png
```

Acceptance:

- `figures/response_field_comparison.png` exists.
- Figure is 2 rows x 3 columns with one shared colorbar.
- Caption numbers match stdout.
- `mean_relative_l2 <= 0.1`, preferably near `0.0075`.
- Backend contains `PhysicsNeMo`.

## 6. Optional: benchmark batched throughput separately

The rebuttal text currently discusses batched speedup. The response figure uses
single-trajectory timing. To report batched speedup, run:

```bash
python benchmark.py \
  --checkpoint checkpoints/lc64_canonical_cpu.pt \
  --resolution 64 \
  --samples 64 \
  --steps 500 \
  --dt 2e-4 \
  --device cpu \
  --require-physicsnemo \
  --init-mode canonical \
  --canonical-mode 3,2 \
  --canonical-amplitude 0.8 \
  --fixed-canonical-params

python benchmark.py \
  --checkpoint checkpoints/lc64_canonical_cpu.pt \
  --resolution 64 \
  --samples 64 \
  --steps 500 \
  --dt 2e-4 \
  --device cuda \
  --require-physicsnemo \
  --init-mode canonical \
  --canonical-mode 3,2 \
  --canonical-amplitude 0.8 \
  --fixed-canonical-params
```

Use these batched numbers in the prose only if the text explicitly says
"batched inference speedup." Use the response-figure script numbers if the text
refers to the plotted single-trajectory figure.

## 7. Optional: train a GPU-native checkpoint

Only do this if we want a checkpoint trained on the GPU machine rather than using
the existing checkpoint:

```bash
python scripts/generate_data.py \
  --samples 512 \
  --resolution 64 \
  --steps 500 \
  --dt 2e-4 \
  --init-mode canonical \
  --canonical-mode 3,2 \
  --canonical-amplitude 0.8 \
  --fixed-canonical-params \
  --fixed-target-time \
  --out data/lc64_canonical_gpu.npz

python train_fno.py \
  --data data/lc64_canonical_gpu.npz \
  --epochs 20 \
  --batch-size 8 \
  --checkpoint checkpoints/lc64_canonical_gpu.pt \
  --latent-channels 24 \
  --modes 12 \
  --physics-weight 0.0 \
  --device cuda \
  --require-physicsnemo
```

Then repeat steps 3-5 with `checkpoints/lc64_canonical_gpu.pt`.

## 8. Optional: PDE-loss comparison

The repository includes a local-in-time spectral PDE residual for PINO-style
training. To compare MSE-only against PDE-regularized training:

```bash
python train_fno.py \
  --data data/lc64_canonical_gpu.npz \
  --epochs 20 \
  --batch-size 8 \
  --checkpoint checkpoints/lc64_canonical_pde_gpu.pt \
  --latent-channels 24 \
  --modes 12 \
  --physics-weight 1e-4 \
  --physics-dt 1e-3 \
  --device cuda \
  --require-physicsnemo
```

Then run the same response-figure and benchmark commands with
`checkpoints/lc64_canonical_pde_gpu.pt`.

## 9. Report back

Please report these exact values:

- checkpoint used
- device and GPU model
- `mean_relative_l2`
- `median_relative_l2`
- CPU `single_trajectory_speedup_x`
- GPU `single_trajectory_speedup_x`
- optional batched CPU/GPU speedups from `benchmark.py`
- whether the figure caption was regenerated with both speedups
