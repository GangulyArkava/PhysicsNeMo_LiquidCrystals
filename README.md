# PhysicsNeMo Liquid-Crystal Demo

This repository is a small working model problem for demonstrating AI-accelerated
liquid-crystal simulation with NVIDIA PhysicsNeMo.

The example uses the simplest useful Landau-de Gennes gradient-flow model in 2D.
The symmetric traceless Q-tensor is represented by two scalar fields,

```text
Q = [[ q1,  q2 ],
     [ q2, -q1 ]]
```

with the rescaled two-component free energy

```text
F[q1,q2] = integral [
    A/2 (q1^2 + q2^2)
  + C/4 (q1^2 + q2^2)^2
  + L/2 (|grad q1|^2 + |grad q2|^2)
] dx dy.
```

The gradient-flow dynamics are

```text
dq1/dt = Gamma [ L laplacian(q1) - A q1 - C (q1^2 + q2^2) q1 ]
dq2/dt = Gamma [ L laplacian(q2) - A q2 - C (q1^2 + q2^2) q2 ]
```

The default parameter sampler uses `A < 0`, `C > 0`, and `L > 0`, so the
uniform nematic magnitude is stable at

```text
sqrt(q1^2 + q2^2) = sqrt(-A/C).
```

With `A > 0`, this same equation is still mathematically valid but relaxes
toward the isotropic state `Q = 0`; that is useful as a smoothing test, but it
is not the right regime for preserving a nematic phase.

This is a deliberately conservative first demo: it has recognizable
Landau-de Gennes physics, periodic boundaries, defect-friendly Q-tensor fields,
and a cheap spectral reference solver for generating training data.

## Why This Demonstrates Speedup

The reference solver advances the PDE through many time steps for each initial
condition and parameter set. A trained neural operator maps

```text
initial Q field + A, C, L, Gamma, target time -> Q field at target time
```

in a single forward pass. The benchmark reports the amortized inference speedup:
classical solve time divided by neural-operator inference time. This is the
right claim for AI surrogates: training is an up-front cost, while inference is
fast for repeated simulations, parameter sweeps, and design loops.

## Install

Use a Python environment with PyTorch. PhysicsNeMo is optional for local smoke
tests, but recommended for the real demo.

```powershell
pip install -r requirements.txt
pip install "nvidia-physicsnemo[sym]"
```

If PhysicsNeMo is unavailable, `train_fno.py` falls back to a tiny local FNO-like
model so the data and benchmark workflow can still be tested.

## Quick Smoke Run

```powershell
python scripts/generate_data.py --samples 16 --resolution 32 --steps 20 --out data/lc32_smoke.npz
python train_fno.py --data data/lc32_smoke.npz --epochs 2 --batch-size 4 --checkpoint checkpoints/smoke.pt
python benchmark.py --checkpoint checkpoints/smoke.pt --resolution 32 --samples 8 --steps 20
python scripts/plot_comparison.py --checkpoint checkpoints/smoke.pt --resolution 32 --steps 20 --out figures/smoke_directors.png
```

## Suggested Real Demo

```powershell
python scripts/generate_data.py --samples 2048 --resolution 64 --steps 500 --out data/lc64_train.npz
python train_fno.py --data data/lc64_train.npz --epochs 100 --batch-size 16 --checkpoint checkpoints/lc64_fno.pt --physics-weight 0.001
python benchmark.py --checkpoint checkpoints/lc64_fno.pt --resolution 64 --samples 128 --steps 200
```

For a GPU benchmark, run the training and benchmark scripts with
`--device cuda`.

## Files

- `lc_pino/solver.py`: semi-implicit spectral reference solver and energy tools.
- `lc_pino/data.py`: dataset generation utilities.
- `lc_pino/models.py`: PhysicsNeMo FNO wrapper plus local fallback model.
- `train_fno.py`: neural-operator training with data and physics residual loss.
- `benchmark.py`: reference-solver timing vs model inference timing.
- `scripts/generate_data.py`: command-line data generation.

## Next Extensions

- Add anchored boundaries instead of periodic boundaries.
- Add chirality or electric-field coupling.
- Train an autoregressive one-step model for full trajectory rollout.
- Use defect-aware metrics for +/-1/2 defect positions and winding number.
