# TASKS.md — PhysicsNeMo Liquid-Crystal Demo

Work list for fixing correctness and benchmark issues in this repo. Tasks are
ordered by priority. **Start with Task 0 — it is blocking and the previous P0
attempt did not actually fix it.** Then continue in order, and **stop for human
review before starting Task 5** (it contains a modeling decision that needs a
human call).

Verify the data-regime fix at the REAL resolution/steps (64×64, 200 steps), not
just the 32×32 smoke config — the two diverged last time and the smoke config
masked the bug. After other P0 changes, also run the smoke commands in the
[Regression check](#regression-check) section as a sanity test.

---

## P0 — Fix the data regime  [STILL BROKEN — DO THIS FIRST]

The previous commit (`8ea6770`) fixed the **initial conditions** but NOT the
**targets**. Verified against the actual `generate_dataset` at 64×64 / 200 steps:

```
IC     median mean|Q| = 0.416   (correct, at equilibrium)
TARGET median mean|Q| = 0.055   (collapsed)
fraction of targets < 0.10 = 71%   (was 64% before the change — slightly worse)
```

The IC-magnitude acceptance check passed, but the field still relaxes to
near-isotropic within ~20 steps, so accuracy numbers remain meaningless and the
"elastic smoothing preserves nematic order" claim is still unsupported.

**Root cause (confirmed numerically), both in `smooth_random_q` in
`lc_pino/solver.py`:**

1. **The lowpass filter barely smooths.** `filt = exp(-lowpass * k2/k2_max)` with
   `lowpass=0.12` retains ~89% amplitude even at the highest wavenumber — it is
   nearly a pass-through, so the "smooth" field is full of high-k content. (This
   bug predates the change; it was harmless for zero-mean fields but is fatal once
   the field is built from a director angle.)
2. **Pointwise magnitude does not survive gradient flow; director *alignment*
   does.** Building `Q = q_eq·[cos 2θ, sin 2θ]` from a θ field with high-k content
   makes `q1, q2` oscillate rapidly in space (~99% of spectral power at |k|>20).
   The elastic term `L∇²Q` smooths those oscillations toward their spatial mean
   (≈0 for cos/sin), destroying |Q| while the pointwise t=0 magnitude check still
   looks fine.

- [ ] **Task 0 — Give the smoothing filter a real correlation length. [BLOCKING]**
  - File: `lc_pino/solver.py` (`smooth_random_q`)
  - Replace the near-pass-through filter with one that has a genuine length scale,
    e.g. a Gaussian cut at a few modes: `filt = exp(-k2 / kc)` where
    `kc = (2π·corr_modes)²` and `corr_modes ≈ 2` (correlation length ~ box/2).
  - Apply the same proper filter to BOTH the director-angle field and the noise.
  - Keep the `equilibrium_magnitude` argument and the backward-compat (zero-mean)
    path — this is about the filter, not the API.
  - Expose `corr_modes` (or an equivalent length scale) as an argument, threaded
    through `lc_pino/data.py` so it is tunable from data generation.
  - **Reference (verified-good):** with `corr_modes ≈ 2`, default params
    (A=-0.1, C=1, L=0.02, γ=1), 200 steps, dt=2e-4: target |Q| 0.313 → 0.202,
    motion ≈ 0.445. Larger `corr_modes` (rougher field) collapses more; tune so
    the Task 0-verify criteria below all pass.

- [ ] **Task 0-verify — acceptance is on TARGETS, with a non-triviality guard.**
  - The diagnostic already in `scripts/generate_data.py` measures targets — keep
    it, and extend it to also report median `||qT - q0|| / ||q0||` so we do not
    overcorrect into a trivial "predict the input back" task.
  - Run `check_regime.py` (in repo root) as the gate before regenerating the full
    dataset. **All three must hold at 64×64 / 200 steps:**
    - median target `mean|Q|` within ~30% of `sqrt(-A/C)`,
    - fraction of targets with `mean|Q| < 0.10` below ~0.15,
    - median `||qT - q0|| / ||q0||` in [0.2, 0.7].

- [x] **Task 1 — IC initialization (DONE, but did not fix the regime).**
  - Files: `lc_pino/solver.py`, `lc_pino/data.py`
  - `smooth_random_q` now accepts `equilibrium_magnitude` and builds
    `Q = q_eq·[cos 2θ, sin 2θ] + noise`; `generate_dataset` passes
    `sqrt(-A/C)`. `--total-time` CLI override added to `generate_data.py`.
  - This correctly fixed the ICs but the targets still collapse — see Task 0. The
    original acceptance criterion was wrong because it was satisfiable by the IC
    construction alone without the dynamics being meaningful.

- [ ] **Task 2 — Regenerate data, re-benchmark, update README.**
  - Files: `README.md`, plus regenerated artifacts in `data/` and `figures/`
  - **Only after Task 0 passes `check_regime.py`**, regenerate the dataset,
    re-run `benchmark.py` and `scripts/plot_comparison.py`, and update all quoted
    numbers and figures in the README to reflect the corrected regime.
  - **Acceptance:** README numbers match a fresh run; no stale pre-fix figures
    remain in `figures/`.

---

## P0 — Confirm the backend is real

There is a real risk that CPU-only runs have been silently using the local
`TinyFNO` fallback while labels still say "PhysicsNeMo FNO".

- [x] **Task 3 — Make the PhysicsNeMo fallback loud and verifiable. (DONE)**
  - Files: `lc_pino/models.py`, `train_fno.py`, `benchmark.py`,
    `scripts/plot_comparison.py`
  - Verified in commit `8ea6770`: `except Exception as e` emits a `warnings.warn`
    naming the actual exception; `require_physicsnemo=True` raises with the cause
    chained; `--require-physicsnemo` added to both scripts; `benchmark.py` prints
    `model_backend` from the checkpoint; the hardcoded `"PhysicsNeMo FNO"` plot
    default was removed in favor of the checkpoint's stored backend.

---

## P1 — Physics residual correctness

- [ ] **Task 4 — Use a spectral Laplacian in the physics residual.**
  - File: `lc_pino/models.py` (`endpoint_physics_residual`,
    `periodic_laplacian_torch`)
  - The residual currently uses a 5-point finite-difference stencil while the
    solver (`lc_pino/solver.py`) uses an exact spectral Laplacian. The physics
    loss therefore penalizes a *different* PDE than the data was generated from.
  - Replace the FD Laplacian with an FFT-based one matching `solver.py`.
  - **Acceptance:** on a known single-mode field, the Torch residual Laplacian
    matches the NumPy solver's Laplacian to floating-point tolerance (add a test
    under `tests/`).

- [ ] **Task 5 — Fix the long-interval time discretization. [STOP — needs human review first]**
  - File: `lc_pino/models.py` (`endpoint_physics_residual`)
  - The current `(pred - q0) / target_time - rhs(pred)` treats a secant over the
    full (large) target time as `dQ/dt`. This is a poor approximation when the
    field changes ~98% over the interval.
  - **Do not implement until reviewed.** Two candidate designs:
    1. Short-Δt consistency loss: predict a small step and enforce the RHS.
    2. Autoregressive one-step rollout (already listed under README "Next
       Extensions").
  - Bring a short written recommendation (tradeoffs of each) to the human before
    coding.

---

## P1 — Input normalization

- [ ] **Task 6 — Normalize parameter and time input channels.**
  - Files: `lc_pino/data.py` (`make_input`), `train_fno.py`, `benchmark.py`,
    `scripts/plot_comparison.py`
  - Channels A, C, L, gamma, target_time span very different scales (e.g. A ~
    -0.3..-0.05, C ~ 1, L ~ 0.02, target_time ~ 0.004..0.1) and feed the FNO
    unnormalized.
  - Normalize these channels to O(1). **Store the normalization constants in the
    checkpoint** and apply them identically at train and inference time.
  - **Acceptance:** normalization constants are saved in the checkpoint and
    applied identically in `train_fno.py`, `benchmark.py`, and
    `scripts/plot_comparison.py`; inference with a saved checkpoint reproduces
    training-time behavior.

---

## P2 — Fairer benchmark and diagnostics

- [ ] **Task 7 — Batch the spectral solver in `benchmark.py`.**
  - File: `benchmark.py`
  - It currently solves trajectories one at a time in a list comprehension and
    compares to a single batched FNO pass. The NumPy FFT solver vectorizes over a
    leading batch axis nearly for free.
  - Make `solve_ldg` (or a batched variant) operate on a leading batch dimension
    and time the batched solve, so the comparison is apples-to-apples.
  - **Acceptance:** benchmark reports batched solver time; results are unchanged
    numerically vs the per-sample loop (same final fields within tolerance).

- [ ] **Task 8 — Add a per-trajectory rel-L2 distribution plot, split by init type.**
  - Files: `benchmark.py` or a new `scripts/plot_error_distribution.py`
  - Plot the distribution (histogram or violin) of per-trajectory relative L2,
    split by random-field vs defect-field initial conditions. This explains the
    mean-vs-median gap.
  - **Acceptance:** a figure is produced showing two clearly labeled
    distributions; the script prints mean and median per group.

- [ ] **Task 9 — Add a wall-clock-vs-grid-size crossover plot.**
  - File: new `scripts/plot_scaling.py`
  - Plot spectral solver time vs FNO inference time as a function of grid
    resolution (log-log), to identify the crossover where the surrogate pays off.
  - **Acceptance:** a log-log figure across at least 4 resolutions (e.g. 32, 64,
    128, 256) with both curves and the crossover visible.

---

## P2 — Test the spectral conv (fallback only)

- [ ] **Task 10 — Unit-test `SpectralConv2d` mode indexing.**
  - Files: `lc_pino/models.py`, new `tests/test_spectral_conv.py`
  - The negative-frequency block (`out_ft[:, :, -m1:, :m2]` written from
    `weights2[:, :, :m1, :m2]`, ~lines 97–98) is unconventional and may mix
    modes. Only affects the TinyFNO fallback, but CPU runs may have used it.
  - Add a test feeding a known single-mode field and checking the transform
    behaves as expected. Fix the indexing if the test reveals a bug.
  - **Acceptance:** test passes (or indexing is corrected and then passes).

---

## Regression check

Run after each P0 change (from README "Quick Smoke Run"):

```bash
python scripts/generate_data.py --samples 16 --resolution 32 --steps 20 --out data/lc32_smoke.npz
python train_fno.py --data data/lc32_smoke.npz --epochs 2 --batch-size 4 --checkpoint checkpoints/smoke.pt
python benchmark.py --checkpoint checkpoints/smoke.pt --resolution 32 --samples 8 --steps 20
python scripts/plot_comparison.py --checkpoint checkpoints/smoke.pt --resolution 32 --steps 20 --out figures/smoke_directors.png
```

(Adjust `--steps`/`--dt` to match the new default time horizon from Task 1.)

## Notes for the agent

- **Task 0 is blocking** — fix it and confirm `check_regime.py` prints ALL PASS
  at 64×64 / 200 steps before doing anything else. Task 3 is already done.
- **Verify the data regime at real settings, not the smoke config.** The smoke
  run (32×32 / 20 steps) passed last time while the real run (64×64 / 200 steps)
  failed, because the collapse needs enough steps to set in. Use the smoke
  commands only as a quick "does it run" check, never as the regime acceptance
  gate.
- The acceptance metric must be measured on **targets** (post-`solve_ldg`), never
  on initial conditions, and must include the motion guard so a near-identity
  task can't pass.
- Pause for human review before Task 5 — it is a modeling decision; present
  options, do not pick unilaterally.
- Keep the two-component traceless Q representation consistent across solver,
  data, models, and plotting.
- After Tasks 0 and 6, old checkpoints are invalid (input distribution changed);
  retrain rather than loading stale checkpoints.
