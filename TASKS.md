# TASKS.md — PhysicsNeMo Liquid-Crystal Demo

Work list for fixing correctness and benchmark issues in this repo. Tasks are
ordered by priority. **Do all P0 tasks first, then stop for human review before
starting Task 5** (it contains a modeling decision that needs a human call).

After each P0 change, run the smoke commands in the
[Regression check](#regression-check) section as a sanity test.

---

## P0 — Fix the data regime

Nothing else measures the right thing until this is fixed. Right now ~64% of
generated targets collapse to near-isotropic (`mean|Q| < 0.10`) because the
random initial conditions are zero-mean and the integration horizon is far too
short for nematic domains to coarsen. This makes accuracy numbers misleading and
contradicts the "elastic smoothing preserves nematic order" claim in the README.

- [ ] **Task 1 — Stop random ICs collapsing to isotropic.**
  - Files: `lc_pino/solver.py`, `lc_pino/data.py`
  - The smooth-random fields relax toward `|Q| ≈ 0` (the unstable isotropic
    state) instead of the nematic equilibrium `|Q| = sqrt(-A/C)`.
  - Implement **(b) as the default**: initialize random fields as a perturbation
    *around* the equilibrium magnitude `sqrt(-A/C)` rather than around zero.
  - Also expose **(a) via CLI**: allow a much longer total integration time
    (`dt * steps` ≈ 0.5–2.0 vs the current ~0.04) so random fields genuinely
    coarsen. Wire this through `scripts/generate_data.py`.
  - **Acceptance:** a freshly generated dataset has median target `mean|Q|`
    within ~20% of `sqrt(-A/C)`, AND the fraction of targets with
    `mean|Q| < 0.10` is below ~0.15 (currently 0.64). Add a small script or
    print statement that reports both numbers so this is checkable.

- [ ] **Task 2 — Regenerate data, re-benchmark, update README.**
  - Files: `README.md`, plus regenerated artifacts in `data/` and `figures/`
  - After Task 1, regenerate the dataset, re-run `benchmark.py` and
    `scripts/plot_comparison.py`, and update all quoted numbers and figures in
    the README to reflect the corrected regime.
  - **Acceptance:** README numbers match a fresh run; no stale pre-fix figures
    remain in `figures/`.

---

## P0 — Confirm the backend is real

There is a real risk that CPU-only runs have been silently using the local
`TinyFNO` fallback while labels still say "PhysicsNeMo FNO".

- [ ] **Task 3 — Make the PhysicsNeMo fallback loud and verifiable.**
  - Files: `lc_pino/models.py`, `train_fno.py`, `benchmark.py`,
    `scripts/plot_comparison.py`
  - In `build_model`, replace the bare `except Exception` around the PhysicsNeMo
    import with handling that logs a **visible warning naming the actual
    exception** before falling back.
  - Add a `--require-physicsnemo` flag to `train_fno.py` and `benchmark.py` that
    **hard-fails** instead of falling back.
  - Ensure the checkpoint always stores the true `backend`, and that
    `scripts/plot_comparison.py` titles from that stored value — remove the
    hardcoded `"PhysicsNeMo FNO"` default (currently ~line 103).
  - **Acceptance:** running without PhysicsNeMo prints a clear warning naming the
    import error; `--require-physicsnemo` raises instead of falling back; plot
    titles reflect the real backend from the checkpoint.

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
    matches the NumPy solver's Laplacian to floating-point tolerance (add a test;
    see Task 10's test file).

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

- Do **P0 (Tasks 1–3) first, then pause for human review** before Task 5.
- Task 5 is a modeling decision — present options, do not pick unilaterally.
- Keep the two-component traceless Q representation consistent across solver,
  data, models, and plotting.
- After Tasks 1 and 6, old checkpoints are invalid (input distribution changed);
  retrain rather than loading stale checkpoints.
