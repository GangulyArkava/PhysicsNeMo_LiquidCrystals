# Task: Generate the reviewer-response field-comparison figure

## Goal
Produce a single publication-quality figure for a grant rebuttal that shows the
PhysicsNeMo FNO reproducing 2D Landau–de Gennes director relaxation against the
spectral reference solver, with a caption whose numbers match the response text.

The existing `scripts/plot_comparison.py` already does ~90% of this. Your job is a
**targeted modification + a new wrapper**, not a rewrite. Read `plot_comparison.py`
first and reuse its `director()` / `draw_director()` / `make_input()` machinery.

## Deliverables
1. `scripts/plot_response_figure.py` — a new script (copy-and-adapt from
   `plot_comparison.py`) that produces the figure described below.
2. `figures/response_field_comparison.png` — the rendered figure (300 dpi).
3. Printed-to-stdout summary block with the exact numbers for the caption.

Do **not** modify `plot_comparison.py`, `solver.py`, `data.py`, or `models.py`.
All new logic lives in the new script.

## Inputs / assumptions
- Trained checkpoint exists (a dict with keys: `model`, `latent_channels`,
  `modes`, `backend`). Path passed via `--checkpoint`. The user runs this locally
  where **PhysicsNeMo is installed**, so `build_model` must load the real FNO,
  not the TinyFNO fallback. Add `--require-physicsnemo` (default True) and pass it
  through to `build_model(require_physicsnemo=...)` so the run FAILS LOUDLY if the
  real backend is unavailable — a figure built on the untrained fallback would be
  meaningless and must never be produced silently.
- Canonical benchmark settings, matching how the model was trained:
  `--init-mode canonical`, `--canonical-mode 3,2`, `--canonical-amplitude 0.8`,
  `--resolution 64`, `--steps 500`, `--dt 2e-4` (target time t = 0.1),
  `--fixed-canonical-params` (A=-0.1, C=1.0, L=0.02, gamma=1.0).
  Expose all as CLI args with these defaults.

## Figure spec (the main panel — same layout as plot_comparison, refined)
- 2 rows × 3 columns. Top row = spectral reference; bottom row = FNO.
- Columns = target times [0.0, t/2, t] labelled "initial", "intermediate", "final".
- Reuse `draw_director()` exactly (director quiver over |Q| heatmap, shared vmin/vmax,
  viridis, shared colorbar labelled scalar order |Q|).
- CRITICAL: preserve the existing convention at line ~133 — the bottom-row "initial"
  panel shows the shared `q0` (the IC the operator consumes), NOT `pred[0]`. The FNO
  panels show `pred[1]` (intermediate) and `pred[2]` (final). Do not "fix" this.
- Title/caption must be clean for a document (no dev-style timing dump). Put the
  headline metrics in a one-line caption: mean relative L2, CPU speedup, GPU speedup,
  dataset size. Use placeholders pulled from the multi-sample eval below, not the
  single-IC number.

## The one substantive addition: multi-sample error, not single-IC
`plot_comparison.py` reports `final_rel` for ONE initial condition. For the rebuttal
the headline error must be a **mean over a held-out set**, or a reviewer can dismiss
it as a lucky sample. Add this:

- New arg `--eval-samples` (default 64). Before plotting, loop over `--eval-samples`
  held-out ICs drawn with `sample_params` / the canonical generator using a FIXED
  seed (`--seed`, default 7) advanced per sample so the set is reproducible.
- For each: spectral solve to t (ground truth) and a single FNO forward pass via
  `make_input(q0, params, t)`. Compute per-sample relative L2 at the final time:
  `||pred_T - spectral_T|| / max(||spectral_T||, 1e-8)`.
- Report **mean** and **median** relative L2 over the set. The figure caption uses
  the mean. Print both to stdout.
- Then pick ONE representative sample for the visual panel: the sample whose error is
  closest to the median (NOT the best — closest-to-median is the honest choice).
  Print which sample index was chosen and its error.

## Timing / speedup
- Measure spectral wall time and FNO wall time as plot_comparison already does
  (with `torch.cuda.synchronize()` guards on CUDA). Report single-trajectory speedup.
- Speedups depend on device. Run the script TWICE locally: once `--device cpu`,
  once `--device cuda`, and record both numbers for the caption. Do NOT hardcode
  35x / 53x — emit whatever this checkpoint actually produces and let the user
  reconcile with the text. (The response currently cites ~35x CPU / ~53x GPU batched;
  if your measured numbers differ, print a clear warning so the user updates the text.)

## Stdout summary block (must print exactly these, labelled)
- `eval_samples`, `mean_relative_l2`, `median_relative_l2`
- `chosen_sample_index`, `chosen_sample_relative_l2`
- `device`, `single_trajectory_speedup_x`
- `spectral_time_s`, `fno_time_s`
- `backend` (must contain "PhysicsNeMo", else abort)

## Acceptance checks
- Script aborts with a clear error if `backend` does not contain "PhysicsNeMo"
  (guards against the TinyFNO fallback silently producing a fake figure).
- `figures/response_field_comparison.png` exists, 300 dpi, 2x3, shared colorbar.
- Mean relative L2 over >=64 samples printed and is ~7e-3 order of magnitude for a
  correctly trained canonical model (sanity bound: abort/warn if mean > 0.1).
- Caption numbers in the PNG match the stdout summary exactly.

## Run commands (document these at top of the script's docstring)
```
# from repo root, PhysicsNeMo env active
python scripts/plot_response_figure.py --checkpoint path/to/ckpt.pt \
    --fixed-canonical-params --device cpu  --eval-samples 64
python scripts/plot_response_figure.py --checkpoint path/to/ckpt.pt \
    --fixed-canonical-params --device cuda --eval-samples 64
```

## Out of scope (do not build)
- No error-vs-time curve (separate task).
- No long-time / extrapolation panel.
- No autoregressive rollout — the operator is queried at target time directly.
