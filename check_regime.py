"""Verify the data-regime fix on EVOLVED TARGETS (not initial conditions).

Run from repo root after the Task 1-fix change:
    python check_regime.py

Checks three criteria and prints PASS/FAIL:
  1. median target mean|Q| within ~30% of sqrt(-A/C)
  2. fraction of targets with mean|Q| < 0.10  below 0.15
  3. median ||qT - q0|| / ||q0||  in [0.2, 0.7]  (non-trivial, non-collapsed)
"""
import numpy as np
from lc_pino.data import generate_dataset

data = generate_dataset(samples=128, resolution=64, steps=200, dt=2e-4, seed=1)
inp, tgt = data["inputs"], data["targets"]
q0 = inp[:, :2]
par = data["params"]  # A, C, L, gamma, target_time

mag_t = np.sqrt(tgt[:, 0]**2 + tgt[:, 1]**2).reshape(len(tgt), -1).mean(1)
mag_0 = np.sqrt(q0[:, 0]**2 + q0[:, 1]**2).reshape(len(q0), -1).mean(1)
eq = np.sqrt(-par[:, 0] / par[:, 1])
rel_move = (np.linalg.norm((tgt - q0).reshape(len(tgt), -1), axis=1)
            / np.linalg.norm(q0.reshape(len(q0), -1), axis=1).clip(1e-8))

med_t, med_eq = np.median(mag_t), np.median(eq)
frac_low = (mag_t < 0.10).mean()
med_move = np.median(rel_move)

print("=== TARGET magnitude (acceptance) ===")
print(f"  median mean|Q| target = {med_t:.3f}   equilibrium = {med_eq:.3f}")
print(f"  fraction target<0.10  = {frac_low:.2%}")
print(f"  (IC median mean|Q|    = {np.median(mag_0):.3f}  -- sanity only)")
print("=== Task non-triviality ===")
print(f"  median ||qT-q0||/||q0|| = {med_move:.3f}")

c1 = abs(med_t - med_eq) / med_eq < 0.30
c2 = frac_low < 0.15
c3 = 0.2 <= med_move <= 0.7
print("=== RESULT ===")
print(f"  [{'PASS' if c1 else 'FAIL'}] target |Q| near equilibrium")
print(f"  [{'PASS' if c2 else 'FAIL'}] few collapsed targets")
print(f"  [{'PASS' if c3 else 'FAIL'}] motion in healthy range")
print("  ALL PASS" if (c1 and c2 and c3) else "  NOT YET — adjust corr_modes")
