from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from lc_pino.data import generate_dataset


def parse_mode(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("mode must be formatted as M,N")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Landau-de Gennes Q-tensor data.")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument(
        "--total-time",
        type=float,
        default=None,
        help="Override dt*steps with a fixed total integration time (option a: long coarsening).",
    )
    parser.add_argument(
        "--corr-modes",
        type=float,
        default=2.0,
        help="Gaussian filter cutoff (in modes) for smooth_random_q. Lower = smoother ICs.",
    )
    parser.add_argument(
        "--init-mode",
        choices=["mixed", "smooth", "defect", "canonical"],
        default="mixed",
        help="Initial-condition family to generate.",
    )
    parser.add_argument("--canonical-mode", type=parse_mode, default=(3, 2))
    parser.add_argument("--canonical-amplitude", type=float, default=0.8)
    parser.add_argument(
        "--fixed-canonical-params",
        action="store_true",
        help="Use A=-0.1, C=1, L=0.02, gamma=1 for the canonical benchmark.",
    )
    parser.add_argument(
        "--fixed-target-time",
        action="store_true",
        help="Use exactly --steps for every sample instead of randomizing target time.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("data/lc64_train.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_time is not None:
        args.steps = max(1, int(round(args.total_time / args.dt)))
        print(f"--total-time {args.total_time}: using steps={args.steps} (dt={args.dt})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    remaining = args.samples
    seed = args.seed
    progress = tqdm(total=args.samples, desc="generating") if tqdm else None
    try:
        while remaining > 0:
            batch = min(128, remaining)
            data = generate_dataset(
                batch,
                args.resolution,
                args.steps,
                args.dt,
                seed,
                corr_modes=args.corr_modes,
                init_mode=args.init_mode,
                canonical_mode=args.canonical_mode,
                canonical_amplitude=args.canonical_amplitude,
                fixed_canonical_params=args.fixed_canonical_params,
                fixed_target_time=args.fixed_target_time,
            )
            chunks.append(data)
            remaining -= batch
            seed += 1
            if progress:
                progress.update(batch)
    finally:
        if progress:
            progress.close()
    merged = {
        "inputs": np.concatenate([c["inputs"] for c in chunks], axis=0),
        "targets": np.concatenate([c["targets"] for c in chunks], axis=0),
        "params": np.concatenate([c["params"] for c in chunks], axis=0),
        "dt": np.array(args.dt, dtype=np.float32),
    }
    np.savez_compressed(args.out, **merged)
    print(f"wrote {args.out} with {args.samples} samples at {args.resolution}x{args.resolution}")

    # Diagnostic: verify nematic order in generated targets
    targets = merged["targets"]  # (N, 2, H, W)
    params_arr = merged["params"]  # (N, 5): [A, C, L, gamma, target_time]
    q_mag = np.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2).mean(axis=(-2, -1))
    eq_mag = np.sqrt(-params_arr[:, 0] / params_arr[:, 1])
    print(f"order diagnostic (targets):")
    print(f"  median mean|Q|:                {np.median(q_mag):.4f}")
    print(f"  median equilibrium sqrt(-A/C): {np.median(eq_mag):.4f}")
    print(f"  fraction mean|Q| < 0.10:       {(q_mag < 0.10).mean():.4f}")
    q0 = merged["inputs"][:, :2]
    motion = np.linalg.norm((targets - q0).reshape(len(targets), -1), axis=1) / np.linalg.norm(
        q0.reshape(len(q0), -1),
        axis=1,
    ).clip(1.0e-8)
    print(f"  median ||qT-q0||/||q0||:       {np.median(motion):.4f}")


if __name__ == "__main__":
    main()
