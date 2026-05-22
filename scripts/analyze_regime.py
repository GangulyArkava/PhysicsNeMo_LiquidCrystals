from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lc_pino.data import generate_dataset
from lc_pino.solver import LdGParams, canonical_director_q, free_energy, solve_ldg


def rel_move(q0: np.ndarray, qT: np.ndarray) -> float:
    return float(
        np.linalg.norm((qT - q0).reshape(-1))
        / max(np.linalg.norm(q0.reshape(-1)), 1.0e-8)
    )


def mean_order(q: np.ndarray) -> float:
    return float(np.mean(np.sqrt(q[0] ** 2 + q[1] ** 2)))


def summarize_dataset_corr_modes(args: argparse.Namespace) -> None:
    print("=== Random Smooth Dataset Corr-Mode Sweep ===")
    for corr_modes in args.corr_modes:
        data = generate_dataset(
            samples=args.samples,
            resolution=args.resolution,
            steps=args.steps,
            dt=args.dt,
            seed=args.seed,
            corr_modes=corr_modes,
        )
        q0 = data["inputs"][:, :2]
        qT = data["targets"]
        params = data["params"]
        mag_t = np.sqrt(qT[:, 0] ** 2 + qT[:, 1] ** 2).reshape(len(qT), -1).mean(axis=1)
        mag_0 = np.sqrt(q0[:, 0] ** 2 + q0[:, 1] ** 2).reshape(len(q0), -1).mean(axis=1)
        eq = np.sqrt(-params[:, 0] / params[:, 1])
        motion = np.linalg.norm((qT - q0).reshape(len(qT), -1), axis=1) / np.linalg.norm(
            q0.reshape(len(q0), -1),
            axis=1,
        ).clip(1.0e-8)
        print(
            f"corr_modes={corr_modes:4.2f} | "
            f"target/eq={np.median(mag_t) / np.median(eq):5.3f} | "
            f"target|Q|={np.median(mag_t):5.3f} | "
            f"eq={np.median(eq):5.3f} | "
            f"low<0.10={(mag_t < 0.10).mean():5.1%} | "
            f"move={np.median(motion):5.3f} | "
            f"ic|Q|={np.median(mag_0):5.3f}"
        )


def summarize_canonical_modes(args: argparse.Namespace) -> None:
    params = LdGParams(A=args.A, C=args.C, L=args.L, gamma=args.gamma)
    qeq = float(np.sqrt(-params.A / params.C))
    print("\n=== Canonical Defect-Free Director Sweep ===")
    print(
        f"params: A={params.A:g}, C={params.C:g}, L={params.L:g}, "
        f"gamma={params.gamma:g}, q_eq={qeq:.4f}, "
        f"resolution={args.resolution}, steps={args.steps}, dt={args.dt:g}"
    )
    for mode_x, mode_y in args.modes:
        for amp in args.amplitudes:
            q0 = canonical_director_q(
                args.resolution,
                equilibrium_magnitude=qeq,
                theta0=args.theta0,
                amplitude=amp,
                mode_x=mode_x,
                mode_y=mode_y,
            )
            qT = solve_ldg(q0, params=params, dt=args.dt, steps=args.steps)
            e0 = free_energy(q0, params)
            eT = free_energy(qT, params)
            print(
                f"mode=({mode_x},{mode_y}) amp={amp:4.2f} | "
                f"target/eq={mean_order(qT) / qeq:5.3f} | "
                f"target|Q|={mean_order(qT):5.3f} | "
                f"move={rel_move(q0, qT):5.3f} | "
                f"energy_drop={(e0 - eT) / max(abs(e0), 1.0e-8):7.2%}"
            )


def summarize_grid_scaling(args: argparse.Namespace) -> None:
    params = LdGParams(A=args.A, C=args.C, L=args.L, gamma=args.gamma)
    qeq = float(np.sqrt(-params.A / params.C))
    print("\n=== Canonical Grid Consistency ===")
    for resolution in args.grid_resolutions:
        q0 = canonical_director_q(
            resolution,
            equilibrium_magnitude=qeq,
            theta0=args.theta0,
            amplitude=args.grid_amplitude,
            mode_x=args.grid_mode[0],
            mode_y=args.grid_mode[1],
        )
        qT = solve_ldg(q0, params=params, dt=args.dt, steps=args.steps)
        print(
            f"N={resolution:4d} | "
            f"target/eq={mean_order(qT) / qeq:5.3f} | "
            f"target|Q|={mean_order(qT):5.3f} | "
            f"move={rel_move(q0, qT):5.3f} | "
            f"energy={free_energy(qT, params): .6e}"
        )


def parse_mode(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("mode must be formatted as M,N")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze liquid-crystal data regimes.")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--corr-modes", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--A", type=float, default=-0.1)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--theta0", type=float, default=0.25)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--modes", type=parse_mode, nargs="+", default=[(1, 1), (2, 1), (2, 2), (3, 2)])
    parser.add_argument("--grid-resolutions", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--grid-mode", type=parse_mode, default=(1, 1))
    parser.add_argument("--grid-amplitude", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_dataset_corr_modes(args)
    summarize_canonical_modes(args)
    summarize_grid_scaling(args)


if __name__ == "__main__":
    main()
