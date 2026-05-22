from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from lc_pino.data import make_input, sample_params
from lc_pino.models import build_model
from lc_pino.solver import defect_pair_q, smooth_random_q, solve_ldg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark solver time against neural inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--require-physicsnemo",
        action="store_true",
        help="Raise an error instead of falling back to TinyFNO if PhysicsNeMo is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cases = []
    for i in range(args.samples):
        params = sample_params(rng)
        q0 = defect_pair_q(args.resolution, rng) if i % 3 == 0 else smooth_random_q(args.resolution, rng)
        target_time = args.steps * args.dt
        cases.append((q0.astype(np.float32), params, target_time))

    t0 = time.perf_counter()
    references = [
        solve_ldg(q0, params=params, dt=args.dt, steps=args.steps)
        for q0, params, _ in cases
    ]
    solver_time = time.perf_counter() - t0

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(
        latent_channels=int(ckpt.get("latent_channels", 32)),
        modes=int(ckpt.get("modes", 12)),
        require_physicsnemo=args.require_physicsnemo,
    ).to(args.device)
    backend = ckpt.get("backend", getattr(model, "backend_name", "unknown backend"))
    print(f"model_backend: {backend}")
    model.load_state_dict(ckpt["model"])
    model.eval()

    batch = np.stack([make_input(q0, params, t) for q0, params, t in cases], axis=0)
    xb = torch.from_numpy(batch).float().to(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        pred = model(xb)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - t1

    ref = torch.from_numpy(np.stack(references, axis=0)).float().to(args.device)
    rel = torch.linalg.vector_norm((pred - ref).flatten(1), dim=1) / torch.linalg.vector_norm(
        ref.flatten(1), dim=1
    ).clamp_min(1.0e-8)
    speedup = solver_time / max(inference_time, 1.0e-12)
    print(f"samples: {args.samples}")
    print(f"resolution: {args.resolution}x{args.resolution}")
    print(f"classical_solver_time_s: {solver_time:.6f}")
    print(f"model_inference_time_s: {inference_time:.6f}")
    print(f"amortized_speedup_x: {speedup:.2f}")
    print(f"mean_relative_l2: {float(torch.mean(rel)):.6f}")
    print(f"median_relative_l2: {float(torch.median(rel)):.6f}")


if __name__ == "__main__":
    main()
