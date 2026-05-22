from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lc_pino.data import make_input, sample_params
from lc_pino.models import build_model
from lc_pino.solver import defect_pair_q, solve_ldg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot spectral vs PhysicsNeMo director fields.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures/director_comparison.png"))
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stride", type=int, default=4)
    return parser.parse_args()


def director(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = 0.5 * np.arctan2(q[1], q[0])
    order = np.sqrt(q[0] ** 2 + q[1] ** 2)
    return np.cos(theta), np.sin(theta), order


def draw_director(
    ax,
    q: np.ndarray,
    title: str,
    stride: int,
    vmin: float,
    vmax: float,
):
    nx = q.shape[-1]
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, nx, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ux, uy, order = director(q)
    im = ax.imshow(
        order.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    s = max(1, stride)
    ax.quiver(
        xx[::s, ::s],
        yy[::s, ::s],
        ux[::s, ::s],
        uy[::s, ::s],
        color="white",
        pivot="middle",
        headwidth=0,
        headlength=0,
        headaxislength=0,
        scale=28,
        width=0.003,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    params = sample_params(rng)
    q0 = defect_pair_q(args.resolution, rng, scalar_order=float(np.sqrt(-params.A / params.C)))
    mid_steps = max(1, args.steps // 2)
    times = [0.0, mid_steps * args.dt, args.steps * args.dt]

    t0 = time.perf_counter()
    mid = solve_ldg(q0, params=params, dt=args.dt, steps=mid_steps)
    final = solve_ldg(q0, params=params, dt=args.dt, steps=args.steps)
    solver_time = time.perf_counter() - t0
    spectral = [q0, mid, final]

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(
        latent_channels=int(ckpt.get("latent_channels", 32)),
        modes=int(ckpt.get("modes", 12)),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    backend = ckpt.get("backend", getattr(model, "backend_name", "unknown backend"))

    batch = np.stack([make_input(q0, params, t) for t in times], axis=0)
    xb = torch.from_numpy(batch).float().to(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        pred = model(xb).detach().cpu().numpy()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    model_time = time.perf_counter() - t1

    # The neural operator consumes the initial condition; show it as the shared
    # initial state and compare learned predictions at later target times.
    nemo = [q0, pred[1], pred[2]]
    final_rel = np.linalg.norm(nemo[-1] - spectral[-1]) / max(np.linalg.norm(spectral[-1]), 1.0e-8)
    speedup = solver_time / max(model_time, 1.0e-12)
    all_orders = [np.sqrt(q[0] ** 2 + q[1] ** 2) for q in spectral + nemo]
    vmin = float(min(np.min(order) for order in all_orders))
    vmax = float(max(np.max(order) for order in all_orders))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    labels = ["initial", "middle", "final"]
    image = None
    for j, label in enumerate(labels):
        image = draw_director(axes[0, j], spectral[j], f"Spectral {label}", args.stride, vmin, vmax)
        draw_director(axes[1, j], nemo[j], f"{backend} {label}", args.stride, vmin, vmax)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.88, pad=0.02)
        colorbar.set_label(r"scalar order $|Q| = \sqrt{q_1^2 + q_2^2}$")
    fig.suptitle(
        f"Landau-de Gennes director field | spectral {solver_time:.4f}s, "
        f"{backend} {model_time:.4f}s, speedup {speedup:.2f}x, final rel L2 {final_rel:.3f}",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"spectral_time_s: {solver_time:.6f}")
    print(f"physicsnemo_time_s: {model_time:.6f}")
    print(f"speedup_x: {speedup:.2f}")
    print(f"final_relative_l2: {final_rel:.6f}")


if __name__ == "__main__":
    main()
