"""Create the reviewer-response field-comparison figure.

Run from the repository root with a PhysicsNeMo environment active:

python scripts/plot_response_figure.py --checkpoint checkpoints/lc64_canonical_cpu.pt \
    --fixed-canonical-params --device cpu --eval-samples 64
python scripts/plot_response_figure.py --checkpoint checkpoints/lc64_canonical_cpu.pt \
    --fixed-canonical-params --device cuda --eval-samples 64

The current run measures one device. Use --caption-cpu-speedup-x and
--caption-gpu-speedup-x to place separately measured speedups in the caption.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lc_pino.data import make_input, sample_params
from lc_pino.models import build_model
from lc_pino.solver import LdGParams, canonical_director_q, solve_ldg


@dataclass(frozen=True)
class EvalSample:
    index: int
    params: LdGParams
    q0: np.ndarray
    mid: np.ndarray
    final: np.ndarray
    pred_mid: np.ndarray
    pred_final: np.ndarray
    relative_l2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication-quality spectral vs PhysicsNeMo response figure."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures/response_field_comparison.png"))
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--init-mode", choices=["canonical"], default="canonical")
    parser.add_argument("--canonical-mode", type=str, default="3,2")
    parser.add_argument("--canonical-amplitude", type=float, default=0.8)
    parser.add_argument("--fixed-canonical-params", action="store_true")
    parser.add_argument("--require-physicsnemo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--caption-cpu-speedup-x", type=float, default=None)
    parser.add_argument("--caption-gpu-speedup-x", type=float, default=None)
    return parser.parse_args()


def parse_canonical_mode(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--canonical-mode must be formatted as mode_x,mode_y")
    return int(parts[0]), int(parts[1])


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
    image = ax.imshow(
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
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def synchronize_if_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def canonical_sample(
    rng: np.random.Generator,
    resolution: int,
    params: LdGParams,
    canonical_mode: tuple[int, int],
    canonical_amplitude: float,
) -> np.ndarray:
    return canonical_director_q(
        resolution,
        equilibrium_magnitude=float(np.sqrt(-params.A / params.C)),
        theta0=float(rng.uniform(0.0, np.pi)),
        amplitude=canonical_amplitude,
        mode_x=canonical_mode[0],
        mode_y=canonical_mode[1],
        phase_x=float(rng.uniform(0.0, 2.0 * np.pi)),
        phase_y=float(rng.uniform(0.0, 2.0 * np.pi)),
    )


def fno_predict(
    model: torch.nn.Module,
    q0: np.ndarray,
    params: LdGParams,
    times: list[float],
    device: str,
) -> np.ndarray:
    batch = np.stack([make_input(q0, params, target_time) for target_time in times], axis=0)
    xb = torch.from_numpy(batch).float().to(device)
    with torch.no_grad():
        return model(xb).detach().cpu().numpy()


def evaluate_samples(args: argparse.Namespace, model: torch.nn.Module) -> tuple[list[EvalSample], float, float]:
    rng = np.random.default_rng(args.seed)
    canonical_mode = parse_canonical_mode(args.canonical_mode)
    mid_steps = max(1, args.steps // 2)
    times = [0.0, mid_steps * args.dt, args.steps * args.dt]
    samples: list[EvalSample] = []

    for index in range(args.eval_samples):
        params = LdGParams() if args.fixed_canonical_params else sample_params(rng)
        q0 = canonical_sample(
            rng,
            args.resolution,
            params,
            canonical_mode,
            args.canonical_amplitude,
        )
        mid = solve_ldg(q0, params=params, dt=args.dt, steps=mid_steps)
        final = solve_ldg(q0, params=params, dt=args.dt, steps=args.steps)
        pred = fno_predict(model, q0, params, times, args.device)
        relative_l2 = np.linalg.norm(pred[2] - final) / max(np.linalg.norm(final), 1.0e-8)
        samples.append(
            EvalSample(
                index=index,
                params=params,
                q0=q0,
                mid=mid,
                final=final,
                pred_mid=pred[1],
                pred_final=pred[2],
                relative_l2=float(relative_l2),
            )
        )

    errors = np.array([sample.relative_l2 for sample in samples])
    return samples, float(np.mean(errors)), float(np.median(errors))


def measure_representative_timing(
    args: argparse.Namespace,
    model: torch.nn.Module,
    sample: EvalSample,
) -> tuple[float, float, float]:
    mid_steps = max(1, args.steps // 2)
    times = [0.0, mid_steps * args.dt, args.steps * args.dt]

    t0 = time.perf_counter()
    _ = solve_ldg(sample.q0, params=sample.params, dt=args.dt, steps=mid_steps)
    _ = solve_ldg(sample.q0, params=sample.params, dt=args.dt, steps=args.steps)
    spectral_time = time.perf_counter() - t0

    synchronize_if_cuda(args.device)
    t1 = time.perf_counter()
    _ = fno_predict(model, sample.q0, sample.params, times, args.device)
    synchronize_if_cuda(args.device)
    fno_time = time.perf_counter() - t1

    speedup = spectral_time / max(fno_time, 1.0e-12)
    return float(spectral_time), float(fno_time), float(speedup)


def format_speedup(value: float | None) -> str:
    if value is None:
        return "not measured"
    return f"{value:.2f}x"


def main() -> None:
    args = parse_args()
    if args.eval_samples < 1:
        raise ValueError("--eval-samples must be at least 1")

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(
        latent_channels=int(ckpt.get("latent_channels", 32)),
        modes=int(ckpt.get("modes", 12)),
        require_physicsnemo=bool(args.require_physicsnemo),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    backend = ckpt.get("backend", getattr(model, "backend_name", "unknown backend"))
    if "PhysicsNeMo" not in str(backend):
        raise RuntimeError(f"expected a PhysicsNeMo backend, got {backend!r}")

    samples, mean_relative_l2, median_relative_l2 = evaluate_samples(args, model)
    if mean_relative_l2 > 0.1:
        raise RuntimeError(
            f"mean_relative_l2={mean_relative_l2:.6f} exceeds sanity bound 0.1"
        )

    chosen = min(samples, key=lambda sample: abs(sample.relative_l2 - median_relative_l2))
    spectral_time, fno_time, speedup = measure_representative_timing(args, model, chosen)

    cpu_speedup = args.caption_cpu_speedup_x
    gpu_speedup = args.caption_gpu_speedup_x
    if args.device == "cpu" and cpu_speedup is None:
        cpu_speedup = speedup
    if args.device.startswith("cuda") and gpu_speedup is None:
        gpu_speedup = speedup

    spectral = [chosen.q0, chosen.mid, chosen.final]
    nemo = [chosen.q0, chosen.pred_mid, chosen.pred_final]
    all_orders = [np.sqrt(q[0] ** 2 + q[1] ** 2) for q in spectral + nemo]
    vmin = float(min(np.min(order) for order in all_orders))
    vmax = float(max(np.max(order) for order in all_orders))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "figure.titlesize": 12,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.9), constrained_layout=False)
    labels = [
        "Initial\n$t = 0$",
        f"Middle\n$t = {0.5 * args.steps * args.dt:.3f}$",
        f"Final\n$t = {args.steps * args.dt:.3f}$",
    ]
    image = None
    for col, label in enumerate(labels):
        image = draw_director(
            axes[0, col],
            spectral[col],
            label,
            args.stride,
            vmin,
            vmax,
        )
        draw_director(
            axes[1, col],
            nemo[col],
            "",
            args.stride,
            vmin,
            vmax,
        )
    axes[0, 0].set_ylabel("Spectral\nreference", fontsize=10, labelpad=8)
    axes[1, 0].set_ylabel("PhysicsNeMo\nFNO", fontsize=10, labelpad=8)
    caption = (
        f"Mean rel. L2 = {mean_relative_l2:.6f} over n = {args.eval_samples}; "
        f"speedup CPU = {format_speedup(cpu_speedup)}, GPU = {format_speedup(gpu_speedup)}; "
        f"training trajectories = {args.dataset_size}."
    )
    fig.suptitle("Landau-de Gennes Elastic Relaxation: Spectral Solver vs PhysicsNeMo FNO")
    fig.subplots_adjust(left=0.08, right=0.86, top=0.86, bottom=0.14, wspace=0.04, hspace=0.08)
    if image is not None:
        cax = fig.add_axes([0.885, 0.20, 0.025, 0.60])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label(r"scalar order $|Q|$")
    fig.text(0.47, 0.045, caption, ha="center", va="center", fontsize=8.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if args.caption_cpu_speedup_x is not None and abs(args.caption_cpu_speedup_x - 35.0) > 5.0:
        print("warning: supplied CPU speedup differs materially from the ~35x draft text")
    if args.caption_gpu_speedup_x is not None and abs(args.caption_gpu_speedup_x - 53.0) > 5.0:
        print("warning: supplied GPU speedup differs materially from the ~53x draft text")

    print(f"wrote {args.out}")
    print(f"eval_samples: {args.eval_samples}")
    print(f"mean_relative_l2: {mean_relative_l2:.6f}")
    print(f"median_relative_l2: {median_relative_l2:.6f}")
    print(f"chosen_sample_index: {chosen.index}")
    print(f"chosen_sample_relative_l2: {chosen.relative_l2:.6f}")
    print(f"device: {args.device}")
    print(f"single_trajectory_speedup_x: {speedup:.2f}")
    print(f"spectral_time_s: {spectral_time:.6f}")
    print(f"fno_time_s: {fno_time:.6f}")
    print(f"backend: {backend}")


if __name__ == "__main__":
    main()
