from __future__ import annotations

import numpy as np

from .solver import LdGParams, canonical_director_q, defect_pair_q, smooth_random_q, solve_ldg


def sample_params(rng: np.random.Generator) -> LdGParams:
    return LdGParams(
        A=float(rng.uniform(-0.30, -0.05)),
        C=float(rng.uniform(0.7, 1.3)),
        L=float(rng.uniform(0.01, 0.04)),
        gamma=float(rng.uniform(0.7, 1.3)),
    )


def make_input(q0: np.ndarray, params: LdGParams, target_time: float) -> np.ndarray:
    n = q0.shape[-1]
    channels = [
        q0[0],
        q0[1],
        np.full((n, n), params.A, dtype=np.float32),
        np.full((n, n), params.C, dtype=np.float32),
        np.full((n, n), params.L, dtype=np.float32),
        np.full((n, n), params.gamma, dtype=np.float32),
        np.full((n, n), target_time, dtype=np.float32),
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def generate_dataset(
    samples: int,
    resolution: int,
    steps: int,
    dt: float,
    seed: int,
    corr_modes: float = 2.0,
    init_mode: str = "mixed",
    canonical_mode: tuple[int, int] = (3, 2),
    canonical_amplitude: float = 0.8,
    fixed_canonical_params: bool = False,
    fixed_target_time: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = np.empty((samples, 7, resolution, resolution), dtype=np.float32)
    targets = np.empty((samples, 2, resolution, resolution), dtype=np.float32)
    params_arr = np.empty((samples, 5), dtype=np.float32)

    for i in range(samples):
        params = LdGParams() if fixed_canonical_params else sample_params(rng)
        equilibrium_order = float(np.sqrt(-params.A / params.C))
        if init_mode == "canonical":
            q0 = canonical_director_q(
                resolution,
                equilibrium_magnitude=equilibrium_order,
                theta0=float(rng.uniform(0.0, np.pi)),
                amplitude=canonical_amplitude,
                mode_x=canonical_mode[0],
                mode_y=canonical_mode[1],
                phase_x=float(rng.uniform(0.0, 2.0 * np.pi)),
                phase_y=float(rng.uniform(0.0, 2.0 * np.pi)),
            )
        elif init_mode == "defect":
            q0 = defect_pair_q(resolution, rng, scalar_order=equilibrium_order)
        elif init_mode == "smooth":
            # Perturb around nematic equilibrium so short integrations stay nematic
            q0 = smooth_random_q(
                resolution, rng, amplitude=0.05,
                corr_modes=corr_modes, equilibrium_magnitude=equilibrium_order,
            )
        elif init_mode == "mixed":
            if rng.random() < 0.35:
                q0 = defect_pair_q(resolution, rng, scalar_order=equilibrium_order)
            else:
                q0 = smooth_random_q(
                    resolution, rng, amplitude=0.05,
                    corr_modes=corr_modes, equilibrium_magnitude=equilibrium_order,
                )
        else:
            raise ValueError(f"unknown init_mode={init_mode!r}")
        local_steps = steps if fixed_target_time else int(rng.integers(max(2, steps // 3), steps + 1))
        target_time = local_steps * dt
        qT = solve_ldg(q0, params=params, dt=dt, steps=local_steps)
        inputs[i] = make_input(q0, params, target_time)
        targets[i] = qT
        params_arr[i] = np.array(
            [params.A, params.C, params.L, params.gamma, target_time],
            dtype=np.float32,
        )

    return {
        "inputs": inputs,
        "targets": targets,
        "params": params_arr,
        "dt": np.array(dt, dtype=np.float32),
    }
