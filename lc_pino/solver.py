from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LdGParams:
    """Parameters for the minimal Landau-de Gennes gradient-flow model."""

    A: float = -0.1
    C: float = 1.0
    L: float = 0.02
    gamma: float = 1.0


def wave_numbers(resolution: int, length: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    freq = 2.0 * np.pi * np.fft.fftfreq(resolution, d=length / resolution)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    return kx, ky


def smooth_random_q(
    resolution: int,
    rng: np.random.Generator,
    amplitude: float = 0.35,
    lowpass: float = 0.12,
) -> np.ndarray:
    """Create a smooth random Q field with shape (2, resolution, resolution)."""

    q = rng.normal(size=(2, resolution, resolution))
    kx, ky = wave_numbers(resolution)
    k2 = kx**2 + ky**2
    k2_max = max(float(np.max(k2)), 1.0)
    filt = np.exp(-lowpass * k2 / k2_max)
    qhat = np.fft.fft2(q, axes=(-2, -1)) * filt
    q = np.fft.ifft2(qhat, axes=(-2, -1)).real
    scale = np.std(q, axis=(-2, -1), keepdims=True) + 1.0e-8
    return amplitude * q / scale


def defect_pair_q(
    resolution: int,
    rng: np.random.Generator,
    scalar_order: float = 0.32,
    noise: float = 0.04,
) -> np.ndarray:
    """Create a simple +/- half-defect-like Q field on a periodic square."""

    x = np.linspace(0.0, 1.0, resolution, endpoint=False)
    y = np.linspace(0.0, 1.0, resolution, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    c1 = np.array([0.33, 0.5]) + rng.normal(scale=0.04, size=2)
    c2 = np.array([0.67, 0.5]) + rng.normal(scale=0.04, size=2)
    theta = 0.5 * np.arctan2(yy - c1[1], xx - c1[0])
    theta -= 0.5 * np.arctan2(yy - c2[1], xx - c2[0])
    theta += rng.uniform(0.0, np.pi)
    q1 = scalar_order * np.cos(2.0 * theta)
    q2 = scalar_order * np.sin(2.0 * theta)
    q = np.stack([q1, q2], axis=0)
    q += noise * smooth_random_q(resolution, rng, amplitude=1.0)
    return q.astype(np.float32)


def rhs(q: np.ndarray, params: LdGParams, length: float = 1.0) -> np.ndarray:
    """Evaluate dq/dt for diagnostic use."""

    kx, ky = wave_numbers(q.shape[-1], length)
    k2 = kx**2 + ky**2
    lap = np.fft.ifft2(-k2 * np.fft.fft2(q, axes=(-2, -1)), axes=(-2, -1)).real
    r2 = q[0] ** 2 + q[1] ** 2
    return params.gamma * (params.L * lap - params.A * q - params.C * r2[None, ...] * q)


def solve_ldg(
    q0: np.ndarray,
    params: LdGParams,
    dt: float,
    steps: int,
    length: float = 1.0,
    keep_every: int | None = None,
) -> np.ndarray:
    """Semi-implicit spectral solve for the minimal Landau-de Gennes dynamics."""

    q = np.asarray(q0, dtype=np.float64).copy()
    n = q.shape[-1]
    _, _ = q.shape[-2:]
    kx, ky = wave_numbers(n, length)
    k2 = kx**2 + ky**2
    denom = 1.0 + dt * params.gamma * (params.A + params.L * k2)
    frames: list[np.ndarray] = []

    if keep_every:
        frames.append(q.astype(np.float32))

    for step in range(1, steps + 1):
        r2 = q[0] ** 2 + q[1] ** 2
        explicit = q - dt * params.gamma * params.C * r2[None, ...] * q
        qhat = np.fft.fft2(explicit, axes=(-2, -1)) / denom[None, ...]
        q = np.fft.ifft2(qhat, axes=(-2, -1)).real
        if keep_every and step % keep_every == 0:
            frames.append(q.astype(np.float32))

    if keep_every:
        return np.stack(frames, axis=0)
    return q.astype(np.float32)


def free_energy(q: np.ndarray, params: LdGParams, length: float = 1.0) -> float:
    n = q.shape[-1]
    dx = length / n
    kx, ky = wave_numbers(n, length)
    qhat = np.fft.fft2(q, axes=(-2, -1))
    dqx = np.fft.ifft2(1j * kx * qhat, axes=(-2, -1)).real
    dqy = np.fft.ifft2(1j * ky * qhat, axes=(-2, -1)).real
    r2 = q[0] ** 2 + q[1] ** 2
    bulk = 0.5 * params.A * r2 + 0.25 * params.C * r2**2
    elastic = 0.5 * params.L * np.sum(dqx**2 + dqy**2, axis=0)
    return float(np.sum(bulk + elastic) * dx * dx)
