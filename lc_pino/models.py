from __future__ import annotations

import importlib
import types
import warnings

import torch
from torch import nn


def build_model(
    in_channels: int = 7,
    out_channels: int = 2,
    latent_channels: int = 32,
    modes: int = 12,
    layers: int = 4,
    require_physicsnemo: bool = False,
) -> nn.Module:
    """Build a PhysicsNeMo FNO when available, otherwise use a small fallback.

    Pass require_physicsnemo=True to raise instead of falling back silently.
    """

    try:
        # PhysicsNeMo 2.0.0 can import Warp 1.13 on CPU-only Windows with a
        # stale type annotation that expects wp.context.Device. FNO does not
        # use that radius-search path, so this compatibility shim keeps the
        # neural-operator import usable without changing PhysicsNeMo itself.
        import warp

        if not hasattr(warp, "context"):
            warp.context = types.SimpleNamespace(Device=object)
        fno_mod = importlib.import_module("physicsnemo.models.fno")
        FNO = getattr(fno_mod, "FNO")
        core = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            dimension=2,
            latent_channels=latent_channels,
            num_fno_layers=layers,
            num_fno_modes=modes,
            decoder_layers=2,
            decoder_layer_size=latent_channels,
            padding=0,
            coord_features=False,
        )
        model = ResidualQModel(core, residual_scale=1.0)
        model.backend_name = "PhysicsNeMo FNO"
        return model
    except Exception as e:
        msg = (
            f"PhysicsNeMo FNO unavailable ({type(e).__name__}: {e}); "
            "falling back to local TinyFNO."
        )
        if require_physicsnemo:
            raise RuntimeError(
                f"--require-physicsnemo set but PhysicsNeMo import failed "
                f"({type(e).__name__}: {e})"
            ) from e
        warnings.warn(msg, stacklevel=2)
        core = TinyFNO2d(in_channels, out_channels, width=latent_channels, modes=modes, layers=layers)
        model = ResidualQModel(core, residual_scale=1.0)
        model.backend_name = "local TinyFNO fallback"
        return model


class ResidualQModel(nn.Module):
    """Predict a correction to the initial Q field instead of the full field."""

    def __init__(self, core: nn.Module, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.core = core
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :2] + self.residual_scale * self.core(x)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            nx,
            ny // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1 = min(self.modes, nx)
        m2 = min(self.modes, ny // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(nx, ny))


class TinyFNO2d(nn.Module):
    """Small local FNO-like model used only when PhysicsNeMo is not installed."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        modes: int = 12,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            x = torch.nn.functional.gelu(spectral(x) + local(x))
        return self.project(x)


def spectral_laplacian_torch(q: torch.Tensor, length: float = 1.0) -> torch.Tensor:
    """Periodic spectral Laplacian matching the NumPy reference solver."""

    nx, ny = q.shape[-2:]
    device = q.device
    dtype = q.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=length / nx, device=device)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(ny, d=length / ny, device=device)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    qhat = torch.fft.fft2(q, dim=(-2, -1))
    lap = torch.fft.ifft2(-k2.to(dtype=dtype)[None, None, ...] * qhat, dim=(-2, -1)).real
    return lap


def periodic_laplacian_torch(q: torch.Tensor, dx: float) -> torch.Tensor:
    """Deprecated finite-difference Laplacian kept for compatibility."""

    return (
        torch.roll(q, 1, dims=-2)
        + torch.roll(q, -1, dims=-2)
        + torch.roll(q, 1, dims=-1)
        + torch.roll(q, -1, dims=-1)
        - 4.0 * q
    ) / (dx * dx)


def ldg_rhs_torch(q: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    A = inputs[:, 2:3]
    C = inputs[:, 3:4]
    L = inputs[:, 4:5]
    gamma = inputs[:, 5:6]
    r2 = q[:, 0:1] ** 2 + q[:, 1:2] ** 2
    return gamma * (L * spectral_laplacian_torch(q) - A * q - C * r2 * q)


def endpoint_physics_residual(inputs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    q0 = inputs[:, :2]
    A = inputs[:, 2:3]
    C = inputs[:, 3:4]
    L = inputs[:, 4:5]
    gamma = inputs[:, 5:6]
    target_time = torch.clamp(inputs[:, 6:7], min=1.0e-6)
    r2 = pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2
    rhs = gamma * (L * spectral_laplacian_torch(pred) - A * pred - C * r2 * pred)
    return (pred - q0) / target_time - rhs


def pino_time_residual(
    model: nn.Module,
    inputs: torch.Tensor,
    time_delta: float = 1.0e-3,
) -> torch.Tensor:
    """Local-in-time PDE residual for an endpoint operator Q0,t -> Q(t).

    The old endpoint residual used (Q(t)-Q(0))/t as dQ/dt, which is a poor
    approximation for the t=0.1 benchmark. This residual instead evaluates the
    model at nearby target times and forms a local backward finite difference.
    """

    target_time = inputs[:, 6:7]
    dt = torch.minimum(
        torch.full_like(target_time, float(time_delta)),
        torch.clamp(target_time, min=1.0e-6),
    )
    current_inputs = inputs
    previous_inputs = inputs.clone()
    previous_inputs[:, 6:7] = target_time - dt
    q_now = model(current_inputs)
    q_prev = model(previous_inputs)
    time_derivative = (q_now - q_prev) / dt
    rhs = ldg_rhs_torch(q_now, current_inputs)
    return time_derivative - rhs
