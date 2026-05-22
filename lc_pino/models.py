from __future__ import annotations

import importlib
import types

import torch
from torch import nn


def build_model(
    in_channels: int = 7,
    out_channels: int = 2,
    latent_channels: int = 32,
    modes: int = 12,
    layers: int = 4,
) -> nn.Module:
    """Build a PhysicsNeMo FNO when available, otherwise use a small fallback."""

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
    except Exception:
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


def periodic_laplacian_torch(q: torch.Tensor, dx: float) -> torch.Tensor:
    return (
        torch.roll(q, 1, dims=-2)
        + torch.roll(q, -1, dims=-2)
        + torch.roll(q, 1, dims=-1)
        + torch.roll(q, -1, dims=-1)
        - 4.0 * q
    ) / (dx * dx)


def endpoint_physics_residual(inputs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    q0 = inputs[:, :2]
    A = inputs[:, 2:3]
    C = inputs[:, 3:4]
    L = inputs[:, 4:5]
    gamma = inputs[:, 5:6]
    target_time = torch.clamp(inputs[:, 6:7], min=1.0e-6)
    dx = 1.0 / pred.shape[-1]
    r2 = pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2
    rhs = gamma * (L * periodic_laplacian_torch(pred, dx) - A * pred - C * r2 * pred)
    return (pred - q0) / target_time - rhs
