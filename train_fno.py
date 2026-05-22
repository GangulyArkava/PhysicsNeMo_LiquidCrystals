from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from lc_pino.models import build_model, endpoint_physics_residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an FNO surrogate for Q-tensor relaxation.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lc_fno.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--physics-weight", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    return parser.parse_args()


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = torch.linalg.vector_norm((pred - target).flatten(1), dim=1)
    den = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1.0e-8)
    return torch.mean(num / den)


def main() -> None:
    args = parse_args()
    raw = np.load(args.data)
    x = torch.from_numpy(raw["inputs"]).float()
    y = torch.from_numpy(raw["targets"]).float()
    dataset = TensorDataset(x, y)
    val_count = max(1, int(0.15 * len(dataset)))
    train_count = len(dataset) - val_count
    train_ds, val_ds = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model(latent_channels=args.latent_channels, modes=args.modes).to(args.device)
    backend = getattr(model, "backend_name", model.__class__.__name__)
    print(f"model_backend={backend}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    mse = torch.nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        iterator = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False) if tqdm else train_loader
        for xb, yb in iterator:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            pred = model(xb)
            data_loss = mse(pred, yb)
            phys = endpoint_physics_residual(xb, pred)
            phys_loss = torch.mean(phys**2)
            loss = data_loss + args.physics_weight * phys_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.detach()) * xb.shape[0]
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_rel = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                pred = model(xb)
                val_loss += float(mse(pred, yb)) * xb.shape[0]
                val_rel += float(relative_l2(pred, yb)) * xb.shape[0]
        train_loss /= train_count
        val_loss /= val_count
        val_rel /= val_count
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4e} "
            f"val_mse={val_loss:.4e} val_rel_l2={val_rel:.4e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "latent_channels": args.latent_channels,
                    "modes": args.modes,
                    "resolution": x.shape[-1],
                    "best_val_mse": best_val,
                    "backend": backend,
                },
                args.checkpoint,
            )
            print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
