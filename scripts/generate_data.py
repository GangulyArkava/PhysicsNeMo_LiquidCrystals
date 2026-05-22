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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Landau-de Gennes Q-tensor data.")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("data/lc64_train.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    remaining = args.samples
    seed = args.seed
    progress = tqdm(total=args.samples, desc="generating") if tqdm else None
    try:
        while remaining > 0:
            batch = min(128, remaining)
            data = generate_dataset(batch, args.resolution, args.steps, args.dt, seed)
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


if __name__ == "__main__":
    main()
