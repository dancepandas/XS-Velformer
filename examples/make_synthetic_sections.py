#!/usr/bin/env python3
"""Generate a synthetic cross-section dataset in the format used by the model.

The generated sections follow a smooth parabolic depth profile and a
monotone depth–velocity relation with multiplicative noise, so the quickstart
exercises realistic-enough shapes without any proprietary data.

Usage:
    python examples/make_synthetic_sections.py --out examples/sections_synthetic.pkl
"""

import argparse
import pickle
from datetime import datetime, timedelta

import numpy as np


def make_sections(n_sections, n_stations, seed):
    rng = np.random.default_rng(seed)
    records = []
    base = datetime(2025, 1, 1, 8, 0, 0)
    for i in range(n_sections):
        station = f"S{i % n_stations:02d}"
        width = float(rng.uniform(20.0, 200.0))
        k = int(rng.integers(20, 70))
        x = np.linspace(0.0, width, k)
        # parabolic cross section with a flat-ish thalweg
        shape = np.clip(1.0 - (2.0 * x / width - 1.0) ** 2, 0.0, 1.0) ** 0.7
        h_max = float(rng.uniform(1.0, 8.0))
        depth = 0.05 + h_max * shape
        v_max = float(rng.uniform(0.5, 3.0))
        velocity = v_max * (depth / h_max) ** 0.6
        velocity = velocity * (0.85 + 0.3 * rng.random(k))
        velocity = np.clip(velocity, 0.0, None)
        # a few zero-velocity lines near the banks, as in real sections
        velocity[(x < 0.03 * width) | (x > 0.97 * width)] = 0.0
        records.append({
            "x": x.astype(np.float64),
            "h": depth.astype(np.float64),
            "v": velocity.astype(np.float64),
            "Z": float(rng.uniform(5.0, 30.0)),
            "station_id": station,
            "time": (base + timedelta(hours=6 * i)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="examples/sections_synthetic.pkl")
    parser.add_argument("--sections", type=int, default=60)
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    records = make_sections(args.sections, args.stations, args.seed)
    with open(args.out, "wb") as handle:
        pickle.dump(records, handle)
    lengths = [len(r["x"]) for r in records]
    print(f"[synthetic] {len(records)} sections, {args.stations} stations, "
          f"K={min(lengths)}-{max(lengths)} → {args.out}")


if __name__ == "__main__":
    main()
