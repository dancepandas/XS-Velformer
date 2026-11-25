#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data: load → split → standardize → mask → forward.

Usage:
    python examples/make_synthetic_sections.py --out examples/sections_synthetic.pkl
    python examples/quickstart.py --data examples/sections_synthetic.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import ModelConfig                      # noqa: E402
from data.preprocess import SectionPreprocessor     # noqa: E402
from evaluation import test_cases as masks          # noqa: E402
from models.query_transformer import HydroQueryTransformer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pre = SectionPreprocessor()
    samples = pre.load_sections(args.data)
    train, val, test = pre.split_by_station_and_time(
        samples, train_ratio=0.8, val_ratio=0.1, seed=42, require_time=True)
    if not test:
        test = samples[:1]
    pre.fit_global_stats(train or samples)
    for sample in samples:
        pre.apply_global_norm(sample)
    print(f"[split] train={len(train)} val={len(val)} test={len(test)}")

    cfg = ModelConfig(d_model=64, num_heads=4, context_encoder_layers=1,
                      query_encoder_layers=1, cross_attention_layers=1,
                      ffn_dim=128)
    model = HydroQueryTransformer(cfg).to(args.device).eval()

    sample = test[0]
    rng = np.random.default_rng(42)
    context_idx, query_idx = masks._random_mask(sample.K, 0.3, rng)
    context, query, target = masks._build_features(sample, context_idx, query_idx)
    batch = {
        "context": torch.as_tensor(context, dtype=torch.float32)[None].to(args.device),
        "context_mask": torch.ones(1, len(context), dtype=torch.bool, device=args.device),
        "query": torch.as_tensor(query, dtype=torch.float32)[None].to(args.device),
        "query_mask": torch.ones(1, len(query), dtype=torch.bool, device=args.device),
    }
    with torch.no_grad():
        prediction = model(batch)["v_norm_pred"][0].cpu().numpy()

    mean = pre.global_stats["v"]["mean"]
    std = pre.global_stats["v"]["std"]
    rmse_norm = float(np.sqrt(np.mean((prediction - target) ** 2)))
    rmse_phys = rmse_norm * std
    print(f"[forward] section K={sample.K}, context={len(context)}, query={len(query)}")
    print(f"[forward] normalized RMSE={rmse_norm:.4f}  physical RMSE={rmse_phys:.4f} m/s")
    print(f"[forward] sample predictions (m/s): "
          f"{np.round(prediction[:5] * std + mean, 3).tolist()}")
    print("[done] pipeline works; see main.py for training and evaluation/ for metrics")


if __name__ == "__main__":
    main()
