"""
可视化脚本: 完整断面 → 随机 mask → 模型预测 → 对比展示

支持所有通过 model_adapters 注册的模型类型。

用法:
    python -m evaluation.visualize --model_type query_transformer --checkpoint checkpoints/model.pt
    python -m evaluation.visualize --model_type lstm --checkpoint checkpoints/lstm_baseline.pt --mask_ratio 0.3
    python -m evaluation.visualize --model_type nearest_x --strategy continuous
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.preprocess import SectionPreprocessor, compute_bank_and_grad_features
from evaluation.model_adapters import create_adapter
from evaluation.metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="流速预测可视化")
    parser.add_argument("--model_type", type=str, default="query_transformer",
                        help="模型类型 (query_transformer, lstm, gan_z0, diffusion_mean5, nearest_x, ...)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt",
                        help="checkpoint 路径 (传统插值方法可忽略)")
    parser.add_argument("--data", type=str, default="v_data/processed/sections.pkl")
    parser.add_argument("--section_idx", type=int, default=None, help="断面索引, 不指定则随机")
    parser.add_argument("--mask_ratio", type=float, default=0.3, help="mask 比例")
    parser.add_argument("--strategy", type=str, default="random",
                        choices=["random", "continuous", "edge"], help="mask 策略")
    parser.add_argument("--output_dir", type=str, default="result/vis")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. 加载数据 ──────────────────────────────────────
    preprocessor = SectionPreprocessor()
    samples = preprocessor.load_sections(args.data)

    # 尝试从 checkpoint 恢复全局统计量
    adapter = create_adapter(args.model_type, device=device)
    if adapter.requires_checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "global_stats" in ckpt:
            preprocessor.global_stats = ckpt["global_stats"]
        for s in samples:
            preprocessor.apply_global_norm(s)

    # 选择断面
    n_sections = len(samples)
    idx = args.section_idx if args.section_idx is not None else np.random.randint(0, n_sections)
    sample = samples[idx]
    K = sample.K
    print(f"断面索引: {idx}/{n_sections} (station={sample.station_id})")
    print(f"测速线数 K={K}, 水位 Z={sample.Z:.2f}, 河宽 W={sample.W:.1f}, 最大水深 h_max={sample.h_max:.2f}")

    n_mask = max(1, int(K * args.mask_ratio))

    # ── 2. 生成 mask ────────────────────────────────────
    if args.strategy == "random":
        query_idx = np.sort(np.random.choice(K, n_mask, replace=False))
    elif args.strategy == "continuous":
        start = np.random.randint(0, max(K - n_mask + 1, 1))
        query_idx = np.arange(start, start + n_mask)
    elif args.strategy == "edge":
        side = np.random.choice(["left", "right"])
        if side == "left":
            query_idx = np.arange(n_mask)
        else:
            query_idx = np.arange(K - n_mask, K)

    context_idx = np.array([j for j in range(K) if j not in set(query_idx)])
    print(f"mask 策略: {args.strategy}, mask 比例: {args.mask_ratio:.0%}")
    print(f"context: {len(context_idx)} 条, query: {len(query_idx)} 条")

    # ── 3. 构造模型输入 (8/7 维) ──────────────────────────
    global_feat = np.array([sample.Z_norm, sample.h_max_norm, sample.W_norm])

    context = np.column_stack([
        sample.x_norm[context_idx],
        sample.d_norm[context_idx],
        sample.bank_norm[context_idx],
        sample.grad_norm[context_idx],
        sample.v_norm[context_idx],
        np.tile(global_feat, (len(context_idx), 1)),
    ])
    query = np.column_stack([
        sample.x_norm[query_idx],
        sample.d_norm[query_idx],
        sample.bank_norm[query_idx],
        sample.grad_norm[query_idx],
        np.tile(global_feat, (len(query_idx), 1)),
    ])
    target = sample.v_norm[query_idx]

    batch = {
        "context": torch.tensor(context, dtype=torch.float32).unsqueeze(0),
        "context_mask": torch.ones(1, len(context_idx), dtype=torch.bool),
        "query": torch.tensor(query, dtype=torch.float32).unsqueeze(0),
        "query_mask": torch.ones(1, len(query_idx), dtype=torch.bool),
        "target": torch.tensor(target, dtype=torch.float32).unsqueeze(0),
    }

    # ── 4. 模型预测 ──────────────────────────────────────
    if adapter.requires_checkpoint:
        adapter.load(args.checkpoint)

    v_norm_pred = adapter.predict(batch).cpu().numpy()[0]

    # 反归一化
    v_mean = preprocessor.global_stats.get("v", {}).get("mean", 0.0)
    v_std = preprocessor.global_stats.get("v", {}).get("std", 1.0)
    v_pred = v_norm_pred * v_std + v_mean

    # ── 5. 计算指标 ──────────────────────────────────────
    v_true = sample.v_raw[query_idx]
    v_mean = preprocessor.global_stats.get("v", {}).get("mean", 0.0)
    v_std = preprocessor.global_stats.get("v", {}).get("std", 1.0)
    full_mask = np.ones(len(query_idx), dtype=np.float32)
    case_metrics = compute_all_metrics(v_norm_pred, target, full_mask, v_mean, v_std)
    mae = case_metrics["mae_raw"]
    rmse = case_metrics["rmse_raw"]
    nse = case_metrics["nse_raw"]
    kge = case_metrics["kge_raw"]
    r2 = case_metrics["r2_raw"]
    print(f"\n预测指标: MAE={mae:.4f} m/s, RMSE={rmse:.4f} m/s, "
          f"NSE={nse:.4f}, KGE={kge:.4f}, R2={r2:.4f}")

    # ── 6. 绘图 ──────────────────────────────────────────
    x_full = sample.x_raw
    v_full = sample.v_raw
    x_ctx = sample.x_raw[context_idx]
    v_ctx = sample.v_raw[context_idx]
    x_qry = sample.x_raw[query_idx]
    v_qry_true = sample.v_raw[query_idx]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # --- 上图: 流速对比 ---
    ax = axes[0]
    ax.plot(x_full, v_full, "k-", linewidth=1.5, alpha=0.4, label="Full ground truth", zorder=1)
    ax.scatter(x_ctx, v_ctx, c="blue", s=40, zorder=3, label=f"Context ({len(context_idx)} pts)")
    ax.scatter(x_qry, v_qry_true, c="green", s=50, marker="x", linewidths=2, zorder=4, label="Ground truth (masked)")
    ax.scatter(x_qry, v_pred, c="red", s=50, marker="o", zorder=5, label=f"{args.model_type} prediction")

    for j in range(len(query_idx)):
        ax.plot([x_qry[j], x_qry[j]], [v_qry_true[j], v_pred[j]],
                "r-", alpha=0.3, linewidth=1)

    ax.set_xlabel("Distance from left bank (m)", fontsize=12)
    ax.set_ylabel("Velocity (m/s)", fontsize=12)
    ax.set_title(f"Section {idx} (station={sample.station_id}, K={K}) | "
                 f"{args.strategy} mask {args.mask_ratio:.0%} | "
                 f"RMSE={rmse:.4f} NSE={nse:.4f} KGE={kge:.4f}", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- 下图: 误差分布 ---
    ax = axes[1]
    errors = v_pred - v_qry_true
    ax.bar(x_qry, errors, width=1.5, color="red", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Distance from left bank (m)", fontsize=12)
    ax.set_ylabel("Prediction error (m/s)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.model_type}_sec{idx}_{args.strategy}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n可视化已保存: {out_path}")


if __name__ == "__main__":
    main()
