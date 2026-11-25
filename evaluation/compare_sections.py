"""
多模型断面流速对比可视化

从固定测试集 (result/fixed_test_cases.pkl) 中选取断面,
在同一张图上展示所有模型的预测结果, 确保使用的是测试集数据。

用法:
    # 随机选取 3 个测试断面
    python -m evaluation.compare_sections --n_sections 3

    # 指定策略和比例筛选
    python -m evaluation.compare_sections --strategy edge --mask_ratio 0.3 --n_sections 3

    # 只对比部分模型
    python -m evaluation.compare_sections --models hydro_qt,lstm,linear_x,idw_xd
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.model_adapters import create_adapter, _ensure_interpolation_loaded
from evaluation.metrics import compute_all_metrics

MODELS_CONFIG = [
    ("hydro_qt", "checkpoints/model.pt", "Hydro-QT", "#e74c3c", "o"),
    ("lstm", "checkpoints/lstm_baseline.pt", "LSTM", "#3498db", "s"),
    ("gan_z0", "checkpoints/gan_baseline.pt", "GAN (z=0)", "#2ecc71", "D"),
    ("gan_mean5", "checkpoints/gan_baseline.pt", "GAN (mean5)", "#27ae60", "d"),
    ("diffusion_mean5", "checkpoints/diffusion_baseline.pt", "Diffusion", "#9b59b6", "^"),
    ("nearest_x", "", "Nearest-X", "#f39c12", "v"),
    ("linear_x", "", "Linear-X", "#e67e22", "P"),
    ("idw_x", "", "IDW-X", "#1abc9c", "p"),
    ("idw_xd", "", "IDW-XD", "#16a085", "h"),
]


def _load_adapters(model_names, device):
    _ensure_interpolation_loaded()
    adapters = {}
    for mtype, ckpt, _, _, _ in MODELS_CONFIG:
        if mtype not in model_names:
            continue
        adapter = create_adapter(mtype, device=device)
        if adapter.requires_checkpoint and ckpt and os.path.exists(ckpt):
            adapter.load(ckpt)
            adapters[mtype] = adapter
        elif not adapter.requires_checkpoint:
            adapters[mtype] = adapter
        elif adapter.requires_checkpoint:
            print(f"  [SKIP] {mtype}: checkpoint {ckpt} not found")
    return adapters


def _predict_adapter(adapter, case, device):
    batch = {
        "context": torch.tensor(case["context"], dtype=torch.float32).unsqueeze(0),
        "context_mask": torch.ones(1, case["context"].shape[0], dtype=torch.bool),
        "query": torch.tensor(case["query"], dtype=torch.float32).unsqueeze(0),
        "query_mask": torch.ones(1, case["query"].shape[0], dtype=torch.bool),
        "target": torch.tensor(case["target"], dtype=torch.float32).unsqueeze(0),
    }
    with torch.no_grad():
        v_norm_pred = adapter.predict(batch).cpu().numpy()[0]
    return v_norm_pred


def _load_test_data(test_cases_path):
    with open(test_cases_path, "rb") as f:
        payload = pickle.load(f)
    return payload["test_cases"], payload["global_stats"], payload.get("meta", {})


def _reconstruct_full_section(case, v_mean, v_std):
    context_idx = np.array(case["context_idx"])
    query_idx = np.array(case["query_idx"])
    K = case["section_length"]

    ctx = case["context"]
    qry = case["query"]

    x_norm = np.zeros(K)
    d_norm = np.zeros(K)
    v_norm = np.zeros(K)
    v_norm[context_idx] = ctx[:, 0] if ctx.shape[1] >= 1 else 0
    d_norm_full = np.zeros(K)

    for i, ci in enumerate(context_idx):
        x_norm[ci] = ctx[i, 0]
        d_norm[ci] = ctx[i, 1]
        v_norm[ci] = ctx[i, 4]

    for i, qi in enumerate(query_idx):
        x_norm[qi] = qry[i, 0]
        d_norm[qi] = qry[i, 1]

    x_raw = x_norm * case.get("_W", 100.0)
    v_raw = v_norm * v_std + v_mean

    return x_raw, v_raw, v_norm, context_idx, query_idx


def plot_case(case, adapters, v_mean, v_std, output_dir, model_configs, device):
    context_idx = np.array(case["context_idx"])
    query_idx = np.array(case["query_idx"])
    K = case["section_length"]

    ctx = case["context"]
    qry = case["query"]
    target_norm = case["target"]

    x_norm_full = np.zeros(K)
    d_norm_full = np.zeros(K)
    v_norm_full = np.zeros(K)

    for i, ci in enumerate(context_idx):
        x_norm_full[ci] = ctx[i, 0]
        d_norm_full[ci] = ctx[i, 1]
        v_norm_full[ci] = ctx[i, 4]

    for i, qi in enumerate(query_idx):
        x_norm_full[qi] = qry[i, 0]
        d_norm_full[qi] = qry[i, 1]

    x_raw = x_norm_full * _infer_width(case)
    v_raw_full = v_norm_full * v_std + v_mean

    x_qry_raw = x_raw[query_idx]
    v_qry_true_raw = v_norm_full[query_idx] * v_std + v_mean
    v_ctx_raw = v_raw_full[context_idx]
    x_ctx_raw = x_raw[context_idx]

    full_mask = np.ones(len(query_idx), dtype=np.float32)

    predictions = {}
    metrics = {}
    for mtype, _, display_name, _, _ in model_configs:
        if mtype not in adapters:
            continue
        v_norm_pred = _predict_adapter(adapters[mtype], case, device)
        v_raw_pred = v_norm_pred * v_std + v_mean
        predictions[mtype] = (display_name, x_qry_raw, v_raw_pred)
        case_m = compute_all_metrics(v_norm_pred, target_norm, full_mask, v_mean, v_std)
        metrics[mtype] = case_m

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(x_raw, v_raw_full, "k-o", linewidth=1.5, markersize=4, alpha=0.6,
            label="Ground truth", zorder=1)

    for mtype, _, display_name, color, marker in model_configs:
        if mtype not in predictions:
            continue
        _, x_p, v_p = predictions[mtype]
        m = metrics[mtype]
        label = f"{display_name} (RMSE={m['rmse_raw']:.3f}, NSE={m['nse_raw']:.3f})"
        x_line = np.concatenate([x_ctx_raw, x_p])
        v_line = np.concatenate([v_ctx_raw, v_p])
        sort_idx = np.argsort(x_line)
        ax.plot(x_line[sort_idx], v_line[sort_idx], color=color, linestyle="--",
                linewidth=1.0, alpha=0.45, zorder=2)
        ax.scatter(x_p, v_p, c=color, s=36, marker=marker,
                   edgecolors="white", linewidths=0.4, zorder=3, label=label)

    ax.set_xlabel("Distance from left bank (m)", fontsize=12)
    ax.set_ylabel("Velocity (m/s)", fontsize=12)
    ax.set_title(
        f"{case['case_id']} | {case['strategy']} mask {case['mask_ratio']:.0%}",
        fontsize=13,
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08)

    safe_id = case["case_id"].replace("/", "_").replace("\\", "_")
    out_path = os.path.join(output_dir, f"{safe_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _infer_width(case):
    ctx = case["context"]
    if ctx.shape[0] < 2:
        return 100.0
    x_vals = ctx[:, 0]
    return (x_vals.max() - x_vals.min()) / max((ctx.shape[0] - 1), 1) * case["section_length"]


def main():
    parser = argparse.ArgumentParser(description="多模型断面流速对比可视化 (测试集)")
    parser.add_argument("--test_cases", type=str, default="result/fixed_test_cases.pkl",
                        help="固定测试集 pkl 路径")
    parser.add_argument("--n_sections", type=int, default=3,
                        help="选取的断面数量")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["random", "continuous", "edge", "sparse_reconstruct", "densify"],
                        help="筛选指定 mask 策略")
    parser.add_argument("--mask_ratio", type=float, default=None,
                        help="筛选指定 mask 比例")
    parser.add_argument("--models", type=str, default=None,
                        help="指定模型, 逗号分隔, 如 hydro_qt,lstm,linear_x")
    parser.add_argument("--output_dir", type=str, default="result/compare")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.models:
        model_names = set(m.strip() for m in args.models.split(","))
    else:
        model_names = set(cfg[0] for cfg in MODELS_CONFIG)

    model_configs = [cfg for cfg in MODELS_CONFIG if cfg[0] in model_names]

    print(f"Loading adapters: {[c[0] for c in model_configs]}")
    adapters = _load_adapters(model_names, device)
    print(f"  Loaded: {list(adapters.keys())}")

    print(f"Loading test cases from {args.test_cases} ...")
    test_cases, global_stats, meta = _load_test_data(args.test_cases)
    print(f"  Total: {len(test_cases)} cases, strategies: {meta.get('strategies', [])}")

    v_mean = global_stats.get("v", {}).get("mean", 0.0)
    v_std = global_stats.get("v", {}).get("std", 1.0)

    filtered = test_cases
    if args.strategy:
        filtered = [c for c in filtered if c["strategy"] == args.strategy]
        print(f"  Filtered by strategy={args.strategy}: {len(filtered)} cases")
    if args.mask_ratio is not None:
        filtered = [c for c in filtered if abs(c["mask_ratio"] - args.mask_ratio) < 0.01]
        print(f"  Filtered by mask_ratio={args.mask_ratio}: {len(filtered)} cases")

    unique_ids = sorted(set(c["case_id"].rsplit("_", 2)[0] for c in filtered))
    print(f"  Unique sections: {len(unique_ids)}")

    rng = np.random.RandomState(args.seed)
    if len(unique_ids) > args.n_sections:
        selected_ids = set(rng.choice(len(unique_ids), args.n_sections, replace=False))
        selected_prefixes = [unique_ids[i] for i in selected_ids]
    else:
        selected_prefixes = unique_ids

    cases_to_plot = [c for c in filtered
                     if any(c["case_id"].startswith(p) for p in selected_prefixes)]
    cases_to_plot = sorted(cases_to_plot, key=lambda c: c["case_id"])

    seen_prefix = set()
    deduped = []
    for c in cases_to_plot:
        prefix = c["case_id"].rsplit("_", 2)[0]
        if prefix not in seen_prefix:
            seen_prefix.add(prefix)
            deduped.append(c)

    print(f"\nGenerating comparison for {len(deduped)} section(s) ...")
    os.makedirs(args.output_dir, exist_ok=True)

    for i, case in enumerate(deduped):
        print(f"  [{i+1}/{len(deduped)}] {case['case_id']}")
        plot_case(case, adapters, v_mean, v_std, args.output_dir, model_configs, device)

    print(f"\nAll comparisons saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
