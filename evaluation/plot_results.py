"""
评估结果可视化

从 result/all_eval.json 和 result/all_eval_per_case.pkl 读取数据,
生成 4 张对比图保存到 result/ 下。

用法:
    python -m evaluation.plot_results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAMES = {
    "query_transformer": "Hydro-QT",
    "lstm": "LSTM",
    "gan_z0": "GAN (z=0)",
    "gan_mean5": "GAN (mean5)",
    "diffusion_mean5": "Diffusion (mean5)",
}

COLORS = {
    "query_transformer": "#e74c3c",
    "lstm": "#3498db",
    "gan_z0": "#2ecc71",
    "gan_mean5": "#27ae60",
    "diffusion_mean5": "#9b59b6",
}

STRATEGY_ORDER = ["random", "continuous", "edge", "sparse_reconstruct", "densify"]
STRATEGY_LABELS = {
    "random": "Random",
    "continuous": "Continuous",
    "edge": "Edge",
    "sparse_reconstruct": "Sparse Recon.",
    "densify": "Densify",
}


def _load_data(result_dir="result"):
    with open(os.path.join(result_dir, "all_eval.json"), "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(os.path.join(result_dir, "all_eval_per_case.pkl"), "rb") as f:
        per_case = pickle.load(f)
    return eval_data, per_case


def _setup_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })


def plot_overall(eval_data, result_dir):
    models = list(eval_data["models"].keys())
    display_names = [MODEL_NAMES.get(m, m) for m in models]
    colors = [COLORS.get(m, "#888888") for m in models]

    metrics_to_plot = [
        ("rmse_raw", "Overall RMSE", "RMSE (m/s)"),
        ("mae_raw", "Overall MAE", "MAE (m/s)"),
        ("nse_raw", "Overall NSE", "NSE"),
        ("kge_raw", "Overall KGE", "KGE"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))

    for ax, (metric, title, ylabel) in zip(axes, metrics_to_plot):
        vals = [eval_data["models"][m]["overall"].get(metric, 0) for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors, width=0.6, edgecolor="white")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(display_names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(result_dir, "overall_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_by_strategy(eval_data, result_dir):
    models = list(eval_data["models"].keys())
    strategies = [s for s in STRATEGY_ORDER if s in eval_data["models"][models[0]]["by_strategy"]]

    metrics_to_plot = [
        ("rmse_raw", "RMSE by Strategy", "RMSE (m/s)"),
        ("nse_raw", "NSE by Strategy", "NSE"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(7 * len(metrics_to_plot), 5))

    for ax, (metric, title, ylabel) in zip(axes, metrics_to_plot):
        x = np.arange(len(strategies))
        n_models = len(models)
        width = 0.8 / n_models

        for i, m in enumerate(models):
            display = MODEL_NAMES.get(m, m)
            color = COLORS.get(m, "#888888")
            vals = [eval_data["models"][m]["by_strategy"][s][metric] for s in strategies]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=display, color=color, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(result_dir, "strategy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_by_mask_ratio(eval_data, result_dir):
    models = list(eval_data["models"].keys())
    by_ratio = eval_data["models"][models[0]]["by_mask_ratio"]
    ratios = sorted(by_ratio.keys(), key=lambda x: float(x))

    metrics_to_plot = [
        ("rmse_raw", "RMSE vs Mask Ratio", "RMSE (m/s)"),
        ("nse_raw", "NSE vs Mask Ratio", "NSE"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(7 * len(metrics_to_plot), 5))

    for ax, (metric, title, ylabel) in zip(axes, metrics_to_plot):
        for m in models:
            display = MODEL_NAMES.get(m, m)
            color = COLORS.get(m, "#888888")
            vals = [eval_data["models"][m]["by_mask_ratio"][r][metric] for r in ratios]
            ax.plot([float(r) for r in ratios], vals, "o-", label=display, color=color,
                    markersize=5, linewidth=1.5)

        ax.set_xlabel("Mask Ratio")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(result_dir, "mask_ratio_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_case_box(per_case, result_dir):
    models = list(per_case.keys())

    metrics_to_plot = [
        ("rmse_raw", "Per-Case RMSE Distribution", "RMSE (m/s)"),
        ("nse_raw", "Per-Case NSE Distribution", "NSE"),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(7 * len(metrics_to_plot), 5))

    for ax, (metric, title, ylabel) in zip(axes, metrics_to_plot):
        data = []
        labels = []
        colors_list = []
        for m in models:
            cases = per_case[m]
            vals = [c[metric] for c in cases if metric in c]
            data.append(vals)
            labels.append(MODEL_NAMES.get(m, m))
            colors_list.append(COLORS.get(m, "#888888"))

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(result_dir, "per_case_box.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    _setup_style()
    result_dir = "result"
    eval_data, per_case = _load_data(result_dir)
    plot_overall(eval_data, result_dir)
    plot_by_strategy(eval_data, result_dir)
    plot_by_mask_ratio(eval_data, result_dir)
    plot_per_case_box(per_case, result_dir)
    print("All plots generated.")


if __name__ == "__main__":
    main()
