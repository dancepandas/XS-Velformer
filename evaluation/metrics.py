"""
统一评估指标模块

实现 masked MSE/RMSE/MAE/R2/NSE/KGE/Bias,
原始速度反归一化指标,
以及按 strategy / mask_ratio / station 分组聚合。
"""

import numpy as np
from typing import Dict, List, Optional


METRIC_KEYS = [
    "mse", "rmse", "mae", "r2", "nse", "kge", "bias",
    "rmse_raw", "mae_raw", "r2_raw", "nse_raw", "kge_raw", "bias_raw",
]


def compute_masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    diff = (pred - target) ** 2
    valid = mask.astype(float)
    return float((diff * valid).sum() / (valid.sum() + 1e-8))


def compute_masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(compute_masked_mse(pred, target, mask)))


def compute_masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(pred - target)
    valid = mask.astype(float)
    return float((diff * valid).sum() / (valid.sum() + 1e-8))


def compute_r2(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(float)
    n = valid.sum()
    if n < 2:
        return 0.0
    mean_t = float((target * valid).sum() / n)
    ss_res = float(((pred - target) ** 2 * valid).sum())
    ss_tot = float(((target - mean_t) ** 2 * valid).sum())
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_nse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return compute_r2(pred, target, mask)


def compute_kge(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(float)
    n = valid.sum()
    if n < 2:
        return 0.0
    mean_p = float((pred * valid).sum() / n)
    mean_t = float((target * valid).sum() / n)
    var_p = float(((pred - mean_p) ** 2 * valid).sum() / n)
    var_t = float(((target - mean_t) ** 2 * valid).sum() / n)
    cov = float(((pred - mean_p) * (target - mean_t) * valid).sum() / n)
    if var_t < 1e-12 or var_p < 1e-12:
        return 0.0
    r = cov / (np.sqrt(var_p) * np.sqrt(var_t))
    alpha = np.sqrt(var_p) / np.sqrt(var_t)
    beta = mean_p / mean_t if abs(mean_t) > 1e-12 else 0.0
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def compute_bias(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(float)
    n = valid.sum()
    if n < 1:
        return 0.0
    return float(((pred - target) * valid).sum() / n)


def denormalize_velocity(v_norm: np.ndarray, v_mean: float, v_std: float) -> np.ndarray:
    return v_norm * v_std + v_mean


def compute_raw_rmse(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mask: np.ndarray,
    v_mean: float,
    v_std: float,
) -> float:
    pred_raw = denormalize_velocity(pred_norm, v_mean, v_std)
    target_raw = denormalize_velocity(target_norm, v_mean, v_std)
    return compute_masked_rmse(pred_raw, target_raw, mask)


def compute_raw_mae(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mask: np.ndarray,
    v_mean: float,
    v_std: float,
) -> float:
    pred_raw = denormalize_velocity(pred_norm, v_mean, v_std)
    target_raw = denormalize_velocity(target_norm, v_mean, v_std)
    return compute_masked_mae(pred_raw, target_raw, mask)


def compute_raw_r2(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mask: np.ndarray,
    v_mean: float,
    v_std: float,
) -> float:
    pred_raw = denormalize_velocity(pred_norm, v_mean, v_std)
    target_raw = denormalize_velocity(target_norm, v_mean, v_std)
    return compute_r2(pred_raw, target_raw, mask)


def compute_raw_nse(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mask: np.ndarray,
    v_mean: float,
    v_std: float,
) -> float:
    return compute_raw_r2(pred_norm, target_norm, mask, v_mean, v_std)


def compute_raw_kge(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mask: np.ndarray,
    v_mean: float,
    v_std: float,
) -> float:
    pred_raw = denormalize_velocity(pred_norm, v_mean, v_std)
    target_raw = denormalize_velocity(target_norm, v_mean, v_std)
    return compute_kge(pred_raw, target_raw, mask)


def compute_raw_bias(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mask: np.ndarray,
    v_mean: float,
    v_std: float,
) -> float:
    pred_raw = denormalize_velocity(pred_norm, v_mean, v_std)
    target_raw = denormalize_velocity(target_norm, v_mean, v_std)
    return compute_bias(pred_raw, target_raw, mask)


def _zero_metrics() -> Dict[str, float]:
    return {k: 0.0 for k in METRIC_KEYS} | {"n_points": 0}


def compute_all_metrics(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    mask: np.ndarray,
    v_mean: float = 0.0,
    v_std: float = 1.0,
) -> Dict[str, float]:
    metrics = {
        "mse": compute_masked_mse(pred_norm, target_norm, mask),
        "rmse": compute_masked_rmse(pred_norm, target_norm, mask),
        "mae": compute_masked_mae(pred_norm, target_norm, mask),
        "r2": compute_r2(pred_norm, target_norm, mask),
        "nse": compute_nse(pred_norm, target_norm, mask),
        "kge": compute_kge(pred_norm, target_norm, mask),
        "bias": compute_bias(pred_norm, target_norm, mask),
    }
    if v_std != 1.0 or v_mean != 0.0:
        metrics["rmse_raw"] = compute_raw_rmse(pred_norm, target_norm, mask, v_mean, v_std)
        metrics["mae_raw"] = compute_raw_mae(pred_norm, target_norm, mask, v_mean, v_std)
        metrics["r2_raw"] = compute_raw_r2(pred_norm, target_norm, mask, v_mean, v_std)
        metrics["nse_raw"] = compute_raw_nse(pred_norm, target_norm, mask, v_mean, v_std)
        metrics["kge_raw"] = compute_raw_kge(pred_norm, target_norm, mask, v_mean, v_std)
        metrics["bias_raw"] = compute_raw_bias(pred_norm, target_norm, mask, v_mean, v_std)
    else:
        for k in ("rmse_raw", "mae_raw", "r2_raw", "nse_raw", "kge_raw", "bias_raw"):
            metrics[k] = 0.0
    return metrics


class MetricsAccumulator:
    def __init__(self, v_mean: float = 0.0, v_std: float = 1.0):
        self.v_mean = v_mean
        self.v_std = v_std
        self._sum_sq = 0.0
        self._sum_abs = 0.0
        self._sum_sq_raw = 0.0
        self._sum_abs_raw = 0.0
        self._sum_pred = 0.0
        self._sum_target = 0.0
        self._sum_pred_raw = 0.0
        self._sum_target_raw = 0.0
        self._sum_pred_sq = 0.0
        self._sum_target_sq = 0.0
        self._sum_pred_sq_raw = 0.0
        self._sum_target_sq_raw = 0.0
        self._sum_pred_target = 0.0
        self._sum_pred_target_raw = 0.0
        self._sum_diff = 0.0
        self._sum_diff_raw = 0.0
        self._n_points = 0

    def update(
        self,
        pred_norm: np.ndarray,
        target_norm: np.ndarray,
        mask: np.ndarray,
    ):
        valid = mask.astype(float)
        n = float(valid.sum())
        if n < 1:
            return

        diff_norm = pred_norm - target_norm
        self._sum_sq += float((diff_norm ** 2 * valid).sum())
        self._sum_abs += float((np.abs(diff_norm) * valid).sum())
        self._sum_diff += float((diff_norm * valid).sum())

        self._sum_pred += float((pred_norm * valid).sum())
        self._sum_target += float((target_norm * valid).sum())
        self._sum_pred_sq += float((pred_norm ** 2 * valid).sum())
        self._sum_target_sq += float((target_norm ** 2 * valid).sum())
        self._sum_pred_target += float((pred_norm * target_norm * valid).sum())

        pred_raw = denormalize_velocity(pred_norm, self.v_mean, self.v_std)
        target_raw = denormalize_velocity(target_norm, self.v_mean, self.v_std)
        diff_raw = pred_raw - target_raw
        self._sum_sq_raw += float((diff_raw ** 2 * valid).sum())
        self._sum_abs_raw += float((np.abs(diff_raw) * valid).sum())
        self._sum_diff_raw += float((diff_raw * valid).sum())

        self._sum_pred_raw += float((pred_raw * valid).sum())
        self._sum_target_raw += float((target_raw * valid).sum())
        self._sum_pred_sq_raw += float((pred_raw ** 2 * valid).sum())
        self._sum_target_sq_raw += float((target_raw ** 2 * valid).sum())
        self._sum_pred_target_raw += float((pred_raw * target_raw * valid).sum())

        self._n_points += n

    def compute(self) -> Dict[str, float]:
        n = self._n_points
        if n < 1:
            result = _zero_metrics()
            result["n_points"] = 0
            return result

        mse = self._sum_sq / n
        mae = self._sum_abs / n
        mse_raw = self._sum_sq_raw / n
        mae_raw = self._sum_abs_raw / n
        bias = self._sum_diff / n
        bias_raw = self._sum_diff_raw / n

        mean_t = self._sum_target / n
        ss_tot = self._sum_target_sq - n * mean_t ** 2
        r2 = 1.0 - self._sum_sq / ss_tot if abs(ss_tot) > 1e-12 else 0.0

        mean_t_raw = self._sum_target_raw / n
        ss_tot_raw = self._sum_target_sq_raw - n * mean_t_raw ** 2
        r2_raw = 1.0 - self._sum_sq_raw / ss_tot_raw if abs(ss_tot_raw) > 1e-12 else 0.0

        mean_p = self._sum_pred / n
        mean_p_raw = self._sum_pred_raw / n

        var_p = self._sum_pred_sq / n - mean_p ** 2
        var_t = ss_tot / n
        cov = self._sum_pred_target / n - mean_p * mean_t
        kge = self._compute_kge_from_moments(r2 if var_t > 1e-12 else 0.0,
                                              var_p, var_t, mean_p, mean_t)

        var_p_raw = self._sum_pred_sq_raw / n - mean_p_raw ** 2
        var_t_raw = ss_tot_raw / n
        cov_raw = self._sum_pred_target_raw / n - mean_p_raw * mean_t_raw
        r_raw_val = cov_raw / (np.sqrt(max(var_p_raw, 0)) * np.sqrt(max(var_t_raw, 0))) \
            if var_p_raw > 1e-12 and var_t_raw > 1e-12 else 0.0
        kge_raw = self._compute_kge_from_moments(r_raw_val,
                                                  var_p_raw, var_t_raw,
                                                  mean_p_raw, mean_t_raw)

        return {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mae),
            "r2": float(r2),
            "nse": float(r2),
            "kge": float(kge),
            "bias": float(bias),
            "rmse_raw": float(np.sqrt(mse_raw)),
            "mae_raw": float(mae_raw),
            "r2_raw": float(r2_raw),
            "nse_raw": float(r2_raw),
            "kge_raw": float(kge_raw),
            "bias_raw": float(bias_raw),
            "n_points": int(n),
        }

    @staticmethod
    def _compute_kge_from_moments(r_val, var_p, var_t, mean_p, mean_t):
        if var_t < 1e-12 or var_p < 1e-12:
            return 0.0
        alpha = np.sqrt(var_p) / np.sqrt(var_t)
        beta = mean_p / mean_t if abs(mean_t) > 1e-12 else 0.0
        return float(1.0 - np.sqrt((r_val - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def aggregate_by_group(
    per_case_results: List[Dict],
    group_key: str,
) -> Dict[str, Dict[str, float]]:
    groups = {}
    for r in per_case_results:
        key = r.get(group_key, "unknown")
        if key not in groups:
            groups[key] = MetricsAccumulator(v_mean=r.get("_v_mean", 0.0), v_std=r.get("_v_std", 1.0))
        groups[key].update(
            np.array(r["pred_norm"]),
            np.array(r["target_norm"]),
            np.array(r["mask"]),
        )

    result = {}
    for key, acc in groups.items():
        result[key] = acc.compute()
    return result
