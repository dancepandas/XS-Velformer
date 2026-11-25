"""
xs_velformer._morphology — bank/grad 形态特征 (内联, 自包含)

从训练仓库 data/preprocess.py 摘取 compute_bank_and_grad_features,
仅保留推理必需的两个 numpy 函数, 不引入 data/* 的其余依赖。

特征说明:
    bank_norm = 2·min(x_norm, 1-x_norm), 相对岸距 [0,1], 0=岸边, 1=中心
    grad_norm = 局部水深梯度 / max(梯度), [0,1]
"""

import numpy as np


def compute_bank_and_grad_features(x_norm: np.ndarray, d_norm: np.ndarray, eps: float = 1e-6):
    """基于完整断面的 x_norm / d_norm 计算局部形态特征。

    Args:
        x_norm: [K] 归一化横向位置 [0, 1]
        d_norm: [K] 归一化水深 [0, 1]
        eps:    防止除零的小常数

    Returns:
        bank_norm: [K] 相对岸距归一化 [0, 1]
        grad_norm: [K] 局部水深梯度归一化 [0, 1]
    """
    bank_norm = 2.0 * np.minimum(x_norm, 1.0 - x_norm)
    bank_norm = np.clip(bank_norm, 0.0, 1.0)

    K = len(x_norm)
    grad = np.zeros(K, dtype=np.float64)
    if K >= 2:
        grad[0] = abs(d_norm[1] - d_norm[0]) / (abs(x_norm[1] - x_norm[0]) + eps)
        grad[-1] = abs(d_norm[-1] - d_norm[-2]) / (abs(x_norm[-1] - x_norm[-2]) + eps)
    if K >= 3:
        dx = np.abs(x_norm[2:] - x_norm[:-2]) + eps
        dd = np.abs(d_norm[2:] - d_norm[:-2])
        grad[1:-1] = dd / dx

    grad_max = grad.max()
    if grad_max > 0:
        grad_norm = grad / (grad_max + eps)
    else:
        grad_norm = grad
    grad_norm = np.clip(grad_norm, 0.0, 1.0)

    return bank_norm.astype(np.float32), grad_norm.astype(np.float32)