"""
传统插值基线方法

本文件实现四种基于空间插值的流速预测方法, 不需要训练, 直接从 context 点
的坐标和流速对 query 点做空间插值预测。

所有方法在标准化流速空间 (v_norm) 中操作, 输入为 context/query 张量。

Context 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
Query   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]

传统方法从中提取:
    ctx_x = context[:, :, 0]   # x_norm
    ctx_d = context[:, :, 1]   # d_norm
    ctx_v = context[:, :, 4]   # v_norm
    qry_x = query[:, :, 0]     # x_norm
    qry_d = query[:, :, 1]     # d_norm
"""

import numpy as np
import torch
from typing import Optional

from scipy.interpolate import CubicSpline, RBFInterpolator


def nearest_x(
    ctx_x: torch.Tensor,
    ctx_v: torch.Tensor,
    qry_x: torch.Tensor,
    context_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """按横向距离 x_norm 最近邻取已知 context 流速

    v_q = v_c[argmin_i |x_q - x_c_i|]

    Args:
        ctx_x: [B, N_ctx] context 横向位置
        ctx_v: [B, N_ctx] context 流速
        qry_x: [B, N_qry] query 横向位置
        context_mask: [B, N_ctx] bool, True=有效

    Returns:
        [B, N_qry] 预测流速
    """
    # [B, N_qry, N_ctx]
    dist = torch.abs(qry_x.unsqueeze(2) - ctx_x.unsqueeze(1))

    if context_mask is not None:
        dist = dist.masked_fill(~context_mask.unsqueeze(1), float("inf"))

    nearest_idx = dist.argmin(dim=2)
    B, N_qry = qry_x.shape
    batch_idx = torch.arange(B, device=qry_x.device).unsqueeze(1).expand(B, N_qry)
    return ctx_v[batch_idx, nearest_idx]


def linear_x(
    ctx_x: torch.Tensor,
    ctx_v: torch.Tensor,
    qry_x: torch.Tensor,
    context_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """基于 x_norm 的一维线性插值

    对 query 点 x_q:
        找到 x_q 两侧最近的 context 点 (x_l, v_l) 和 (x_r, v_r)
        v_q = v_l + (v_r - v_l) * (x_q - x_l) / (x_r - x_l + eps)

    超出 context 范围的 query 点使用最近端点值。

    Args:
        ctx_x: [B, N_ctx] context 横向位置
        ctx_v: [B, N_ctx] context 流速
        qry_x: [B, N_qry] query 横向位置
        context_mask: [B, N_ctx] bool, True=有效

    Returns:
        [B, N_qry] 预测流速
    """
    B, N_ctx = ctx_x.shape
    _, N_qry = qry_x.shape

    if context_mask is None:
        context_mask = torch.ones(B, N_ctx, dtype=torch.bool, device=ctx_x.device)

    result = torch.zeros(B, N_qry, device=qry_x.device)

    for b in range(B):
        valid = context_mask[b]
        if valid.sum() < 1:
            continue

        cx = ctx_x[b, valid].cpu().numpy()
        cv = ctx_v[b, valid].cpu().numpy()
        qx = qry_x[b].cpu().numpy()

        sort_idx = np.argsort(cx)
        cx_sorted = cx[sort_idx]
        cv_sorted = cv[sort_idx]

        v_interp = np.interp(qx, cx_sorted, cv_sorted)
        result[b] = torch.from_numpy(v_interp.astype(np.float32)).to(qry_x.device)

    return result


def idw_x(
    ctx_x: torch.Tensor,
    ctx_v: torch.Tensor,
    qry_x: torch.Tensor,
    context_mask: Optional[torch.Tensor] = None,
    p: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """只按横向距离做反距离加权

    w_i = 1 / (|x_q - x_c_i| + eps)^p
    v_q = sum(w_i * v_c_i) / sum(w_i)

    Args:
        ctx_x: [B, N_ctx] context 横向位置
        ctx_v: [B, N_ctx] context 流速
        qry_x: [B, N_qry] query 横向位置
        context_mask: [B, N_ctx] bool, True=有效
        p:   距离衰减指数
        eps: 防止除零

    Returns:
        [B, N_qry] 预测流速
    """
    # [B, N_qry, N_ctx]
    dist = torch.abs(qry_x.unsqueeze(2) - ctx_x.unsqueeze(1)) + eps
    weights = 1.0 / dist.pow(p)

    if context_mask is not None:
        weights = weights * context_mask.unsqueeze(1).float()

    weight_sum = weights.sum(dim=2, keepdim=True).clamp(min=eps)
    weights = weights / weight_sum

    # [B, N_qry]
    return (weights * ctx_v.unsqueeze(1)).sum(dim=2)


def idw_xd(
    ctx_x: torch.Tensor,
    ctx_d: torch.Tensor,
    ctx_v: torch.Tensor,
    qry_x: torch.Tensor,
    qry_d: torch.Tensor,
    context_mask: Optional[torch.Tensor] = None,
    p: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """在二维特征空间 [x_norm, d_norm] 上做 IDW

    dist_i = sqrt((x_q - x_c_i)^2 + (d_q - d_c_i)^2)
    w_i = 1 / (dist_i + eps)^p
    v_q = sum(w_i * v_c_i) / sum(w_i)

    Args:
        ctx_x: [B, N_ctx] context 横向位置
        ctx_d: [B, N_ctx] context 水深
        ctx_v: [B, N_ctx] context 流速
        qry_x: [B, N_qry] query 横向位置
        qry_d: [B, N_qry] query 水深
        context_mask: [B, N_ctx] bool, True=有效
        p:   距离衰减指数
        eps: 防止除零

    Returns:
        [B, N_qry] 预测流速
    """
    # [B, N_qry, N_ctx]
    dx = (qry_x.unsqueeze(2) - ctx_x.unsqueeze(1)).pow(2)
    dd = (qry_d.unsqueeze(2) - ctx_d.unsqueeze(1)).pow(2)
    dist = torch.sqrt(dx + dd) + eps

    weights = 1.0 / dist.pow(p)

    if context_mask is not None:
        weights = weights * context_mask.unsqueeze(1).float()

    weight_sum = weights.sum(dim=2, keepdim=True).clamp(min=eps)
    weights = weights / weight_sum

    return (weights * ctx_v.unsqueeze(1)).sum(dim=2)


def cubic_spline(
    ctx_x: torch.Tensor,
    ctx_v: torch.Tensor,
    qry_x: torch.Tensor,
    context_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """基于 x_norm 的三次样条插值 (R1-2 要求的 additional interpolation)

    对每条断面拟合 scipy CubicSpline (自然边界), 范围外的 query 点
    取最近端点值 (与 linear_x 的 np.interp 行为一致)。

    退化情形: 有效点 <4 条退化为线性插值, <2 条输出 0。

    Args:
        ctx_x: [B, N_ctx] context 横向位置
        ctx_v: [B, N_ctx] context 流速
        qry_x: [B, N_qry] query 横向位置
        context_mask: [B, N_ctx] bool, True=有效

    Returns:
        [B, N_qry] 预测流速
    """
    B, N_ctx = ctx_x.shape
    _, N_qry = qry_x.shape

    if context_mask is None:
        context_mask = torch.ones(B, N_ctx, dtype=torch.bool, device=ctx_x.device)

    result = torch.zeros(B, N_qry, device=qry_x.device)

    for b in range(B):
        valid = context_mask[b]
        if valid.sum() < 2:
            continue

        cx = ctx_x[b, valid].cpu().numpy()
        cv = ctx_v[b, valid].cpu().numpy()
        qx = qry_x[b].cpu().numpy()

        sort_idx = np.argsort(cx)
        cx_sorted = cx[sort_idx]
        cv_sorted = cv[sort_idx]

        # CubicSpline 要求 x 严格递增: 对重复 x 取均值
        cx_unique, inverse = np.unique(cx_sorted, return_inverse=True)
        if len(cx_unique) < len(cx_sorted):
            cv_unique = np.zeros(len(cx_unique))
            counts = np.zeros(len(cx_unique))
            np.add.at(cv_unique, inverse, cv_sorted)
            np.add.at(counts, inverse, 1.0)
            cv_unique = cv_unique / np.maximum(counts, 1.0)
        else:
            cv_unique = cv_sorted

        if len(cx_unique) < 2:
            continue
        if len(cx_unique) < 4:
            # 样条需要至少 4 个节点, 退化为线性
            v_interp = np.interp(qx, cx_unique, cv_unique)
        else:
            spline = CubicSpline(cx_unique, cv_unique, extrapolate=False)
            inside = (qx >= cx_unique[0]) & (qx <= cx_unique[-1])
            v_interp = np.empty_like(qx)
            v_interp[inside] = spline(qx[inside])
            # 范围外取端点值 (与 np.interp 一致)
            v_interp[qx < cx_unique[0]] = cv_unique[0]
            v_interp[qx > cx_unique[-1]] = cv_unique[-1]

        result[b] = torch.from_numpy(v_interp.astype(np.float32)).to(qry_x.device)

    return result


def rbf_xd(
    ctx_x: torch.Tensor,
    ctx_d: torch.Tensor,
    ctx_v: torch.Tensor,
    qry_x: torch.Tensor,
    qry_d: torch.Tensor,
    context_mask: Optional[torch.Tensor] = None,
    smoothing: float = 1e-6,
    eps: float = 1e-6,
) -> torch.Tensor:
    """在二维特征空间 [x_norm, d_norm] 上的薄板样条 RBF 插值 (R1-2)

    thin-plate spline RBF: phi(r) = r^2 * log(r), 对 2-D 散点插值平滑。

    退化情形: 有效点不足或矩阵奇异时, 该断面退化为 idw_xd。

    Args:
        ctx_x: [B, N_ctx] context 横向位置
        ctx_d: [B, N_ctx] context 水深
        ctx_v: [B, N_ctx] context 流速
        qry_x: [B, N_qry] query 横向位置
        qry_d: [B, N_qry] query 水深
        context_mask: [B, N_ctx] bool, True=有效
        smoothing: RBF 平滑系数 (0 为严格插值, 微小值防奇异)
        eps: 退化回退的 IDW 防除零

    Returns:
        [B, N_qry] 预测流速
    """
    from numpy.linalg import LinAlgError

    B, N_ctx = ctx_x.shape
    _, N_qry = qry_x.shape

    if context_mask is None:
        context_mask = torch.ones(B, N_ctx, dtype=torch.bool, device=ctx_x.device)

    result = torch.zeros(B, N_qry, device=qry_x.device)

    for b in range(B):
        valid = context_mask[b]
        n_valid = int(valid.sum())
        cx = ctx_x[b, valid].cpu().numpy()
        cd = ctx_d[b, valid].cpu().numpy()
        cv = ctx_v[b, valid].cpu().numpy()
        qx = qry_x[b].cpu().numpy()
        qd = qry_d[b].cpu().numpy()

        # RBF 至少需要与维度相称的节点数; 不足时退化为 IDW
        if n_valid < 6:
            ctx_x_b = ctx_x[b:b + 1]
            ctx_d_b = ctx_d[b:b + 1]
            ctx_v_b = ctx_v[b:b + 1]
            qry_x_b = qry_x[b:b + 1]
            qry_d_b = qry_d[b:b + 1]
            mask_b = valid.unsqueeze(0)
            result[b] = idw_xd(ctx_x_b, ctx_d_b, ctx_v_b, qry_x_b, qry_d_b,
                               mask_b, p=2.0, eps=eps)[0]
            continue

        ctx_points = np.stack([cx, cd], axis=1)
        qry_points = np.stack([qx, qd], axis=1)

        try:
            interpolator = RBFInterpolator(
                ctx_points, cv, smoothing=smoothing, kernel="thin_plate_spline",
            )
            v_interp = interpolator(qry_points)
        except LinAlgError:
            ctx_x_b = ctx_x[b:b + 1]
            ctx_d_b = ctx_d[b:b + 1]
            ctx_v_b = ctx_v[b:b + 1]
            qry_x_b = qry_x[b:b + 1]
            qry_d_b = qry_d[b:b + 1]
            mask_b = valid.unsqueeze(0)
            result[b] = idw_xd(ctx_x_b, ctx_d_b, ctx_v_b, qry_x_b, qry_d_b,
                               mask_b, p=2.0, eps=eps)[0]
            continue

        result[b] = torch.from_numpy(
            np.nan_to_num(v_interp, nan=0.0).astype(np.float32),
        ).to(qry_x.device)

    return result
