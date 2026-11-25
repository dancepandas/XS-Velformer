"""
xs_velformer.features — 原始物理量 → 模型特征张量

复刻训练期 (data/preprocess.py + data/dataset.py) 的特征工程:

    Context 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
    Query   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]

特征一致性说明:
    - x_min / W / h_max 从整条断面全部点派生 (W = x.max()-x.min()), 不使用外部 width
    - x_norm = (x - x_min)/W, d_norm = h/h_max  (逐断面 [0,1]) —— 与训练逐位一致
    - bank_norm = 2·min(x_norm, 1-x_norm) 是纯逐点量, 与点集无关 —— 与训练一致
    - grad_norm 在 context∪query 上用 compute_bank_and_grad_features 计算:
        · impute_missing: context∪query == 整条断面 → 与训练逐位一致
        · reconstruct/densify: query 是合成网格点, context∪query ≠ 训练断面,
          grad 的邻域与 grad_max 分母与训练略有差异 (见下方 TODO, 待后续更新对齐)
      该行为与旧 predict.py 完全一致 (数值对齐 <1e-6), 保证已发表结果可复现。
    - v_norm / Z_norm / h_max_norm / W_norm 用 Normalizer 全局标准化
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ._morphology import compute_bank_and_grad_features  # 复用训练期同名实现 (内联)

from .normalizer import Normalizer
from .schemas import QuerySpec, SectionData

CONTEXT_DIM = 8
QUERY_DIM = 7


@dataclass
class SectionFeatures:
    """单条断面的特征化结果 (numpy, 未 padding)"""
    context: np.ndarray         # [N_ctx, 8]
    query: np.ndarray           # [N_qry, 7]
    n_context: int
    n_query: int
    # 派生量 (供诊断/对齐)
    x_min: float
    width: float
    h_max: float
    # query 实际使用的水深 (物理 m): 显式给定或由 context 插值; 供结果组装复用, 避免二次插值
    query_depths: np.ndarray = None  # type: ignore[assignment]
    # query 的物理起点距 (m), 与 query_depths 一一对应; 供结果组装复用
    query_positions: np.ndarray = None  # type: ignore[assignment]


def _interp_depth_on_query(
    ctx_x_phys: np.ndarray,
    ctx_h_phys: np.ndarray,
    qry_x_phys: np.ndarray,
) -> np.ndarray:
    """query 水深由 context 水深在物理坐标上线性插值"""
    order = np.argsort(ctx_x_phys)
    xs = ctx_x_phys[order]
    hs = ctx_h_phys[order]
    return np.interp(qry_x_phys, xs, hs)


def build_section_features(
    section: SectionData,
    query: QuerySpec,
    normalizer: Normalizer,
) -> SectionFeatures:
    """把一条断面 + query 规格转为模型输入特征。

    Args:
        section:    原始断面 (物理单位)
        query:      目标预测位置 (物理单位; depths 可为 None 由 context 插值)
        normalizer: 全局归一化器
    """
    # ---- 1. 取出 context (已知流速点) ----
    known_idx = section.known_indices()
    ctx_x_phys = np.asarray([section.distances[i] for i in known_idx], dtype=np.float64)
    ctx_h_phys = np.asarray([section.depths[i] for i in known_idx], dtype=np.float64)
    ctx_v_phys = np.asarray([section.velocities[i] for i in known_idx], dtype=np.float64)

    # ---- 2. query 位置 / 水深 ----
    qry_x_phys = np.asarray(query.distances, dtype=np.float64)
    if query.depths is not None:
        qry_h_phys = np.asarray(query.depths, dtype=np.float64)
    else:
        qry_h_phys = _interp_depth_on_query(ctx_x_phys, ctx_h_phys, qry_x_phys)

    # ---- 3. 派生断面几何 (与训练一致: 从整条断面全部点派生, 而非 context∪query) ----
    # 整条断面的几何 (x_min / W / h_max) 是断面固有属性, 不随 query 场景变化;
    # 训练期同样用全部 K 个真实点计算。这样 impute / reconstruct / densify 三场景
    # 共用同一套几何标定, 与训练分布一致。
    sec_x = np.asarray(section.distances, dtype=np.float64)
    sec_h = np.asarray(section.depths, dtype=np.float64)
    x_min = float(np.min(sec_x))
    width = float(np.max(sec_x) - np.min(sec_x))
    h_max_raw = float(np.max(sec_h))
    h_max = h_max_raw if h_max_raw > 0 else 1.0   # 与 preprocess 一致的兜底

    # ---- 4. 逐断面归一化 x_norm / d_norm (context + query 点, 用整条断面标定) ----
    all_x = np.concatenate([ctx_x_phys, qry_x_phys])
    all_h = np.concatenate([ctx_h_phys, qry_h_phys])
    x_norm_all = (all_x - x_min) / width if width > 0 else np.zeros_like(all_x)
    d_norm_all = all_h / h_max

    # ---- 5. bank / grad: 在 context∪query 上, 按 x 排序后计算, 再散回原顺序 ----
    # TODO(后续更新·选项B): 当前 grad_norm 在 context∪query 上计算。对 reconstruct/densify
    # (query 为合成网格点) 与训练期 (grad 在整条断面真实点上预算) 存在二阶分布差异。
    # 训练一致的做法: 在整条断面真实点 (sec_x/sec_h) 上算 grad 场, 再插值到 query 的 x。
    # 当前实现与旧 predict.py 逐位一致 (对齐 <1e-6), 保持以复现已发表结果; 改造见评审记录。
    sort_idx = np.argsort(x_norm_all, kind="stable")
    unsort_idx = np.argsort(sort_idx, kind="stable")
    bank_sorted, grad_sorted = compute_bank_and_grad_features(
        x_norm_all[sort_idx], d_norm_all[sort_idx]
    )
    bank_all = bank_sorted[unsort_idx]
    grad_all = grad_sorted[unsort_idx]

    n_ctx = len(ctx_x_phys)
    n_qry = len(qry_x_phys)
    ctx_bank, qry_bank = bank_all[:n_ctx], bank_all[n_ctx:]
    ctx_grad, qry_grad = grad_all[:n_ctx], grad_all[n_ctx:]

    # ---- 6. 全局标准化量 ----
    ctx_x_norm = x_norm_all[:n_ctx]
    ctx_d_norm = d_norm_all[:n_ctx]
    qry_x_norm = x_norm_all[n_ctx:]
    qry_d_norm = d_norm_all[n_ctx:]

    ctx_v_norm = normalizer.normalize_velocity(ctx_v_phys)
    globals_norm = normalizer.normalize_section_globals(
        Z=section.water_stage, h_max=h_max, W=width,
    )
    global_feat = np.array([
        globals_norm["Z_norm"],
        globals_norm["h_max_norm"],
        globals_norm["W_norm"],
    ], dtype=np.float64)

    # ---- 7. 组装 context[8] / query[7] ----
    context = np.column_stack([
        ctx_x_norm, ctx_d_norm, ctx_bank, ctx_grad, ctx_v_norm,
        np.tile(global_feat, (n_ctx, 1)),
    ]).astype(np.float32)

    query_feat = np.column_stack([
        qry_x_norm, qry_d_norm, qry_bank, qry_grad,
        np.tile(global_feat, (n_qry, 1)),
    ]).astype(np.float32)

    return SectionFeatures(
        context=context, query=query_feat,
        n_context=n_ctx, n_query=n_qry,
        x_min=x_min, width=width, h_max=h_max,
        query_depths=qry_h_phys,
        query_positions=qry_x_phys,
    )


def pad_batch(features_list):
    """把多条变长 SectionFeatures padding 成一个 batch (与 data.dataset.collate_fn 一致)。

    返回: {context, context_mask, query, query_mask} (torch.float32 / bool)
    维度从数据推导 (不硬编码), 与 data.dataset.collate_fn 保持同样的自适应语义。
    """
    import torch

    B = len(features_list)
    max_ctx = max(f.n_context for f in features_list)
    max_qry = max(f.n_query for f in features_list)
    ctx_dim = features_list[0].context.shape[1]
    qry_dim = features_list[0].query.shape[1]

    context = torch.zeros(B, max_ctx, ctx_dim, dtype=torch.float32)
    context_mask = torch.zeros(B, max_ctx, dtype=torch.bool)
    query = torch.zeros(B, max_qry, qry_dim, dtype=torch.float32)
    query_mask = torch.zeros(B, max_qry, dtype=torch.bool)

    for i, f in enumerate(features_list):
        context[i, :f.n_context] = torch.from_numpy(f.context)
        context_mask[i, :f.n_context] = True
        query[i, :f.n_query] = torch.from_numpy(f.query)
        query_mask[i, :f.n_query] = True

    return {
        "context": context,
        "context_mask": context_mask,
        "query": query,
        "query_mask": query_mask,
    }
