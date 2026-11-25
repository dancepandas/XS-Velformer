"""
数据集模块 —— 基于变长断面的自监督样本动态拆分

本文件实现训练用的 SectionDataset 和变长序列的 collate_fn。

核心逻辑:
    完整断面 S = {p_1, ..., p_K}, K 为该断面的测速线数量 (不同断面可不同),
    每次训练时基于列索引动态 mask, x/h/v 通过列索引天然对齐,
    同步拆分为 context/query/target。

    v_norm, bank_norm, grad_norm 已在 preprocess.py 中计算并保存,
    此处只需按索引切分, 无需重新计算。

Context 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
Query   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]

5 种采样策略 (均基于当前断面的动态 K 生成):
    - random:             随机缺失, 测试模型对散点缺失的插补能力
    - continuous:         连续块缺失, 测试模型对连续缺测区间的恢复能力
    - edge:               边缘缺失, 测试模型对左岸/右岸边缘区域的预测能力
    - sparse_reconstruct: 稀疏重构, 只给少量实测点重构完整断面
    - densify:            加密模拟, 等间距抽取稀疏 context 预测更多位置
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from data.preprocess import SectionSample


class SectionDataset(Dataset):
    """
    基于变长断面的自监督训练数据集

    通过 repeats 参数控制每条断面每 epoch 被采样的次数。
    例如 2929 条断面 x 20 repeats = 58580 条训练样本/epoch,
    每次采样随机选择不同的 mask 策略和 mask 比例,
    同一条断面可产生大量不同的 context/query 组合。

    每个 __getitem__ 调用:
    1. idx 映射到 sample_idx = idx % N, 随机选择 mask 策略
    2. 基于当前断面的动态 K 生成 query_idx / context_idx
    3. 按索引切分构建 context/query/target 张量
    """

    def __init__(
        self,
        samples: List[SectionSample],
        mask_ratio_range: Tuple[float, float] = (0.1, 0.4),
        strategies: Optional[List[str]] = None,
        repeats: int = 20,
    ):
        """
        Args:
            samples:          断面样本列表 (已经归一化)
            mask_ratio_range: mask 比例范围 (均匀采样)
            strategies:       可选的策略子集, None 表示使用全部 5 种
            repeats:          每条断面每 epoch 被采样的次数
        """
        self.samples = samples
        self.n_sections = len(samples)
        self.mask_ratio_range = mask_ratio_range
        self.strategies = strategies or [
            "random", "continuous", "edge",
            "sparse_reconstruct", "densify",
        ]
        self.repeats = repeats

    def __len__(self) -> int:
        return self.n_sections * self.repeats

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # idx 取模映射到实际断面, 再随机选策略生成 context/query
        sample = self.samples[idx % self.n_sections]
        strategy = np.random.choice(self.strategies)
        K = sample.K
        if K < 2:
            raise ValueError(
                f"断面测速线数 K={K} < 2, 无法构造非空 context/query (station={sample.station_id})")
        context_idx, query_idx = self._split_by_strategy(strategy, K)
        context_data, query_data, target = self._build_features(sample, context_idx, query_idx)
        return {
            "context": torch.tensor(context_data, dtype=torch.float32),
            "query": torch.tensor(query_data, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }

    def _split_by_strategy(self, strategy: str, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """根据策略生成列索引, 返回 (context_idx, query_idx)"""
        if strategy == "random":
            context_idx, query_idx = self._random_mask(K)
        elif strategy == "continuous":
            context_idx, query_idx = self._continuous_mask(K)
        elif strategy == "edge":
            context_idx, query_idx = self._edge_mask(K)
        elif strategy == "sparse_reconstruct":
            context_idx, query_idx = self._sparse_reconstruct(K)
        elif strategy == "densify":
            context_idx, query_idx = self._densify(K)
        else:
            context_idx, query_idx = self._random_mask(K)
        # 小 K 退化保护: sparse/densify 在 K=3 时可能让 context 占满 (空 query)
        if len(query_idx) == 0 or len(context_idx) == 0:
            context_idx, query_idx = self._random_mask(K)
        return context_idx, query_idx

    def _random_mask(self, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """随机缺失: 随机抽取 mask_ratio_range 范围内的比例作为 query"""
        ratio = np.random.uniform(*self.mask_ratio_range)
        n_query = max(1, int(K * ratio))
        perm = np.random.permutation(K)
        query_idx = np.sort(perm[:n_query])
        context_idx = np.sort(perm[n_query:])
        return context_idx, query_idx

    def _continuous_mask(self, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """连续块缺失: 随机选择连续区间作为 query"""
        ratio = np.random.uniform(*self.mask_ratio_range)
        block_len = max(1, int(K * ratio))
        start = np.random.randint(0, K - block_len + 1)
        query_idx = np.arange(start, start + block_len)
        context_idx = np.array([j for j in range(K) if j not in set(query_idx)])
        # 安全检查: 保证至少有一个 context 点
        if len(context_idx) == 0:
            context_idx = np.array([query_idx[0]])
            query_idx = query_idx[1:]
        return context_idx, query_idx

    def _edge_mask(self, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """边缘缺失: 选择左岸或右岸若干测点作为 query"""
        ratio = np.random.uniform(*self.mask_ratio_range)
        n_query = max(1, int(K * ratio))
        side = np.random.choice(["left", "right"])
        if side == "left":
            query_idx = np.arange(n_query)
        else:
            query_idx = np.arange(K - n_query, K)
        context_idx = np.array([j for j in range(K) if j not in set(query_idx)])
        if len(context_idx) == 0:
            context_idx = np.array([0])
            query_idx = np.array([1])
        return context_idx, query_idx

    def _sparse_reconstruct(self, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """稀疏重建: 保留少量均匀分布测点作为 context"""
        n_context = max(3, int(K * np.random.uniform(0.1, 0.3)))
        n_context = min(n_context, K - 1)  # 小 K: 至少给 query 留 1 条
        context_idx = np.sort(np.random.choice(K, n_context, replace=False))
        query_idx = np.array([j for j in range(K) if j not in set(context_idx)])
        return context_idx, query_idx

    def _densify(self, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """加密模拟: 等间距抽取稀疏 context"""
        n_context = max(3, int(K * np.random.uniform(0.2, 0.4)))
        step = max(1, K // n_context)
        context_idx = np.arange(0, K, step)
        if len(context_idx) < 2:
            context_idx = np.array([0, K // 2])
        context_idx = context_idx[:max(1, K - 1)]  # 小 K: 至少给 query 留 1 条
        query_idx = np.array([j for j in range(K) if j not in set(context_idx)])
        return context_idx, query_idx

    def _build_features(
        self,
        sample: SectionSample,
        context_idx: np.ndarray,
        query_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        基于 context/query 列索引构建特征

        Context: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]  8维
        Query:   [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]            7维
        Target:  v_norm[query_idx]                                                              1维

        bank_norm 和 grad_norm 在 preprocess.py 中基于完整断面预先计算,
        此处直接按索引切分。
        """
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

        return context, query, target


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    变长序列的 padding 和 mask

    由于不同断面的 K 不同, 同一 batch 内 context/query 长度各异,
    需要按 batch 内最大长度 padding, 并生成 mask 标记有效位置。

    输出:
        context:       [batch_size, max_ctx_len, 8]  零填充
        context_mask:  [batch_size, max_ctx_len]     bool, True=有效
        query:         [batch_size, max_qry_len, 7]  零填充
        query_mask:    [batch_size, max_qry_len]     bool, True=有效
        target:        [batch_size, max_qry_len]     零填充
    """
    max_ctx = max(item["context"].shape[0] for item in batch)
    max_qry = max(item["query"].shape[0] for item in batch)
    ctx_dim = batch[0]["context"].shape[1]
    qry_dim = batch[0]["query"].shape[1]

    context_padded = torch.zeros(len(batch), max_ctx, ctx_dim)
    context_mask = torch.zeros(len(batch), max_ctx, dtype=torch.bool)
    query_padded = torch.zeros(len(batch), max_qry, qry_dim)
    query_mask = torch.zeros(len(batch), max_qry, dtype=torch.bool)
    target_padded = torch.zeros(len(batch), max_qry)

    for i, item in enumerate(batch):
        n_ctx = item["context"].shape[0]
        n_qry = item["query"].shape[0]
        context_padded[i, :n_ctx] = item["context"]
        context_mask[i, :n_ctx] = True
        query_padded[i, :n_qry] = item["query"]
        query_mask[i, :n_qry] = True
        target_padded[i, :n_qry] = item["target"]

    return {
        "context": context_padded,
        "context_mask": context_mask,
        "query": query_padded,
        "query_mask": query_mask,
        "target": target_padded,
    }
