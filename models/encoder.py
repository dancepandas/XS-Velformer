"""
编码器模块 —— Context Encoder 和 Query Encoder

本文件实现 Hydro-Query Transformer 中两个核心编码器:
    1. ContextEncoder: 编码已知测速线 (context) 的特征
    2. QueryEncoder:   编码待预测位置 (query) 的特征

两者共享相同的 Transformer 编码器结构, 仅输入维度不同:
    - Context 输入: context_dim=8 → [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
    - Query   输入: query_dim=7  → [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]

核心组件:
    - RMSNorm:         Root Mean Square 归一化, 比 LayerNorm 更高效
    - DistanceBiasedAttention: 带横向距离偏置的多头自注意力
        偏置公式: b_ij = -α * |x_norm_i - x_norm_j|
        让物理距离近的点更关注彼此, 远距离点注意力衰减
    - TransformerEncoderBlock: Pre-LN 结构的编码器块
        先归一化再计算, 训练更稳定

输出处理:
    编码器输出会对 padding 位置显式清零, 避免后续模块误用无效表示。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square 归一化

    与 LayerNorm 相比不需要计算均值, 仅除以 RMS 再乘以可学习权重。
    计算量更小, 在 Transformer 中常作为 LayerNorm 的替代。

    公式: output = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class DistanceBiasedAttention(nn.Module):
    """带横向距离偏置的多头自注意力

    在标准 Scaled Dot-Product Attention 基础上, 加入位置距离偏置:
        b_ij = -α * |x_norm_i - x_norm_j|

    其中 α 是可学习参数 (取绝对值保证偏置非正), 使得:
    - 物理距离近的测速线之间注意力更强
    - 远距离测速线的注意力被抑制

    这对河道断面数据尤为重要, 因为流速在空间上具有局部连续性。

    Args:
        d_model:   模型维度
        num_heads: 多头注意力头数
        alpha:     距离偏置的初始缩放系数
        dropout:   注意力 dropout 率
    """

    def __init__(self, d_model: int, num_heads: int, alpha: float = 1.0, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 可学习距离偏置缩放系数, 取绝对值保证偏置非正 (近距离偏好)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_positions: Optional[torch.Tensor] = None,
        k_positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            query:       [B, N_q, d_model] 查询张量
            key:         [B, N_k, d_model] 键张量
            value:       [B, N_k, d_model] 值张量
            q_positions: [B, N_q] 查询位置的 x_norm (用于距离偏置)
            k_positions: [B, N_k] 键位置的 x_norm (用于距离偏置)
            mask:        [B, N_k] bool 类型, True 表示有效位置

        Returns:
            [B, N_q, d_model] 注意力输出
        """
        B, N_q, _ = query.shape
        _, N_k, _ = key.shape

        # 线性投影并拆分多头: [B, N, d_model] -> [B, heads, N, d_k]
        Q = self.W_q(query).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product: [B, heads, N_q, d_k] x [B, heads, d_k, N_k] -> [B, heads, N_q, N_k]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 加入横向距离偏置: b_ij = -|α| * |x_i - x_j|
        if q_positions is not None and k_positions is not None:
            q_pos = q_positions.unsqueeze(2)   # [B, N_q, 1]
            k_pos = k_positions.unsqueeze(1)   # [B, 1, N_k]
            dist_bias = -self.alpha.abs() * torch.abs(q_pos - k_pos)  # [B, N_q, N_k]
            attn = attn + dist_bias.unsqueeze(1)  # 广播到 [B, heads, N_q, N_k]

        # 对 padding 位置 (mask=False) 的注意力权重设为 -inf
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和并合并多头
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        return self.W_o(out)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer 编码器块

    结构: x -> RMSNorm -> Self-Attention -> 残差 -> RMSNorm -> FFN -> 残差

    Pre-LN (先归一化再计算) 比 Post-LN 训练更稳定,
    不需要 warmup 即可收敛。

    Args:
        d_model:   模型维度
        num_heads: 注意力头数
        ffn_dim:   FFN 中间层维度
        alpha:     距离偏置缩放系数
        dropout:   dropout 率
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, alpha: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.attn = DistanceBiasedAttention(d_model, num_heads, alpha, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN: 先归一化再计算注意力
        h = self.norm1(x)
        x = x + self.attn(h, h, h, positions, positions, mask)
        # Pre-LN: 先归一化再过 FFN
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class ContextEncoder(nn.Module):
    """
    Context Encoder (§5.2)

    编码已知测速线 (context) 的特征。
    Context 点包含流速信息 v_norm, 通过自注意力聚合空间关系。

    输入:  [B, N_ctx, 8] → [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
    输出:  [B, N_ctx, d_model]

    输出对 padding 位置 (mask=False) 显式清零,
    避免后续 cross-attention 误用无效表示。

    Args:
        input_dim:  输入特征维度 (默认 6)
        d_model:    模型维度
        num_heads:  注意力头数
        num_layers: 编码器层数
        ffn_dim:    FFN 中间层维度
        alpha:      距离偏置缩放系数
        dropout:    dropout 率
    """

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 256,
        alpha: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ffn_dim, alpha, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 输入投影: [B, N_ctx, 6] -> [B, N_ctx, d_model]
        h = self.input_proj(x)
        # 逐层编码, 每层包含自注意力 + FFN
        for layer in self.layers:
            h = layer(h, positions, mask)
        # 最终归一化
        h = self.norm(h)
        # 对 padding 位置显式清零, 避免后续误用
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        return h


class QueryEncoder(nn.Module):
    """
    Query Encoder (§5.3)

    编码待预测位置 (query) 的特征。
    Query 点不包含流速信息, 只有位置和全局特征。

    输入:  [B, N_qry, 7] → [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]
    输出:  [B, N_qry, d_model]

    输出对 padding 位置 (mask=False) 显式清零,
    避免后续 cross-attention 和 prediction head 误用无效表示。

    Args:
        input_dim:  输入特征维度 (默认 5)
        d_model:    模型维度
        num_heads:  注意力头数
        num_layers: 编码器层数
        ffn_dim:    FFN 中间层维度
        alpha:      距离偏置缩放系数
        dropout:    dropout 率
    """

    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        alpha: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ffn_dim, alpha, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 输入投影: [B, N_qry, 5] -> [B, N_qry, d_model]
        h = self.input_proj(x)
        # 逐层编码
        for layer in self.layers:
            h = layer(h, positions, mask)
        # 最终归一化
        h = self.norm(h)
        # 对 padding 位置显式清零
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        return h
