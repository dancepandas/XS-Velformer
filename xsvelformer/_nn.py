"""
xs_velformer._nn — 模型架构 (内联, 自包含)

从训练仓库 models/ 摘取推理必需的 nn.Module:
    RMSNorm, DistanceBiasedAttention, TransformerEncoderBlock,
    ContextEncoder, QueryEncoder, CrossAttentionLayer,
    PredictionHead, HydroQueryTransformer

与训练期 models/ 内的代码逐位一致 (仅改 import 为包内相对引用);
已训练权重 state_dict 与本实现完全兼容。

数据流:
    Context [B, N_ctx, 8] -> ContextEncoder -> H_c [B, N_ctx, d_model]
    Query   [B, N_qry, 7] -> QueryEncoder   -> H_q [B, N_qry, d_model]
    H_q, H_c -> L 层 Cross-Attention (形态感知) -> 更新后的 H_q
    H_q -> Prediction Head -> v_norm_pred [B, N_qry]

特征维度:
    Context 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
    Query   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square 归一化 (比 LayerNorm 更高效)。

    output = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class DistanceBiasedAttention(nn.Module):
    """带横向距离偏置的多头自注意力。

    偏置 b_ij = -|α| * |x_norm_i - x_norm_j|, 让物理距离近的点更关注彼此。
    """

    def __init__(self, d_model: int, num_heads: int, alpha: float = 1.0, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
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
        B, N_q, _ = query.shape
        _, N_k, _ = key.shape

        Q = self.W_q(query).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if q_positions is not None and k_positions is not None:
            q_pos = q_positions.unsqueeze(2)
            k_pos = k_positions.unsqueeze(1)
            dist_bias = -self.alpha.abs() * torch.abs(q_pos - k_pos)
            attn = attn + dist_bias.unsqueeze(1)

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        return self.W_o(out)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer 编码器块。

    x -> RMSNorm -> Self-Attention -> 残差 -> RMSNorm -> FFN -> 残差
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
        h = self.norm1(x)
        x = x + self.attn(h, h, h, positions, positions, mask)
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class ContextEncoder(nn.Module):
    """Context Encoder (§5.2): 编码已知测速线 (含 v_norm)。

    输入: [B, N_ctx, 8] -> 输出: [B, N_ctx, d_model]
    对 padding 位置显式清零。
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
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, positions, mask)
        h = self.norm(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        return h


class QueryEncoder(nn.Module):
    """Query Encoder (§5.3): 编码待预测位置 (不含流速)。

    输入: [B, N_qry, 7] -> 输出: [B, N_qry, d_model]
    对 padding 位置显式清零。
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
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, positions, mask)
        h = self.norm(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        return h


class CrossAttentionLayer(nn.Module):
    """带形态感知距离偏置的 Cross-Attention 层 (§5.4)。

    B_ij = -βx·|x| - βd·|d| - βb·|bank| - βg·|grad|, 四个系数可学习且非正。
    Query 通过聚合 Context 信息更新 hidden, 对 padding query 清零。
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        beta_x: float = 1.0,
        beta_d: float = 0.5,
        beta_bank: float = 0.5,
        beta_grad: float = 0.25,
        use_hydro_bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.beta_x = nn.Parameter(torch.tensor(beta_x))
        self.beta_d = nn.Parameter(torch.tensor(beta_d))
        self.beta_bank = nn.Parameter(torch.tensor(beta_bank))
        self.beta_grad = nn.Parameter(torch.tensor(beta_grad))
        self.use_hydro_bias = use_hydro_bias

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        q_x_norm: Optional[torch.Tensor] = None,
        q_d_norm: Optional[torch.Tensor] = None,
        q_bank_norm: Optional[torch.Tensor] = None,
        q_grad_norm: Optional[torch.Tensor] = None,
        c_x_norm: Optional[torch.Tensor] = None,
        c_d_norm: Optional[torch.Tensor] = None,
        c_bank_norm: Optional[torch.Tensor] = None,
        c_grad_norm: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_q, _ = query_hidden.shape
        _, N_c, _ = context_hidden.shape

        h_q = self.norm1(query_hidden)

        Q = self.W_q(h_q).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(context_hidden).view(B, N_c, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(context_hidden).view(B, N_c, self.num_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.use_hydro_bias:
            hydro_bias = torch.zeros(B, N_q, N_c, device=attn.device, dtype=attn.dtype)

            if q_x_norm is not None and c_x_norm is not None:
                q_x = q_x_norm.unsqueeze(2)
                c_x = c_x_norm.unsqueeze(1)
                hydro_bias = hydro_bias - self.beta_x.abs() * torch.abs(q_x - c_x)

            if q_d_norm is not None and c_d_norm is not None:
                q_d = q_d_norm.unsqueeze(2)
                c_d = c_d_norm.unsqueeze(1)
                hydro_bias = hydro_bias - self.beta_d.abs() * torch.abs(q_d - c_d)

            if q_bank_norm is not None and c_bank_norm is not None:
                q_b = q_bank_norm.unsqueeze(2)
                c_b = c_bank_norm.unsqueeze(1)
                hydro_bias = hydro_bias - self.beta_bank.abs() * torch.abs(q_b - c_b)

            if q_grad_norm is not None and c_grad_norm is not None:
                q_g = q_grad_norm.unsqueeze(2)
                c_g = c_grad_norm.unsqueeze(1)
                hydro_bias = hydro_bias - self.beta_grad.abs() * torch.abs(q_g - c_g)

            attn = attn + hydro_bias.unsqueeze(1)

        if context_mask is not None:
            attn = attn.masked_fill(~context_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        out = self.W_o(out)

        query_hidden = query_hidden + self.dropout(out)

        h = self.norm2(query_hidden)
        query_hidden = query_hidden + self.ffn(h)

        if query_mask is not None:
            query_hidden = query_hidden * query_mask.unsqueeze(-1).float()

        return query_hidden


class PredictionHead(nn.Module):
    """流速预测头: 两层 MLP (d_model -> d_model -> 1), 可选 SoftPlus 非负输出。"""

    def __init__(self, d_model: int, dropout: float = 0.1, use_softplus: bool = False):
        super().__init__()
        self.use_softplus = use_softplus
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        if use_softplus:
            self.softplus = nn.Softplus(beta=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x).squeeze(-1)
        if self.use_softplus:
            out = self.softplus(out) + self.softplus.beta * 1e-3
        return out


class HydroQueryTransformer(nn.Module):
    """XS-VelFormer (Hydro-Query Transformer) 完整模型。

    forward(batch) -> {v_norm_pred: [B, N_qry], query_mask}
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg

        self.context_encoder = ContextEncoder(
            input_dim=cfg.context_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.context_encoder_layers,
            ffn_dim=cfg.ffn_dim,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        )

        self.query_encoder = QueryEncoder(
            input_dim=cfg.query_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.query_encoder_layers,
            ffn_dim=cfg.ffn_dim,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        )

        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.ffn_dim,
                beta_x=cfg.beta_x,
                beta_d=cfg.beta_d,
                beta_bank=cfg.beta_bank,
                beta_grad=cfg.beta_grad,
                use_hydro_bias=cfg.use_hydro_bias,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.cross_attention_layers)
        ])

        self.prediction_head = PredictionHead(
            cfg.d_model, cfg.dropout,
            use_softplus=getattr(cfg, "use_softplus_output", False),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context = batch["context"]
        query = batch["query"]
        context_mask = batch.get("context_mask")
        query_mask = batch.get("query_mask")

        c_x_norm = context[:, :, 0]
        c_d_norm = context[:, :, 1]
        c_bank_norm = context[:, :, 2]
        c_grad_norm = context[:, :, 3]
        q_x_norm = query[:, :, 0]
        q_d_norm = query[:, :, 1]
        q_bank_norm = query[:, :, 2]
        q_grad_norm = query[:, :, 3]

        context_hidden = self.context_encoder(context, c_x_norm, context_mask)
        query_hidden = self.query_encoder(query, q_x_norm, query_mask)

        for cross_layer in self.cross_attention_layers:
            query_hidden = cross_layer(
                query_hidden,
                context_hidden,
                q_x_norm=q_x_norm,
                q_d_norm=q_d_norm,
                q_bank_norm=q_bank_norm,
                q_grad_norm=q_grad_norm,
                c_x_norm=c_x_norm,
                c_d_norm=c_d_norm,
                c_bank_norm=c_bank_norm,
                c_grad_norm=c_grad_norm,
                context_mask=context_mask,
                query_mask=query_mask,
            )

        v_norm_pred = self.prediction_head(query_hidden)

        return {
            "v_norm_pred": v_norm_pred,
            "query_mask": query_mask,
        }


# 生产别名 (与训练仓库 models/__init__.py 一致)
XSVelFormerModel = HydroQueryTransformer