"""
Hydro-Query Transformer 完整模型 (§5)

本文件组装 Hydro-Query Transformer 的完整前向流程:
    1. ContextEncoder:  编码已知测速线特征 (含流速 v_norm)
    2. QueryEncoder:    编码待预测位置特征 (不含流速)
    3. CrossAttention:  query 通过形态感知交叉注意力聚合 context 信息 (多层堆叠)
    4. PredictionHead:  从更新后的 query hidden 预测流速

数据流:
    Context [B, N_ctx, 8] -> ContextEncoder -> H_c [B, N_ctx, d_model]
    Query   [B, N_qry, 7] -> QueryEncoder   -> H_q [B, N_qry, d_model]
    H_q, H_c -> L 层 Cross-Attention (形态感知) -> 更新后的 H_q
    H_q -> Prediction Head -> v_norm_pred [B, N_qry]

特征维度说明:
    Context 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
    Query   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]

    x_norm     = (x - x_min) / W, 横向位置归一化 [0,1]
    d_norm     = h / h_max, 纵向水深归一化 [0,1]
    bank_norm  = 2 * min(x_norm, 1-x_norm), 相对岸距归一化 [0,1]
    grad_norm  = 局部水深梯度归一化 [0,1]
    v_norm     = (v - v_mean) / v_std, 全局标准化流速
    Z_norm     = (Z - Z_mean) / Z_std, 标准化水位
    h_max_norm = (h_max - h_max_mean) / h_max_std, 标准化最大水深
    W_norm     = (W - W_mean) / W_std, 标准化河宽
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from models.encoder import ContextEncoder, QueryEncoder
from models.cross_attention import CrossAttentionLayer
from config import ModelConfig


class PredictionHead(nn.Module):
    """流速预测头

    两层 MLP: d_model -> d_model -> 1
    使用 GELU 激活和 Dropout 正则化。
    可选 SoftPlus 输出激活保证非负物理流速。
    最终 squeeze 掉最后一维, 输出 [B, N_qry] 的流速预测。
    """

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
            # SoftPlus ensures non-negative output, eliminating physically impossible
            # negative velocity predictions after inverse normalization
            out = self.softplus(out) + self.softplus.beta * 1e-3  # tiny positive floor
        return out


class HydroQueryTransformer(nn.Module):
    """
    Hydro-Query Transformer 完整模型

    训练/推理流程:
        1. 从 batch 中取出 context, query 及对应 mask
        2. 提取位置信息 (x_norm, d_norm, bank_norm, grad_norm) 用于形态感知偏置
        3. ContextEncoder 编码已知测速线
        4. QueryEncoder 编码待预测位置
        5. 多层 CrossAttention 让 query 聚合 context 信息 (形态感知距离偏置)
        6. PredictionHead 预测各 query 位置的标准化流速

    Context 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
    Query   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg

        # Context Encoder: 编码含流速的已知测速线
        self.context_encoder = ContextEncoder(
            input_dim=cfg.context_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.context_encoder_layers,
            ffn_dim=cfg.ffn_dim,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        )

        # Query Encoder: 编码不含流速的待预测位置
        self.query_encoder = QueryEncoder(
            input_dim=cfg.query_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.query_encoder_layers,
            ffn_dim=cfg.ffn_dim,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
        )

        # 多层 Cross-Attention: query 聚合 context 信息 (形态感知距离偏置)
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
                enable_bias_x=getattr(cfg, "enable_bias_x", True),
                enable_bias_d=getattr(cfg, "enable_bias_d", True),
                enable_bias_bank=getattr(cfg, "enable_bias_bank", True),
                enable_bias_grad=getattr(cfg, "enable_bias_grad", True),
                dropout=cfg.dropout,
            )
            for _ in range(cfg.cross_attention_layers)
        ])

        # 流速预测头
        self.prediction_head = PredictionHead(
            cfg.d_model, cfg.dropout,
            use_softplus=getattr(cfg, 'use_softplus_output', False),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            batch: 包含以下键的字典
                context:       [B, N_ctx, 8] context 特征
                query:         [B, N_qry, 7] query 特征
                context_mask:  [B, N_ctx] bool, True=有效
                query_mask:    [B, N_qry] bool, True=有效

        Returns:
            v_norm_pred: [B, N_qry] 预测的标准化流速
            query_mask:  [B, N_qry] 原样返回, 供损失函数使用
        """
        context = batch["context"]       # [B, N_ctx, 8]
        query = batch["query"]           # [B, N_qry, 7]
        context_mask = batch.get("context_mask")
        query_mask = batch.get("query_mask")

        # 提取位置和形态特征: context [x_norm, d_norm, bank_norm, grad_norm] = 列 0,1,2,3
        c_x_norm = context[:, :, 0]
        c_d_norm = context[:, :, 1]
        c_bank_norm = context[:, :, 2]
        c_grad_norm = context[:, :, 3]
        # 提取位置和形态特征: query [x_norm, d_norm, bank_norm, grad_norm] = 列 0,1,2,3
        q_x_norm = query[:, :, 0]
        q_d_norm = query[:, :, 1]
        q_bank_norm = query[:, :, 2]
        q_grad_norm = query[:, :, 3]

        # 编码 context 和 query
        context_hidden = self.context_encoder(context, c_x_norm, context_mask)
        query_hidden = self.query_encoder(query, q_x_norm, query_mask)

        # 多层 Cross-Attention: 逐层更新 query hidden (形态感知距离偏置)
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

        # 预测流速
        v_norm_pred = self.prediction_head(query_hidden)

        return {
            "v_norm_pred": v_norm_pred,
            "query_mask": query_mask,
        }
