"""
xs_velformer._config — 模型结构超参数 (内联, 自包含)

从训练仓库 config.py 摘取推理必需的 ModelConfig;
TrainConfig / DataConfig 属于训练侧, 不进 SDK。
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """XS-VelFormer (Hydro-Query Transformer) 模型结构超参数。

    ┌─────────────────────────────────────────────────────────────────┐
    │  Context Encoder (context_encoder_layers 层)                    │
    │    输入: context_dim=8 → [x_norm, d_norm, bank_norm, grad_norm, │
    │                          v_norm, Z_norm, h_max_norm, W_norm]    │
    │    输出: context_hidden [B, N_ctx, d_model]                    │
    ├─────────────────────────────────────────────────────────────────┤
    │  Query Encoder (query_encoder_layers 层)                       │
    │    输入: query_dim=7 → [x_norm, d_norm, bank_norm, grad_norm,   │
    │                          Z_norm, h_max_norm, W_norm]            │
    │    输出: query_hidden [B, N_qry, d_model]                      │
    ├─────────────────────────────────────────────────────────────────┤
    │  Cross-Attention (cross_attention_layers 层)                   │
    │    Q = query_hidden, K/V = context_hidden                      │
    │    带 beta_x·|x| + beta_d·|d| + beta_bank·|bank|              │
    │       + beta_grad·|grad| 形态感知距离偏置                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  Prediction Head (MLP)                                         │
    │    d_model → d_model → 1 → v_norm_pred                        │
    └─────────────────────────────────────────────────────────────────┘
    """
    d_model: int = 256
    num_heads: int = 8
    context_encoder_layers: int = 4
    query_encoder_layers: int = 2
    cross_attention_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    context_dim: int = 8
    query_dim: int = 7
    alpha: float = 1.0
    beta_x: float = 1.0
    beta_d: float = 0.5
    beta_bank: float = 0.5
    beta_grad: float = 0.25
    use_hydro_bias: bool = True
    use_softplus_output: bool = False  # SoftPlus output for non-negative physical velocity