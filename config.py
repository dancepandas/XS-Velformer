"""
配置模块 —— Hydro-Query Transformer 河道断面流速插补与测线加密方案的全部超参数定义

特征维度说明 (根据方案文档 §3, §4):
    Context 点 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
    Query 点   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]

    其中:
        x_norm     = (x - x_min) / W, 值域 [0, 1], 左对齐归一化
        d_norm     = h / h_max, 值域 [0, 1], h 为测速线水深 (非河底高程)
        bank_norm  = 2 * min(x_norm, 1 - x_norm), 值域 [0, 1], 0=岸边, 1=断面中心
        grad_norm  = 局部水深梯度 / max(梯度), 值域 [0, 1], 形态变化强度
        v_norm     = (v - mean(v_train)) / std(v_train), 全局 StandardScaler
        Z_norm     = (Z - mean(Z_train)) / std(Z_train), 全局标准化
        h_max_norm = (h_max - mean(h_max_train)) / std(h_max_train), 全局标准化
        W_norm     = (W - mean(W_train)) / std(W_train), 全局标准化

    多站点数据河宽 W 跨站点差异极大 (1.3m~404.7m), 是区分站点的关键特征。
    测流时间 time 仅用于数据集划分, 不作为模型输入。

数据说明:
    不同站点断面测速线数量 K 不同 (10~100), 模型原生支持变长输入,
    不需要将数据统一到固定长度。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """
    Hydro-Query Transformer 模型结构超参数

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
    # 逐偏置消融开关: use_hydro_bias=True 时, 每个偏置项可单独关闭
    # (参数仍创建以保持 state_dict 兼容, 但其偏置项不参与前向计算)
    enable_bias_x: bool = True
    enable_bias_d: bool = True
    enable_bias_bank: bool = True
    enable_bias_grad: bool = True
    use_softplus_output: bool = False  # SoftPlus output for non-negative physical velocity


@dataclass
class TrainConfig:
    """训练过程超参数"""
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 300
    early_stopping_patience: int = 20
    lambda_smooth: float = 0.05
    lambda_depth: float = 0.0
    lambda_depth_corr: float = 0.02
    lambda_bank: float = 0.01
    lambda_bank_zero: float = 0.02
    lambda_grad: float = 0.01
    bank_threshold: float = 0.2
    depth_threshold: float = 0.3
    v_bank_limit: float = 0.5
    grad_smooth_gamma: float = 2.0
    bank_zero_tau: float = 0.15
    depth_zero_tau: float = 0.25
    mask_ratio_range: List[float] = field(default_factory=lambda: [0.1, 0.4])
    repeats: int = 10
    num_workers: int = 0
    scheduler: str = "cosine"


@dataclass
class DataConfig:
    """
    数据处理与划分超参数

    Attributes:
        data_path: sections.pkl 文件路径
        split_mode: 划分模式, "station" 按测站划分 (测试跨站泛化),
                    "time" 按时间划分 (避免未来数据泄漏)
        train_ratio: 训练集占比
        val_ratio: 验证集占比
        std_threshold: 标准差阈值, 低于此值视为退化情况
    """
    data_path: str = "v_data/processed/sections.pkl"
    split_mode: str = "stratified"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    random_seed: int = 42
    # True: 拒绝"全部缺 time"的旧语料回退 (station_and_time 划分)。新语料协议必须 True,
    # 旧语料复现保持 False —— 见 docs/revision/rerun-protocol.md §2
    require_time: bool = False
