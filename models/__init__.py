"""
models 包 —— Hydro-Query Transformer 模型组件

Context 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
Query   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]
"""

from .encoder import ContextEncoder, QueryEncoder
from .cross_attention import CrossAttentionLayer
from .query_transformer import HydroQueryTransformer

# 生产别名 (非破坏): 产品名 XS-VelFormer 对应同一模型类
XSVelFormerModel = HydroQueryTransformer
