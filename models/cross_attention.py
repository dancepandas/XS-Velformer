"""
交叉注意力模块 —— Hydro-Query Transformer 的核心组件 (§5.4)

本文件实现带形态感知距离偏置的 Cross-Attention 层, 是 Hydro-Query Transformer 的核心:
    Query hidden (待预测位置) 通过 cross-attention 聚合 Context hidden (已知测速线) 的信息。

形态感知距离偏置设计:
    B_j_i = -βx * |x_norm_q_j - x_norm_c_i|
           -βd * |d_norm_q_j - d_norm_c_i|
           -βb * |bank_norm_q_j - bank_norm_c_i|
           -βg * |grad_norm_q_j - grad_norm_c_i|

    - βx 控制横向距离 (起点距) 对注意力的影响
    - βd 控制纵向距离 (水深) 对注意力的影响
    - βb 控制相对岸距差异对注意力的影响 (近岸优先参考近岸, 主槽优先参考主槽)
    - βg 控制局部水深梯度差异对注意力的影响 (形态相似区域优先互相关注)
    - 四个系数都是可学习参数, 取绝对值保证偏置非正

    这使得:
    - 物理距离近的 query-context 对注意力更强
    - 岸距角色相似的点之间注意力更强
    - 局部断面形态相似的点之间注意力更强
    - 远距离或形态差异大的 context 对 query 的影响被抑制

输出处理:
    对 padding 的 query 位置显式清零, 防止无效表示污染后续计算。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .encoder import RMSNorm


class CrossAttentionLayer(nn.Module):
    """带形态感知距离偏置的 Cross-Attention 层

    数据流:
        query_hidden  -> W_q -> Q   (来自 QueryEncoder 的输出)
        context_hidden -> W_k -> K   (来自 ContextEncoder 的输出)
        context_hidden -> W_v -> V

        attn = softmax(Q·K^T / sqrt(d_k) + hydro_bias)  (屏蔽 padding context)
        out = attn · V
        query_hidden = query_hidden + dropout(out)          (残差连接)
        query_hidden = query_hidden + FFN(norm(query_hidden)) (FFN 残差)

    最终对 padding query 位置清零。

    Args:
        d_model:   模型维度
        num_heads: 注意力头数
        ffn_dim:   FFN 中间层维度
        beta_x:    横向距离偏置初始系数
        beta_d:    水深距离偏置初始系数
        beta_bank: 岸距偏置初始系数
        beta_grad: 梯度偏置初始系数
        use_hydro_bias: 总开关 (False 时四个偏置全部关闭)
        enable_bias_x/d/bank/grad: 逐偏置消融开关, 仅在 use_hydro_bias=True
                    时生效; 被关闭的偏置项不参与前向 (beta 参数保留以兼容
                    state_dict, 但不接收梯度)
        dropout:   dropout 率
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
        enable_bias_x: bool = True,
        enable_bias_d: bool = True,
        enable_bias_bank: bool = True,
        enable_bias_grad: bool = True,
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
        self.enable_bias_x = enable_bias_x
        self.enable_bias_d = enable_bias_d
        self.enable_bias_bank = enable_bias_bank
        self.enable_bias_grad = enable_bias_grad

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
        """
        前向传播

        Args:
            query_hidden:   [B, N_q, d_model] QueryEncoder 输出
            context_hidden: [B, N_c, d_model] ContextEncoder 输出
            q_x_norm:       [B, N_q] query 横向位置
            q_d_norm:       [B, N_q] query 水深
            q_bank_norm:    [B, N_q] query 相对岸距
            q_grad_norm:    [B, N_q] query 局部水深梯度
            c_x_norm:       [B, N_c] context 横向位置
            c_d_norm:       [B, N_c] context 水深
            c_bank_norm:    [B, N_c] context 相对岸距
            c_grad_norm:    [B, N_c] context 局部水深梯度
            context_mask:   [B, N_c] bool, True=有效 context
            query_mask:     [B, N_q] bool, True=有效 query

        Returns:
            [B, N_q, d_model] 更新后的 query hidden
        """
        B, N_q, _ = query_hidden.shape
        _, N_c, _ = context_hidden.shape

        # Pre-LN: 先归一化 query_hidden
        h_q = self.norm1(query_hidden)

        # 线性投影并拆分多头
        Q = self.W_q(h_q).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(context_hidden).view(B, N_c, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(context_hidden).view(B, N_c, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 形态感知距离偏置 (逐偏置消融: enable_bias_* 为 False 的项跳过)
        if self.use_hydro_bias:
            hydro_bias = torch.zeros(B, N_q, N_c, device=attn.device, dtype=attn.dtype)

            if self.enable_bias_x and q_x_norm is not None and c_x_norm is not None:
                q_x = q_x_norm.unsqueeze(2)
                c_x = c_x_norm.unsqueeze(1)
                hydro_bias = hydro_bias - self.beta_x.abs() * torch.abs(q_x - c_x)

            if self.enable_bias_d and q_d_norm is not None and c_d_norm is not None:
                q_d = q_d_norm.unsqueeze(2)
                c_d = c_d_norm.unsqueeze(1)
                hydro_bias = hydro_bias - self.beta_d.abs() * torch.abs(q_d - c_d)

            if self.enable_bias_bank and q_bank_norm is not None and c_bank_norm is not None:
                q_b = q_bank_norm.unsqueeze(2)
                c_b = c_bank_norm.unsqueeze(1)
                hydro_bias = hydro_bias - self.beta_bank.abs() * torch.abs(q_b - c_b)

            if self.enable_bias_grad and q_grad_norm is not None and c_grad_norm is not None:
                q_g = q_grad_norm.unsqueeze(2)
                c_g = c_grad_norm.unsqueeze(1)
                hydro_bias = hydro_bias - self.beta_grad.abs() * torch.abs(q_g - c_g)

            attn = attn + hydro_bias.unsqueeze(1)

        # 屏蔽 padding context 位置
        if context_mask is not None:
            attn = attn.masked_fill(~context_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和并合并多头
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        out = self.W_o(out)

        # 残差连接 + dropout
        query_hidden = query_hidden + self.dropout(out)

        # Pre-LN FFN 残差
        h = self.norm2(query_hidden)
        query_hidden = query_hidden + self.ffn(h)

        # 对 padding query 位置显式清零
        if query_mask is not None:
            query_hidden = query_hidden * query_mask.unsqueeze(-1).float()

        return query_hidden
