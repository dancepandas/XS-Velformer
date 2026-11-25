"""
Conditional Diffusion Baseline — DDPM

本文件实现基于 DDPM (Denoising Diffusion Probabilistic Models) 的条件扩散模型,
作为河道断面流速插补的生成式 baseline。

核心组件:
    1. SinusoidalTimeEmbedding:  正弦时间步嵌入, 将离散时间步编码为连续向量
    2. ConditionalDiffusionDenoiser: 条件去噪网络
        - Context [B, N_ctx, 8] -> context_lstm -> context_summary (全局上下文摘要)
        - Query [B, N_qry, 7] -> query_lstm -> query_hidden (逐位置查询表示)
        - Noisy velocity x_t -> velocity_proj (噪声速度嵌入)
        - Timestep t -> sinusoidal_embedding -> time_mlp (时间步嵌入)
        - [query_hidden, context_summary, velocity_proj, time_mlp] -> MLP -> epsilon_pred
    3. DiffusionScheduler: 扩散调度器
        - 线性 beta schedule
        - q_sample: 前向加噪
        - p_sample: 单步去噪 (使用标准 DDPM 后验方差)
        - sample: 完整 DDPM 采样循环

当前仅支持 DDPM 采样方式 (sampler="ddpm"),
后续可扩展 DDIM 等快速采样方式。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from config import ModelConfig


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间步嵌入

    将离散时间步 t 编码为 d_model 维的连续向量,
    使用正弦/余弦函数使相近时间步有相似的表示。

    公式:
        emb_i = exp(-i * log(10000) / (d/2-1))
        output = [sin(t * emb_0), ..., sin(t * emb_{d/2-1}),
                  cos(t * emb_0), ..., cos(t * emb_{d/2-1})]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] 整数时间步
        Returns:
            [B, d_model] 时间步嵌入
        """
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        # 奇数维度时补零
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalDiffusionDenoiser(nn.Module):
    """条件去噪网络

    给定加噪流速 x_t、时间步 t、context 和 query 位置,
    预测添加的噪声 epsilon。

    数据流:
        context -> context_proj -> context_lstm -> context_summary [B, d_model]
        query -> query_proj -> query_lstm -> query_hidden [B, N_qry, d_model]
        x_t -> velocity_proj -> vel_embed [B, N_qry, d_model]
        t -> sinusoidal_embedding -> time_mlp -> time_embed [B, d_model]

        fusion = [query_hidden, context_summary_expand, vel_embed, time_expand]
        epsilon_pred = denoising_head(fusion)  -> [B, N_qry]

    Args:
        cfg:        模型配置
        num_layers: LSTM 层数
    """

    def __init__(
        self,
        cfg: Optional[ModelConfig] = None,
        num_layers: int = 2,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.d_model = cfg.d_model

        # Context 编码: 线性投影 + LSTM
        self.context_proj = nn.Linear(cfg.context_dim, cfg.d_model)
        self.context_lstm = nn.LSTM(
            cfg.d_model,
            cfg.d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=cfg.dropout if num_layers > 1 else 0.0,
        )

        # Query 编码: 线性投影 + LSTM
        self.query_proj = nn.Linear(cfg.query_dim, cfg.d_model)
        self.query_lstm = nn.LSTM(
            cfg.d_model,
            cfg.d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=cfg.dropout if num_layers > 1 else 0.0,
        )

        # 加噪流速嵌入: 1 维 -> d_model 维
        self.velocity_proj = nn.Linear(1, cfg.d_model)

        # 时间步嵌入: 正弦编码 + 2 层 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # 去噪头: 4*d_model -> d_model -> 1
        self.denoising_head = nn.Sequential(
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def encode_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """编码 context 为全局摘要向量 [B, d_model]"""
        ctx_embed = self.context_proj(context)
        if context_mask is not None:
            lengths = context_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                ctx_embed, lengths, batch_first=True, enforce_sorted=False,
            )
            _, (h_n, _) = self.context_lstm(packed)
        else:
            _, (h_n, _) = self.context_lstm(ctx_embed)
        return h_n[-1]  # 取最后一层的隐状态

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        前向传播: 预测噪声 epsilon

        Args:
            x_t:   [B, N_qry] 加噪流速
            t:     [B] 时间步
            batch: 包含 context, query, context_mask 的字典

        Returns:
            [B, N_qry] 预测的噪声
        """
        context = batch["context"]
        query = batch["query"]
        context_mask = batch.get("context_mask")
        query_mask = batch.get("query_mask")

        # 编码 context 为全局摘要并扩展到每个 query 位置
        context_summary = self.encode_context(context, context_mask)
        B, N_qry, _ = query.shape
        context_expand = context_summary.unsqueeze(1).expand(B, N_qry, -1)

        # 编码 query 位置特征
        qry_embed = self.query_proj(query)
        if query_mask is not None:
            lengths = query_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                qry_embed, lengths, batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.query_lstm(packed)
            query_hidden, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=N_qry,
            )
        else:
            query_hidden, _ = self.query_lstm(qry_embed)

        # 编码加噪流速
        vel_embed = self.velocity_proj(x_t.unsqueeze(-1))

        # 编码时间步并扩展
        time_embed = self.time_mlp(t)
        time_expand = time_embed.unsqueeze(1).expand_as(query_hidden)

        # 拼接所有特征并预测噪声
        fusion = torch.cat([query_hidden, context_expand, vel_embed, time_expand], dim=-1)
        epsilon_pred = self.denoising_head(fusion).squeeze(-1)

        return epsilon_pred


class DiffusionScheduler:
    """DDPM 扩散调度器

    管理前向加噪和反向去噪过程。

    前向过程 (加噪):
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    反向过程 (去噪, 标准 DDPM):
        mean_t = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_pred)
        x_{t-1} = mean_t + sqrt(posterior_variance_t) * noise  (t>0 时)

    后验方差使用标准 DDPM 公式:
        posterior_variance_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)

    当前仅支持 DDPM 采样方式, 后续可扩展 DDIM 等。

    Args:
        num_timesteps: 扩散步数
        beta_start:    beta 起始值
        beta_end:      beta 终止值
        sampler:       采样方式, 当前仅支持 "ddpm"
    """

    def __init__(
        self,
        num_timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        sampler: str = "ddpm",
    ):
        if sampler != "ddpm":
            raise ValueError(f"Only DDPM sampler is supported currently, got '{sampler}'")

        self.sampler = sampler
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 线性 beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # 标准 DDPM 后验方差: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        alpha_bars_prev = torch.cat([
            torch.ones(1),
            self.alpha_bars[:-1],
        ])
        self.posterior_variance = self.betas * (1 - alpha_bars_prev) / (1 - self.alpha_bars)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def to(self, device: torch.device):
        """将调度器张量移到指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向加噪: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0:   [B, N_qry] 原始流速
            t:     [B] 时间步
            noise: [B, N_qry] 可选噪声, 默认随机生成
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bars[t]
        # 扩展维度以匹配 x_0
        while alpha_bar_t.dim() < x_0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_scalar: int,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """单步去噪 (标准 DDPM)

        计算去噪均值, 并在 t>0 时加入后验方差噪声。

        Args:
            model:    去噪网络
            x_t:      [B, N_qry] 当前加噪状态
            t_scalar: 当前时间步 (整数)
            batch:    条件输入
        """
        B = x_t.shape[0]
        t = torch.full((B,), t_scalar, dtype=torch.long, device=x_t.device)
        epsilon_pred = model(x_t, t, batch)

        beta_t = self.betas[t_scalar]
        alpha_t = self.alphas[t_scalar]
        alpha_bar_t = self.alpha_bars[t_scalar]

        # 去噪均值
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred
        )

        if t_scalar > 0:
            # 使用标准 DDPM 后验方差 (而非简化版 sqrt(beta_t))
            variance_t = self.posterior_variance[t_scalar]
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance_t) * noise
        return mean

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        shape: tuple,
    ) -> torch.Tensor:
        """完整 DDPM 采样循环

        从纯噪声开始, 逐步去噪 T 步得到流速预测。

        Args:
            model: 去噪网络
            batch: 条件输入
            shape: 输出形状 (B, N_qry)
        """
        x = torch.randn(shape, device=self.betas.device)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t, batch)
        return x
