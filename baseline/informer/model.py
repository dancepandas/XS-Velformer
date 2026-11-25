"""
Informer Baseline — ProbSparse self-attention + distilling for cross-section velocity interpolation.

Architecture (simplified Informer adapted for spatial cross-section data):

  1. Input Projection:  Linear(context: 8→d_model), Linear(query: 7→d_model)
  2. Context Encoder:
       Layer 0:  ProbSparse Self-Attn → FFN → Conv1d+ELU+MaxPool1d (distilling, halves seq)
       Layer 1:  ProbSparse Self-Attn → FFN → Conv1d+ELU+MaxPool1d (distilling, halves seq)
       (at least 2 layers; d_model kept throughout)
  3. Cross-Attention:  query (Q) attends to distilled context (K, V) — standard multi-head
  4. Prediction Head:  Linear → GELU → Dropout → Linear(1)

ProbSparse attention (Zhou et al., AAAI 2021):
  - Sparsity score:  M(q_i, K) = max_j(q_i·K_j^T/√d) - mean_j(q_i·K_j^T/√d)
  - Select top-u queries by M; only those get full dot-product attention
  - Remaining queries use mean V (the "stale" context summary)
  - u = factor * ln(N_ctx), clamped to [1, N_ctx]

Forward interface:
  forward(batch: Dict) -> Dict with {"v_norm_pred": [B, N_qry], "query_mask": ...}
  batch keys: "context" [B, N_ctx, 8], "query" [B, N_qry, 7],
              "context_mask" [B, N_ctx], "query_mask" [B, N_qry]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from config import ModelConfig


# ---------------------------------------------------------------------------
# ProbSparse Self-Attention
# ---------------------------------------------------------------------------

class ProbSparseAttention(nn.Module):
    """
    ProbSparse multi-head self-attention.

    Instead of computing the full N x N attention matrix, we measure a sparsity
    score M(q, K) = max(qK^T/√d) - mean(qK^T/√d) per query, select the top-u
    queries (the most "active" ones), and compute full attention only for them.
    The remaining queries simply receive the mean V — a cheap context summary.

    This is the core innovation of the Informer (Zhou et al., AAAI 2021),
    reducing the quadratic O(N²) attention cost to O(N·u).

    The sparsity measurement itself uses a random sample of K keys (factor *
    ln(N) keys) for further efficiency, consistent with the paper.
    """

    def __init__(self, cfg: ModelConfig, factor: int = 5):
        super().__init__()
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        self.factor = factor          # sampling multiplier for K in measurement
        self.scale = self.head_dim ** 0.5

        self.W_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout_attn = nn.Dropout(cfg.dropout)

    def _sparsity_score(
        self,
        Q: torch.Tensor,          # [B, H, N, d_k]
        K_sample: torch.Tensor,   # [B, H, n_sample, d_k]
    ) -> torch.Tensor:           # [B, H, N]
        """M(q_i, K) = max_j(q_i·K_j^T/√d) - mean_j(q_i·K_j^T/√d)"""
        scores = torch.matmul(Q, K_sample.transpose(-2, -1)) / self.scale
        return scores.max(dim=-1)[0] - scores.mean(dim=-1)

    def forward(
        self,
        x: torch.Tensor,                     # [B, N, d_model]
        mask: Optional[torch.Tensor] = None,  # [B, N] bool, True = valid
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads

        # ---- linear projections + reshape to [B, H, N, d_k] -----------------
        Q = self.W_q(x).view(B, N, H, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, H, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, self.head_dim).transpose(1, 2)

        # ---- sparsity measurement on a sample of valid keys (per-sample) ----
        if mask is not None:
            L = mask.sum(dim=1).long()                       # [B] 各样本有效长度
            if int(L.min().item()) < 1:
                raise ValueError(
                    "ProbSparseAttention: 存在全部 padding 的样本, 无法计算注意力")
            # 每样本独立的采样数与 top-u 数 (u=factor*ln(L)+1, 封顶 L),
            # 消除按 batch 最短长度收缩带来的跨样本耦合
            log_L = torch.log(L.to(torch.float32))
            n_b = (self.factor * log_L + 1).long().clamp_(min=1)
            n_b = torch.minimum(n_b, L)
            u_b = (self.factor * log_L + 1).long().clamp_(min=1)
            u_b = torch.minimum(u_b, L)
            n_max = int(n_b.max().item())
            u_max = int(u_b.max().item())

            # 只在有效前缀内采样 key: 前缀打乱后取前 n_max 个, 前 n_b[b] 个必为有效位置
            scores = torch.rand(B, N, device=x.device)
            scores = scores.masked_fill(~mask, float("-inf"))
            sample_idx = scores.argsort(dim=-1, descending=True)[:, :n_max]
            idx_exp = sample_idx.unsqueeze(1).unsqueeze(-1).expand(
                B, H, n_max, self.head_dim)
            K_sample = torch.gather(K, 2, idx_exp)     # [B, H, n_max, d_k]

            # 稀疏度评分只统计各样本自己的 n_b 个采样 key
            m_sample = (torch.arange(n_max, device=x.device).unsqueeze(0)
                        < n_b.unsqueeze(1)).view(B, 1, 1, n_max)
            sc = torch.matmul(Q, K_sample.transpose(-2, -1)) / self.scale
            mean_sc = (sc * m_sample).sum(-1) / m_sample.sum(-1).clamp(min=1)
            max_sc = sc.masked_fill(~m_sample, float("-inf")).max(dim=-1)[0]
            M = max_sc - mean_sc                                        # [B, H, N]

            # padded query 不参与 top-u 竞争 (否则有效 query 被挤成 mean-V 回退)
            M = M.masked_fill(~mask.unsqueeze(1), float("-inf"))
            _, top_idx = torch.topk(M, u_max, dim=-1)                   # [B, H, u_max]
            top_idx_exp = top_idx.unsqueeze(-1).expand(B, H, u_max, self.head_dim)
            Q_top = torch.gather(Q, 2, top_idx_exp)

            # ---- full attention for top-u queries only -------------------------
            attn_top = torch.matmul(Q_top, K.transpose(-2, -1)) / self.scale
            attn_top = attn_top.masked_fill(
                ~mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            attn_top = self.dropout_attn(F.softmax(attn_top, dim=-1))
            # NaN 只出现在整行均为 -inf 的槽位 (该样本有效 key 数 < n_max 时,
            # padding 槽位 softmax 分母为 0); 存在有效 key 的行不受影响。
            # nan_to_num 清掉 NaN; rank ≥ u_b 的槽位随后被 V_mean 回退整体覆盖,
            # 因此 NaN 归零值不会进入输出。
            attn_top = torch.nan_to_num(attn_top, nan=0.0)
            V_top = torch.matmul(attn_top, V)                # [B, H, u_max, d_k]
        else:
            # ---- 无 mask 路径: 保持原始 ProbSparse 行为 -------------------------
            n_sample = max(1, min(
                self.factor * int(math.log(N)) + 1 if N > 1 else 1, N
            ))
            sample_idx = torch.randperm(N, device=x.device)[:n_sample]
            K_sample = K[:, :, sample_idx, :]              # [B, H, n_sample, d_k]
            M = self._sparsity_score(Q, K_sample)           # [B, H, N]
            u = max(1, min(
                self.factor * int(math.log(N)) + 1 if N > 1 else 1, N
            ))
            _, top_idx = torch.topk(M, u, dim=-1)           # [B, H, u]
            u_max = u
            top_idx_exp = top_idx.unsqueeze(-1).expand(B, H, u, self.head_dim)
            Q_top = torch.gather(Q, 2, top_idx_exp)
            attn_top = torch.matmul(Q_top, K.transpose(-2, -1)) / self.scale
            attn_top = self.dropout_attn(F.softmax(attn_top, dim=-1))
            V_top = torch.matmul(attn_top, V)

        # ---- mean V for remaining (non-top-u) queries ----------------------
        if mask is not None:
            mask_float = mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, N, 1]
            V_sum = (V * mask_float).sum(dim=2, keepdim=True)     # [B, H, 1, d_k]
            valid_count = mask_float.sum(dim=2, keepdim=True).clamp(min=1)
            V_mean = V_sum / valid_count                           # [B, H, 1, d_k]
        else:
            V_mean = V.mean(dim=2, keepdim=True)                   # [B, H, 1, d_k]

        if mask is not None:
            # rank ≥ u_b[b] 的槽位不属于该样本的 top-u, 回退为 mean-V,
            # 保证每样本的生效 top-u 数与单独推理时一致 (无跨样本耦合)
            keep = (torch.arange(u_max, device=x.device).view(1, 1, u_max, 1)
                    < u_b.view(B, 1, 1, 1))
            V_top = torch.where(keep, V_top,
                                V_mean.expand(B, H, u_max, self.head_dim))

        # ---- fill output: start with mean V, scatter top-u in place --------
        out = V_mean.expand(B, H, N, self.head_dim).clone()

        # scatter_:  along dim=2, index = target position, src = value
        out.scatter_(2, top_idx_exp, V_top)

        # ---- re-assemble [B, N, D] -----------------------------------------
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Encoder building blocks
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """One Informer encoder layer: ProbSparse self-attention + FFN, both with
    residual connection, layer norm, and dropout."""

    def __init__(self, cfg: ModelConfig, attention: ProbSparseAttention):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self-attention
        attn_out = self.attention(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        # feed-forward
        x = self.norm2(x + self.ffn(x))
        if mask is not None:
            # 残差+FFN 会让 padding 行重新非零, 归零以保护下一层 ConvLayer 的边界
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class ConvLayer(nn.Module):
    """
    Self-attention distilling layer from Informer.

    Conv1d(kernel=3) → ELU → MaxPool1d(kernel=3, stride=2, padding=1)

    This halves the sequence length, progressively distilling a focused
    self-attention feature map in the encoder.  The mask is updated to
    reflect the new valid length.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=cfg.d_model,
            out_channels=cfg.d_model,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,                     # [B, N, d_model]
        mask: Optional[torch.Tensor] = None,  # [B, N]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Conv1d expects [B, C, L]; ELU activation
        x = x.transpose(1, 2)                # [B, d_model, N]
        x = F.elu(self.conv(x))              # [B, d_model, N]
        if mask is not None:
            # 池化前把 padding 位置置 -inf: 用 0 会胜过负的 ELU 值, 破坏池化语义
            x = x.masked_fill(~mask.unsqueeze(1), float("-inf"))
        x = self.pool(x)                     # [B, d_model, out_N]
        x = x.transpose(1, 2)                # [B, out_N, d_model]

        if mask is not None:
            # Valid positions are at the head of the sequence.
            # MaxPool1d(k=3,s=2,p=1) reduces N→ceil(N/2).
            valid_len = mask.sum(dim=-1).long()
            out_len = x.shape[1]
            new_len = ((valid_len + 1) // 2).clamp(max=out_len)
            new_mask = torch.zeros(mask.shape[0], out_len, dtype=torch.bool, device=mask.device)
            for b in range(mask.shape[0]):
                new_mask[b, :new_len[b]] = True
            # 全 -inf 池化窗 → -inf, 归零 (注意: -inf*0=NaN, 必须用 masked_fill)
            x = x.masked_fill(~new_mask.unsqueeze(-1), 0.0)
        else:
            new_mask = None

        return x, new_mask


class InformerEncoder(nn.Module):
    """Stack of (EncoderLayer + ConvLayer) pairs.

    Each pair:
        1. ProbSparse self-attention + FFN (EncoderLayer)
        2. Conv1d + ELU + MaxPool1d distilling (ConvLayer) — halves seq length

    At least 2 layers so the sequence is distilled at least twice.
    """

    def __init__(self, cfg: ModelConfig, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = ProbSparseAttention(cfg)
            self.layers.append(EncoderLayer(cfg, attn))
            self.layers.append(ConvLayer(cfg))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        for layer in self.layers:
            if isinstance(layer, EncoderLayer):
                x = layer(x, mask=mask)
            else:
                x, mask = layer(x, mask=mask)     # ConvLayer updates mask too
        return x, mask


# ---------------------------------------------------------------------------
# Full Informer Baseline
# ---------------------------------------------------------------------------

class InformerBaseline(nn.Module):
    """
    Informer baseline for cross-section velocity interpolation.

    Data flow:
        context [B, N_ctx, 8]  → Linear → Encoder (ProbSparse + distilling)
        query   [B, N_qry, 7]  → Linear → Cross-Attn(Q=query, KV=ctx)
                                        → Prediction Head → [B, N_qry]
    """

    def __init__(
        self,
        cfg: Optional[ModelConfig] = None,
        encoder_layers: int = 2,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.d_model = cfg.d_model

        # ---- input projections ----------------------------------------------
        self.context_proj = nn.Linear(cfg.context_dim, cfg.d_model)
        self.query_proj = nn.Linear(cfg.query_dim, cfg.d_model)

        # ---- context encoder (ProbSparse + distilling) ----------------------
        self.encoder = InformerEncoder(cfg, num_layers=encoder_layers)

        # ---- cross-attention: query attends to distilled context ------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(cfg.d_model)
        self.cross_dropout = nn.Dropout(cfg.dropout)

        # ---- prediction head ------------------------------------------------
        self.prediction_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context = batch["context"]            # [B, N_ctx, 8]
        query = batch["query"]                # [B, N_qry, 7]
        context_mask = batch.get("context_mask")  # [B, N_ctx] bool
        query_mask = batch.get("query_mask")      # [B, N_qry] bool

        # ---- 1. project inputs to d_model -----------------------------------
        ctx = self.context_proj(context)      # [B, N_ctx, d_model]
        qry = self.query_proj(query)          # [B, N_qry, d_model]
        if context_mask is not None:
            # 投影后 padding 行是偏置向量而非零, 显式清零 (encoder 的 conv/注意力均受益)
            ctx = ctx * context_mask.unsqueeze(-1).to(ctx.dtype)

        # ---- 2. encode context (ProbSparse self-attn + distilling) ----------
        ctx_enc, ctx_mask = self.encoder(ctx, mask=context_mask)
        # ctx_enc: [B, N_ctx_reduced, d_model]  (distilled, shorter seq)

        # ---- 3. cross-attention: query → context ---------------------------
        # Build key_padding_mask: True = ignore (opposite of our convention)
        key_pad_mask = None
        if ctx_mask is not None:
            key_pad_mask = ~ctx_mask    # nn.MultiheadAttention uses True=ignore

        attn_out, _ = self.cross_attn(
            query=qry,                  # [B, N_qry, d_model]
            key=ctx_enc,                # [B, N_ctx_reduced, d_model]
            value=ctx_enc,
            key_padding_mask=key_pad_mask,  # [B, N_ctx_reduced]
        )
        qry = self.cross_norm(qry + self.cross_dropout(attn_out))

        # ---- 4. prediction head ---------------------------------------------
        v_norm_pred = self.prediction_head(qry).squeeze(-1)  # [B, N_qry]

        return {
            "v_norm_pred": v_norm_pred,
            "query_mask": query_mask,
        }
