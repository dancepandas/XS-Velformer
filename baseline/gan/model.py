"""
Conditional GAN Baseline — LSGAN

Generator:
    Context [B, N_ctx, 8] -> context_lstm -> context_summary
    Query   [B, N_qry, 7] -> query_lstm -> query_hidden
    Noise z -> noise_proj
    [query_hidden, context_summary, noise_proj] -> MLP -> v_norm_fake

Discriminator:
    Context [B, N_ctx, 8] -> context_lstm -> context_summary
    [query(7维), v_norm(1维), context_summary] -> MLP -> real/fake score
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from config import ModelConfig


class ConditionalGANGenerator(nn.Module):
    def __init__(
        self,
        cfg: Optional[ModelConfig] = None,
        num_layers: int = 2,
        z_dim: int = 16,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.z_dim = z_dim

        self.context_proj = nn.Linear(cfg.context_dim, cfg.d_model)
        self.context_lstm = nn.LSTM(
            cfg.d_model,
            cfg.d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=cfg.dropout if num_layers > 1 else 0.0,
        )

        self.query_proj = nn.Linear(cfg.query_dim, cfg.d_model)
        self.query_lstm = nn.LSTM(
            cfg.d_model,
            cfg.d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=cfg.dropout if num_layers > 1 else 0.0,
        )

        self.noise_proj = nn.Linear(z_dim, cfg.d_model)

        self.generator_head = nn.Sequential(
            nn.Linear(3 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def encode_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx_embed = self.context_proj(context)
        if context_mask is not None:
            lengths = context_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                ctx_embed, lengths, batch_first=True, enforce_sorted=False,
            )
            _, (h_n, _) = self.context_lstm(packed)
        else:
            _, (h_n, _) = self.context_lstm(ctx_embed)
        return h_n[-1]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        z: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        context = batch["context"]
        query = batch["query"]
        context_mask = batch.get("context_mask")
        query_mask = batch.get("query_mask")

        B, N_qry, _ = query.shape

        context_summary = self.encode_context(context, context_mask)

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

        if z is None:
            z = torch.randn(B, N_qry, self.z_dim, device=query.device)
        noise_embed = self.noise_proj(z)

        context_expand = context_summary.unsqueeze(1).expand_as(query_hidden)
        fusion = torch.cat([query_hidden, context_expand, noise_embed], dim=-1)
        v_norm_fake = self.generator_head(fusion).squeeze(-1)

        return {
            "v_norm_pred": v_norm_fake,
            "query_mask": query_mask,
        }


class ConditionalGANDiscriminator(nn.Module):
    def __init__(
        self,
        cfg: Optional[ModelConfig] = None,
        num_layers: int = 2,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.d_model = cfg.d_model

        self.context_proj = nn.Linear(cfg.context_dim, cfg.d_model)
        self.context_lstm = nn.LSTM(
            cfg.d_model,
            cfg.d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=cfg.dropout if num_layers > 1 else 0.0,
        )

        disc_input_dim = cfg.query_dim + 1 + cfg.d_model
        self.disc_head = nn.Sequential(
            nn.Linear(disc_input_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def encode_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx_embed = self.context_proj(context)
        if context_mask is not None:
            lengths = context_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                ctx_embed, lengths, batch_first=True, enforce_sorted=False,
            )
            _, (h_n, _) = self.context_lstm(packed)
        else:
            _, (h_n, _) = self.context_lstm(ctx_embed)
        return h_n[-1]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        v_norm: torch.Tensor,
    ) -> torch.Tensor:
        context = batch["context"]
        query = batch["query"]
        context_mask = batch.get("context_mask")

        context_summary = self.encode_context(context, context_mask)
        B, N_qry, _ = query.shape
        context_expand = context_summary.unsqueeze(1).expand(B, N_qry, -1)
        v_norm_expand = v_norm.unsqueeze(-1)

        disc_input = torch.cat([query, v_norm_expand, context_expand], dim=-1)
        scores = self.disc_head(disc_input).squeeze(-1)

        return scores
