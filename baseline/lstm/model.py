"""
LSTM Baseline — 条件式 Encoder-Decoder LSTM

结构:
    Context [B, N_ctx, 8] -> context_proj -> context_lstm -> context_summary [B, H]
    Query   [B, N_qry, 7] -> query_proj  -> query_lstm  -> query_hidden [B, N_qry, H]
    [query_hidden, context_summary_expand] -> MLP -> v_norm_pred [B, N_qry]
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from config import ModelConfig


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        cfg: Optional[ModelConfig] = None,
        num_layers: int = 2,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.num_layers = num_layers

        self.context_proj = nn.Linear(cfg.context_dim, cfg.d_model)
        self.query_proj = nn.Linear(cfg.query_dim, cfg.d_model)

        self.context_lstm = nn.LSTM(
            cfg.d_model,
            cfg.d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=cfg.dropout if num_layers > 1 else 0.0,
        )

        self.query_lstm = nn.LSTM(
            cfg.d_model,
            cfg.d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=cfg.dropout if num_layers > 1 else 0.0,
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context = batch["context"]       # [B, N_ctx, 8]
        query = batch["query"]           # [B, N_qry, 7]
        context_mask = batch.get("context_mask")
        query_mask = batch.get("query_mask")

        ctx_embed = self.context_proj(context)
        if context_mask is not None:
            lengths = context_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                ctx_embed, lengths, batch_first=True, enforce_sorted=False,
            )
            _, (h_n, _) = self.context_lstm(packed)
        else:
            _, (h_n, _) = self.context_lstm(ctx_embed)

        context_summary = h_n[-1]  # [B, d_model]

        qry_embed = self.query_proj(query)
        if query_mask is not None:
            lengths = query_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                qry_embed, lengths, batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.query_lstm(packed)
            query_hidden, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=query.shape[1],
            )
        else:
            query_hidden, _ = self.query_lstm(qry_embed)  # [B, N_qry, d_model]

        context_expand = context_summary.unsqueeze(1).expand_as(query_hidden)
        fusion = torch.cat([query_hidden, context_expand], dim=-1)
        v_norm_pred = self.prediction_head(fusion).squeeze(-1)

        return {
            "v_norm_pred": v_norm_pred,
            "query_mask": query_mask,
        }
