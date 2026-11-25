"""
TFT Baseline — Simplified Temporal Fusion Transformer

Adapted for cross-section velocity interpolation.

Architecture:
    1. Variable Selection: Linear gating over context (8-dim) and query (7-dim) features
    2. LSTM Encoders: Bidirectional LSTM for context sequence, LSTM for query sequence
    3. Multi-head Cross-Attention: Query attends to context using standard MHA
    4. Gated Residual Network (GRN): ELU + sigmoid gating with residual connection
    5. Prediction Head: Linear -> GELU -> Dropout -> Linear(1)

Batch dictionary:
    "context"      [B, N_ctx, 8]  — x_norm, d_norm, bank_norm, grad_norm, v_norm,
                                    Z_norm, h_max_norm, W_norm
    "query"        [B, N_qry, 7]  — x_norm, d_norm, bank_norm, grad_norm,
                                    Z_norm, h_max_norm, W_norm
    "context_mask" [B, N_ctx]
    "query_mask"   [B, N_qry]

Output:
    {"v_norm_pred": [B, N_qry], "query_mask": ...}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from config import ModelConfig


class VariableSelection(nn.Module):
    """Linear gating over input features — simplified TFT variable selection network.

    Each input feature dimension is projected to d_model and gated through
    a learned sigmoid gate. This allows the model to selectively weight
    different input features at each position.
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.gate = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear gating.

        Args:
            x: [B, N, input_dim] input features

        Returns:
            [B, N, d_model] gated feature representation
        """
        features = self.feature_proj(x)
        gate = torch.sigmoid(self.gate(x))
        return features * gate


class GRN(nn.Module):
    """Gated Residual Network with ELU activation.

    Applies a two-layer feedforward transformation with ELU activation,
    dropout, and sigmoid gating. A residual connection and layer norm
    are applied around the gated output for stable training.

    h = ELU(Linear(x))
    h = Dropout(h)
    h = Linear(h)
    gate = sigmoid(Linear(x))
    output = LayerNorm(h * gate + x)
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GRN transformation.

        Args:
            x: [B, N, d_model] or [B, d_model]

        Returns:
            [B, N, d_model] or [B, d_model] gated output
        """
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return self.norm(h * g + x)


class TFTBaseline(nn.Module):
    """Simplified Temporal Fusion Transformer for cross-section velocity interpolation.

    The model takes context points (with known velocity v_norm) and query points
    (without velocity) as input, and predicts the normalized velocity at each
    query position.

    Data flow:
        context [B,N_ctx,8] ──> VarSelect ──> BiLSTM ────┐
                                                          ├──> Cross-Attn ──> GRN ──> PredHead ──> v_norm_pred
        query   [B,N_qry,7] ──> VarSelect ──> LSTM  ────┘
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        d_model = cfg.d_model

        # ── 1. Variable Selection ──
        self.context_var_select = VariableSelection(cfg.context_dim, d_model)
        self.query_var_select = VariableSelection(cfg.query_dim, d_model)

        # ── 2. LSTM Encoders ──
        # Bidirectional LSTM on context: hidden_size = d_model // 2,
        # output = 2 * d_model // 2 = d_model
        self.context_lstm = nn.LSTM(
            d_model, d_model // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Unidirectional LSTM on query
        self.query_lstm = nn.LSTM(
            d_model, d_model,
            batch_first=True,
        )

        # ── 3. Multi-head Cross-Attention ──
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # ── 4. Gated Residual Network ──
        self.grn = GRN(d_model, cfg.dropout)

        # ── 5. Prediction Head ──
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: dict with keys "context", "query", "context_mask", "query_mask"

        Returns:
            dict with keys "v_norm_pred" [B, N_qry] and "query_mask"
        """
        context = batch["context"]               # [B, N_ctx, 8]
        query = batch["query"]                    # [B, N_qry, 7]
        context_mask = batch.get("context_mask")  # [B, N_ctx]
        query_mask = batch.get("query_mask")      # [B, N_qry]

        # ── 1. Variable Selection ──
        ctx = self.context_var_select(context)  # [B, N_ctx, d_model]
        qry = self.query_var_select(query)      # [B, N_qry, d_model]

        # ── 2. LSTM Encoders ──
        # Context: bidirectional LSTM with masked packing for valid positions
        if context_mask is not None:
            lengths = context_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                ctx, lengths, batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.context_lstm(packed)
            ctx, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=context.shape[1],
            )
        else:
            ctx, _ = self.context_lstm(ctx)  # [B, N_ctx, d_model]

        # Query: LSTM with masked packing
        if query_mask is not None:
            lengths = query_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                qry, lengths, batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.query_lstm(packed)
            qry, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=query.shape[1],
            )
        else:
            qry, _ = self.query_lstm(qry)  # [B, N_qry, d_model]

        # ── 3. Multi-head Cross-Attention ──
        # key_padding_mask: True entries are ignored by attention
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.bool()

        attn_out, _ = self.cross_attn(
            query=qry,
            key=ctx,
            value=ctx,
            key_padding_mask=key_padding_mask,
        )
        # Residual connection with layer norm
        qry = self.attn_norm(qry + attn_out)  # [B, N_qry, d_model]

        # ── 4. Gated Residual Network ──
        qry = self.grn(qry)  # [B, N_qry, d_model]

        # ── 5. Prediction Head ──
        v_norm_pred = self.pred_head(qry).squeeze(-1)  # [B, N_qry]

        return {
            "v_norm_pred": v_norm_pred,
            "query_mask": query_mask,
        }
