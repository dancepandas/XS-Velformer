"""
PatchTST Baseline — patch-based time series model adapted for spatial cross-section
velocity interpolation.

Adapts PatchTST (Nie et al., ICLR 2023) from temporal forecasting to spatial
interpolation over river cross-section verticals.

Architecture:
    1. Patching:      Divide context/query vertical sequences into overlapping
                      patches (patch_len=4, stride=2, 50% overlap).
    2. Encoder:       3-layer self-attention over context patches (pre-norm).
    3. Cross-Attn:    Query patches attend to encoded context patches via MHA.
    4. Prediction:    Flatten patches back to per-vertical predictions through
                      patch_to_point reconstruction + per-vertical refinement MLP.

Interface:
    forward(batch) -> {"v_norm_pred": [B, N_qry], "query_mask": ...}
    batch keys:
        "context"       [B, N_ctx, 8]   — context vertical features
        "query"         [B, N_qry, 7]   — query vertical features
        "context_mask"  [B, N_ctx]      — bool, True = valid
        "query_mask"    [B, N_qry]      — bool, True = valid
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from config import ModelConfig


# ---------------------------------------------------------------------------
# Transformer Encoder Layer (pre-norm)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer with pre-norm (LayerNorm before
    MHA and FFN), dropout, and residual connections.

    pre-norm:  x = x + MHA(LN(x))
               x = x + FFN(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-norm self-attention + FFN.

        Args:
            x:                 [B, S, d_model]
            key_padding_mask:  [B, S]  bool, True = masked (ignore)
        """
        # Self-attention with pre-norm
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
        )
        x = residual + self.dropout1(attn_out)

        # FFN with pre-norm
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out

        return x


# ---------------------------------------------------------------------------
# PatchTST Baseline
# ---------------------------------------------------------------------------

class PatchTSTBaseline(nn.Module):
    """PatchTST adapted for river cross-section velocity interpolation.

    Replaces temporal patching with spatial patching: each "time step" becomes
    a vertical profile position across the river cross-section.  Context
    verticals (with known velocity) are patched, self-attended, then used as
    keys/values for cross-attention from query patches.  Query patch
    representations are flattened back to per-vertical velocity predictions
    via a learned patch_to_point projection with overlap averaging.

    Args:
        cfg:                ModelConfig instance with d_model, num_heads, etc.
        patch_len:          Number of verticals per patch (default 4).
        stride:             Stride between consecutive patches (default 2,
                            50% overlap).
        num_encoder_layers: Number of self-attention encoder layers (default 3).
        max_num_patches:    Upper bound on patch count for position encoding
                            buffer (default 500).
    """

    def __init__(
        self,
        cfg: Optional[ModelConfig] = None,
        patch_len: int = 4,
        stride: int = 2,
        num_encoder_layers: int = 3,
        max_num_patches: int = 500,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.patch_len = patch_len
        self.stride = stride
        self.context_dim = cfg.context_dim
        self.query_dim = cfg.query_dim

        # ---- Context patching ------------------------------------------------
        # patch_len * context_dim  →  d_model
        self.ctx_patch_proj = nn.Linear(patch_len * self.context_dim, cfg.d_model)
        self.ctx_pos_encoding = nn.Parameter(
            torch.randn(1, max_num_patches, cfg.d_model) * 0.02
        )

        # ---- Self-attention encoder (pre-norm) -------------------------------
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.ffn_dim,
                dropout=cfg.dropout,
            )
            for _ in range(num_encoder_layers)
        ])

        # ---- Query patching --------------------------------------------------
        # patch_len * query_dim  →  d_model
        self.qry_patch_proj = nn.Linear(patch_len * self.query_dim, cfg.d_model)
        self.qry_pos_encoding = nn.Parameter(
            torch.randn(1, max_num_patches, cfg.d_model) * 0.02
        )

        # ---- Cross-attention (pre-norm) --------------------------------------
        self.cross_norm_q = nn.LayerNorm(cfg.d_model)
        self.cross_norm_kv = nn.LayerNorm(cfg.d_model)
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.num_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.cross_dropout = nn.Dropout(cfg.dropout)

        # ---- Prediction head -------------------------------------------------
        # Each query patch → patch_len scalar values (one per vertical)
        self.patch_to_point = nn.Linear(cfg.d_model, patch_len)
        # Per-vertical refinement after un-patching
        self.pred_head = nn.Sequential(
            nn.Linear(1, cfg.d_model // 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform initialisation for all weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Patching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_padding(N: int, patch_len: int, stride: int) -> int:
        """Compute zero-padding size so that (N_padded - patch_len) is
        divisible by stride, with padded length at least patch_len.

        Args:
            N:         Original sequence length.
            patch_len: Patch size.
            stride:    Patch stride.

        Returns:
            Total number of zeros to pad (right-side only).
        """
        eff = max(N, patch_len)
        remainder = (eff - patch_len) % stride
        tail = (stride - remainder) % stride
        return (eff - N) + tail

    def _pad_and_patch(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple:
        """Pad a variable-length sequence and extract overlapping patches via
        torch.Tensor.unfold.

        Args:
            x:    [B, N, D] input features.
            mask: [B, N]   bool valid mask (True = valid, None → all valid).

        Returns:
            patches:     [B, num_patches, patch_len * D]
            patch_mask:  [B, num_patches]  bool, True = valid (all elements non-padded)
            padded_len:  int
        """
        B, N, D = x.shape
        pad_size = self._compute_padding(N, self.patch_len, self.stride)
        padded_len = N + pad_size

        # Right-pad with zeros
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_size), value=False)

        # Unfold along the sequence dimension
        # x: [B, N_pad, D] → [B, D, N_pad] → unfold → [B, D, num_patches, patch_len]
        x_t = x.transpose(1, 2)  # [B, D, N_pad]
        patches = x_t.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B, D, num_patches, patch_len]
        patches = patches.permute(0, 2, 3, 1).contiguous()
        # [B, num_patches, patch_len, D]
        patches = patches.view(B, -1, self.patch_len * D)
        # [B, num_patches, patch_len * D]
        num_patches = patches.shape[1]

        # Patch mask: a patch is valid iff every element is valid
        if mask is not None:
            mask_t = mask.unsqueeze(1)  # [B, 1, N_pad]
            mask_unfold = mask_t.unfold(
                dimension=-1, size=self.patch_len, step=self.stride,
            )
            # [B, 1, num_patches, patch_len]
            patch_mask = mask_unfold.all(dim=-1).squeeze(1)  # [B, num_patches]
        else:
            patch_mask = torch.ones(
                B, num_patches, dtype=torch.bool, device=x.device,
            )

        return patches, patch_mask, padded_len

    # ------------------------------------------------------------------
    # Un-patching / reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_from_patches(
        self,
        patch_vals: torch.Tensor,
        original_len: int,
    ) -> torch.Tensor:
        """Reconstruct per-vertical values from overlapping patch predictions
        by averaging contributions from all patches that cover each vertical.

        Args:
            patch_vals:  [B, num_patches, patch_len]  patch-level predictions.
            original_len: int  length before padding.

        Returns:
            [B, original_len]  per-vertical values.
        """
        B, num_patches, plen = patch_vals.shape
        device = patch_vals.device

        # padded_len = (num_patches - 1) * stride + patch_len
        padded_len = (num_patches - 1) * self.stride + self.patch_len

        accumulated = torch.zeros(B, padded_len, device=device)
        counts = torch.zeros(B, padded_len, device=device)

        for p in range(num_patches):
            start = p * self.stride
            end = start + plen
            accumulated[:, start:end] += patch_vals[:, p, :]
            counts[:, start:end] += 1.0

        counts = counts.clamp(min=1.0)
        reconstructed = accumulated / counts  # [B, padded_len]
        return reconstructed[:, :original_len]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: dict with keys
                "context"       [B, N_ctx, 8]
                "query"         [B, N_qry, 7]
                "context_mask"  [B, N_ctx]  (optional, bool)
                "query_mask"    [B, N_qry]  (optional, bool)

        Returns:
            dict with keys
                "v_norm_pred"   [B, N_qry]
                "query_mask"    [B, N_qry] or None
        """
        context = batch["context"]       # [B, N_ctx, 8]
        query = batch["query"]           # [B, N_qry, 7]
        context_mask = batch.get("context_mask")
        query_mask = batch.get("query_mask")

        B, N_ctx, _ = context.shape
        N_qry = query.shape[1]

        # ---- 1. Context patching + position encoding --------------------------
        ctx_patches, ctx_patch_mask, _ = self._pad_and_patch(context, context_mask)
        num_ctx_patches = ctx_patches.shape[1]

        ctx_embed = self.ctx_patch_proj(ctx_patches)          # [B, P_ctx, d_model]
        ctx_embed = ctx_embed + self.ctx_pos_encoding[:, :num_ctx_patches, :]

        # ---- 2. Self-attention encoder for context patches --------------------
        ctx_key_mask = ~ctx_patch_mask  # True = ignore in MHA

        for layer in self.encoder_layers:
            ctx_embed = layer(ctx_embed, key_padding_mask=ctx_key_mask)

        # ---- 3. Query patching + position encoding ----------------------------
        qry_patches, qry_patch_mask, _ = self._pad_and_patch(query, query_mask)
        num_qry_patches = qry_patches.shape[1]

        qry_embed = self.qry_patch_proj(qry_patches)          # [B, P_qry, d_model]
        qry_embed = qry_embed + self.qry_pos_encoding[:, :num_qry_patches, :]

        # ---- 4. Cross-attention: query → context ------------------------------
        qry_normed = self.cross_norm_q(qry_embed)
        ctx_normed = self.cross_norm_kv(ctx_embed)

        cross_out, _ = self.cross_attn(
            query=qry_normed,
            key=ctx_normed,
            value=ctx_normed,
            key_padding_mask=ctx_key_mask,
        )
        qry_embed = qry_embed + self.cross_dropout(cross_out)

        # ---- 5. patch_to_point: each patch → its constituent verticals --------
        patch_vals = self.patch_to_point(qry_embed)  # [B, P_qry, patch_len]

        # ---- 6. Un-patch: reconstruct per-vertical predictions ----------------
        v_raw = self._reconstruct_from_patches(patch_vals, N_qry)  # [B, N_qry]

        # ---- 7. Per-vertical refinement MLP -----------------------------------
        v_norm_pred = self.pred_head(v_raw.unsqueeze(-1)).squeeze(-1)  # [B, N_qry]

        # Numerical safety: clip extreme values and replace NaN
        v_norm_pred = torch.nan_to_num(v_norm_pred, nan=0.0, posinf=5.0, neginf=-5.0)
        v_norm_pred = torch.clamp(v_norm_pred, min=-10.0, max=10.0)

        return {
            "v_norm_pred": v_norm_pred,
            "query_mask": query_mask,
        }
