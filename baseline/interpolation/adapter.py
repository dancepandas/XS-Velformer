"""
传统插值基线适配器

本文件实现 InterpolationAdapter 基类和四个具体适配器,
统一传统插值方法与深度模型的评估接口。

适配器从 batch 的 context/query 张量中提取所需字段:
    ctx_x = context[:, :, 0]   # x_norm
    ctx_d = context[:, :, 1]   # d_norm
    ctx_v = context[:, :, 4]   # v_norm
    qry_x = query[:, :, 0]     # x_norm
    qry_d = query[:, :, 1]     # d_norm

Context 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
Query   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]
"""

import torch
from typing import Dict, Optional
from abc import ABC, abstractmethod

from baseline.interpolation.methods import nearest_x, linear_x, idw_x, idw_xd, cubic_spline, rbf_xd


class InterpolationAdapter(ABC):
    """传统插值基类

    不需要 checkpoint, load() 为空操作。
    子类实现 interpolate() 方法。

    Attributes:
        requires_checkpoint: 是否需要 checkpoint, 传统方法始终为 False
        device:              计算设备
    """

    requires_checkpoint: bool = False

    def __init__(self, device: str = "cpu"):
        self.device = device

    def load(self, checkpoint_path: str):
        pass

    @abstractmethod
    def interpolate(
        self,
        ctx_x: torch.Tensor,
        ctx_d: torch.Tensor,
        ctx_v: torch.Tensor,
        qry_x: torch.Tensor,
        qry_d: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        query_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._batch_to_device(batch)
        ctx_x = batch["context"][:, :, 0]
        ctx_d = batch["context"][:, :, 1]
        ctx_v = batch["context"][:, :, 4]
        qry_x = batch["query"][:, :, 0]
        qry_d = batch["query"][:, :, 1]
        context_mask = batch.get("context_mask")
        query_mask = batch.get("query_mask")
        return self.interpolate(ctx_x, ctx_d, ctx_v, qry_x, qry_d,
                                context_mask, query_mask)

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class NearestXAdapter(InterpolationAdapter):
    """Nearest-X: 按横向距离最近邻"""

    def interpolate(self, ctx_x, ctx_d, ctx_v, qry_x, qry_d,
                    context_mask, query_mask):
        return nearest_x(ctx_x, ctx_v, qry_x, context_mask)


class LinearXAdapter(InterpolationAdapter):
    """Linear-X: 基于 x_norm 的一维线性插值"""

    def interpolate(self, ctx_x, ctx_d, ctx_v, qry_x, qry_d,
                    context_mask, query_mask):
        return linear_x(ctx_x, ctx_v, qry_x, context_mask)


class IDWXAdapter(InterpolationAdapter):
    """IDW-X: 按横向距离反距离加权 (p=2)"""

    def interpolate(self, ctx_x, ctx_d, ctx_v, qry_x, qry_d,
                    context_mask, query_mask):
        return idw_x(ctx_x, ctx_v, qry_x, context_mask, p=2.0)


class IDWXDAdapter(InterpolationAdapter):
    """IDW-XD: 在 [x_norm, d_norm] 二维空间反距离加权 (p=2)"""

    def interpolate(self, ctx_x, ctx_d, ctx_v, qry_x, qry_d,
                    context_mask, query_mask):
        return idw_xd(ctx_x, ctx_d, ctx_v, qry_x, qry_d,
                       context_mask, p=2.0)


class CubicSplineAdapter(InterpolationAdapter):
    """Cubic-Spline: 基于 x_norm 的三次样条 (R1-2 additional interpolation)"""

    def interpolate(self, ctx_x, ctx_d, ctx_v, qry_x, qry_d,
                    context_mask, query_mask):
        return cubic_spline(ctx_x, ctx_v, qry_x, context_mask)


class RBFXDAdapter(InterpolationAdapter):
    """RBF-XD: [x_norm, d_norm] 二维薄板样条 RBF (R1-2 additional interpolation)"""

    def interpolate(self, ctx_x, ctx_d, ctx_v, qry_x, qry_d,
                    context_mask, query_mask):
        return rbf_xd(ctx_x, ctx_d, ctx_v, qry_x, qry_d, context_mask)
