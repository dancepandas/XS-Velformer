"""
xs_velformer.model — 模型别名封装

复用包内 _nn.HydroQueryTransformer (与训练期 models/ 逐位一致), 零拷贝、零重训练。
提供面向生产的薄封装:
    - 从 ModelConfig + state_dict 重建
    - 批量前向 (输入 batch dict, 输出 [B, N_qry] 归一化流速)

不改动原训练仓库代码, 生产代码统一使用 XSVelFormerModel 这个产品名。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ._config import ModelConfig
from ._nn import HydroQueryTransformer


class XSVelFormerModel:
    """XS-VelFormer 推理模型 (HydroQueryTransformer 的生产别名)。

    这是一个轻量包装器, 内部直接持有 HydroQueryTransformer 实例;
    不复制参数、不重训练, 完全复用已训练权重。
    """

    def __init__(self, model: HydroQueryTransformer, cfg: ModelConfig):
        self._model = model
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: Optional[ModelConfig] = None, device: str = "cpu") -> "XSVelFormerModel":
        """从配置构建空模型 (未加载权重)"""
        cfg = cfg or ModelConfig()
        model = HydroQueryTransformer(cfg).to(device)
        model.eval()
        return cls(model, cfg)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        cfg: Optional[ModelConfig] = None,
        device: str = "cpu",
    ) -> "XSVelFormerModel":
        """从 state_dict + 配置重建推理模型"""
        cfg = cfg or ModelConfig()
        model = HydroQueryTransformer(cfg).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return cls(model, cfg)

    @property
    def torch_module(self) -> HydroQueryTransformer:
        """底层 nn.Module (供高级用途, 如导出 ONNX)"""
        return self._model

    def to(self, device: str) -> "XSVelFormerModel":
        self._model.to(device)
        return self

    def forward_normalized(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向, 返回归一化流速 [B, N_qry]。

        batch = {context, context_mask, query, query_mask}
        """
        with torch.no_grad():
            out = self._model(batch)
        return out["v_norm_pred"]   # [B, N_qry]

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self._model.parameters())
        return (
            f"XSVelFormerModel(d_model={self.cfg.d_model}, "
            f"cross_layers={self.cfg.cross_attention_layers}, "
            f"params={n_params/1e6:.2f}M)"
        )
