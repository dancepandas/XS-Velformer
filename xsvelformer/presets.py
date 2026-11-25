"""
xs_velformer.presets — 三种业务场景的便捷方法 (原始输入版)

挂载到 XSVelFormerPipeline 上, 调用方直接:
    pipe.impute_missing(section)
    pipe.reconstruct(section, n_lines=100)
    pipe.densify(section, n_target=200)

这些方法只负责把业务意图翻译成 QuerySpec, 然后委托给 pipeline.predict。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .schemas import PredictionResult, QuerySpec, SectionData, make_query_from_missing


class ScenarioMixin:
    """业务场景便捷方法 (由 XSVelFormerPipeline 继承)。

    依赖宿主类实现: predict(section, query) -> PredictionResult
    """

    # ------------------------------------------------------------------ 缺失插补
    def impute_missing(self, section: SectionData) -> PredictionResult:
        """插补断面中缺测的流速点 (SectionData.velocities 为 None/NaN 的位置)。

        Returns:
            仅含被插补点的 PredictionResult (is_predicted 全为 True)。
        """
        query = make_query_from_missing(section)
        return self.predict(section, query)   # type: ignore[attr-defined]

    # ------------------------------------------------------------------ 标准断面重构
    def reconstruct(self, section: SectionData, n_lines: int = 100) -> PredictionResult:
        """在整条断面物理坐标 [x_min, x_max] 上等间距生成 n_lines 条测线, 重构流速剖面。

        网格跨整条断面 (含缺测但真实的位置), 保证几何标定与训练一致。
        query 水深由已知点水深插值得到 (越界处 np.interp 钳到端点)。
        """
        if n_lines < 2:
            raise ValueError("n_lines 必须 >= 2")
        all_x = np.asarray(section.distances, dtype=np.float64)
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        targets = np.linspace(x_min, x_max, n_lines)
        query = QuerySpec(distances=[float(t) for t in targets], depths=None)
        return self.predict(section, query)   # type: ignore[attr-defined]

    # ------------------------------------------------------------------ 测线加密
    def densify(self, section: SectionData, n_target: int = 200) -> PredictionResult:
        """把断面加密到 n_target 条等间距测线 (与 reconstruct 同实现, 语义为加密)。"""
        return self.reconstruct(section, n_lines=n_target)   # type: ignore[attr-defined]
