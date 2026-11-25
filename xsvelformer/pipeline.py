"""
xs_velformer.pipeline — 生产推理编排器

XSVelFormerPipeline 把 "原始物理量 → 流速预测 (m/s)" 全流程封装:
    校验 → 归一化 → 建特征 → 模型前向 → 反归一化 → 物理后处理 → 结果

调用方只接触 SectionData (m / m/s) 和 PredictionResult (m/s),
所有归一化 / 特征工程 / bank-grad / padding 全部内部完成。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .artifact import load_artifact
from .features import SectionFeatures, build_section_features, pad_batch
from .model import XSVelFormerModel
from .normalizer import Normalizer
from .presets import ScenarioMixin
from .schemas import PredictionResult, QuerySpec, SectionData
from .validation import ValidationError, validate_query, validate_section


class XSVelFormerPipeline(ScenarioMixin):
    """XS-VelFormer 生产推理 pipeline。

    通过 from_artifact() 加载版本化产物包构建。
    内置三种业务场景方法: impute_missing / reconstruct / densify (见 presets.ScenarioMixin)。
    """

    def __init__(
        self,
        model: XSVelFormerModel,
        normalizer: Normalizer,
        *,
        device: str = "cpu",
        clip_negative: bool = True,
    ):
        self.model = model.to(device)
        self.normalizer = normalizer
        self.device = device
        self.clip_negative = clip_negative

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_artifact(
        cls,
        dirpath: str,
        *,
        device: Optional[str] = None,
        clip_negative: bool = True,
    ) -> "XSVelFormerPipeline":
        """从标准产物包加载 pipeline。

        Args:
            dirpath:      产物包目录 (含 model.pt / config.json / normalizer.json / manifest.json)
            device:       'cpu' / 'cuda'; None 自动选择
            clip_negative: 是否把负预测流速裁剪到 0 (物理兜底)
        """
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        art = load_artifact(dirpath)
        model = XSVelFormerModel.from_state_dict(
            art["state_dict"], cfg=art["model_cfg"], device=device,
        )
        return cls(
            model=model,
            normalizer=art["normalizer"],
            device=device,
            clip_negative=clip_negative,
        )

    # ------------------------------------------------------------------ 单条
    def predict(self, section: SectionData, query: QuerySpec) -> PredictionResult:
        """对单条断面的一组 query 位置预测流速。"""
        return self.predict_many([section], [query])[0]

    # ------------------------------------------------------------------ 批量
    def predict_many(
        self,
        sections: Sequence[SectionData],
        queries: Sequence[QuerySpec],
    ) -> List[PredictionResult]:
        """批量推理 (变长断面内部 padding, 单次前向)。

        Args:
            sections: 断面列表
            queries:  与 sections 等长的 query 规格列表
        Returns:
            与输入等长的 PredictionResult 列表
        """
        if len(sections) != len(queries):
            raise ValueError(f"sections ({len(sections)}) 与 queries ({len(queries)}) 长度不一致")
        if len(sections) == 0:
            return []

        # 1. 校验 + 建特征
        feats: List[SectionFeatures] = []
        for sec, qry in zip(sections, queries):
            validate_section(sec, require_known=True)
            validate_query(qry, sec)
            feats.append(build_section_features(sec, qry, self.normalizer))

        # 2. padding 成一个 batch (与训练 collate_fn 一致)
        batch = pad_batch(feats)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # 3. 前向 → 归一化流速 [B, N_qry_padded]
        v_norm = self.model.forward_normalized(batch)

        # 4. 逐条切片 + 反归一化 + 后处理
        results: List[PredictionResult] = []
        for i, (sec, qry, f) in enumerate(zip(sections, queries, feats)):
            valid = batch["query_mask"][i].cpu().numpy()[: f.n_query]
            v_norm_i = v_norm[i].cpu().numpy()[: f.n_query][valid]
            v_phys = self.normalizer.denormalize_velocity(v_norm_i)

            diagnostics: Dict = {}
            if self.clip_negative:
                n_neg = int(np.sum(v_phys < 0))
                if n_neg > 0:
                    diagnostics["negative_clipped"] = n_neg
                v_phys = np.clip(v_phys, 0.0, None)

            results.append(self._build_result(f, v_phys, diagnostics))
        return results

    # ------------------------------------------------------------------ 结果组装
    @staticmethod
    def _build_result(f: SectionFeatures, v_phys: np.ndarray, diagnostics: Dict) -> PredictionResult:
        """用 FeatureBuilder 已算好的 query 水深, 避免二次插值"""
        depths = [float(d) for d in f.query_depths]
        return PredictionResult(
            distances=[float(x) for x in f.query_positions],
            depths=depths,
            velocities=[float(v) for v in v_phys],
            is_predicted=[True] * len(v_phys),
            diagnostics=diagnostics,
        )

    def __repr__(self) -> str:
        return f"XSVelFormerPipeline(model={self.model!r}, device='{self.device}', clip_negative={self.clip_negative})"
