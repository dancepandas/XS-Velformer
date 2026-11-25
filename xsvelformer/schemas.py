"""
xs_velformer.schemas — 原始物理量的输入/输出数据契约

本模块定义生产推理接口的全部数据结构, 全部使用原始物理单位 (m, m/s),
不依赖 torch / numpy 的张量语义, 调用方只需面向业务数据。

约定:
    - distances: 起点距 (m), 升序非降, 落在 [0, width]
    - depths:    测速线水深 (m), 非负
    - velocities: 已知流速 (m/s); 用 None / NaN 表示该测点缺测待插补
    - water_stage: 水位 (m)
    - width:    水面宽度 (m)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

Number = Union[int, float]


@dataclass
class SectionData:
    """单条河流断面的原始观测数据 (物理单位)。

    Attributes:
        distances:   [K] 起点距 (m), 升序非降, 范围 [0, width]
        depths:      [K] 水深 (m), 非负
        velocities:  [K] 已知流速 (m/s); None/NaN 表示该点缺测, 由模型插补
        water_stage: 水位 (m)
        width:       水面宽度 (m), 必须 > 0
        station_id:  可选测站标识, 仅用于追溯, 不参与计算
    """

    distances: List[float]
    depths: List[float]
    velocities: List[Optional[float]]
    water_stage: float
    width: float
    station_id: Optional[str] = None

    def __post_init__(self):
        n = len(self.distances)
        if not (len(self.depths) == n == len(self.velocities)):
            raise ValueError(
                f"distances/depths/velocities 长度必须一致, got "
                f"{len(self.distances)}/{len(self.depths)}/{len(self.velocities)}"
            )

    @property
    def K(self) -> int:
        """测速线数量"""
        return len(self.distances)

    def known_mask(self) -> List[bool]:
        """哪些点是已知流速 (非 None 且非 NaN)"""
        out = []
        for v in self.velocities:
            out.append(v is not None and not (isinstance(v, float) and math.isnan(v)))
        return out

    def known_indices(self) -> List[int]:
        """已知流速点的下标"""
        return [i for i, m in enumerate(self.known_mask()) if m]

    def known_velocities(self) -> List[float]:
        """已知流速值 (m/s), 顺序与 known_indices 对应"""
        return [self.velocities[i] for i in self.known_indices()]


@dataclass
class QuerySpec:
    """目标预测位置的规格。

    distances/depths 给定要预测流速的 query 位置 (物理单位)。
    若 depths 省略, 将由 context 水深线性插值得到。
    """

    distances: List[float]
    depths: Optional[List[float]] = None

    def __post_init__(self):
        if self.depths is not None and len(self.depths) != len(self.distances):
            raise ValueError("QuerySpec.distances 与 depths 长度必须一致")

    @property
    def M(self) -> int:
        return len(self.distances)


@dataclass
class PredictionResult:
    """单条断面的预测结果 (物理单位)。

    Attributes:
        distances:    预测位置的起点距 (m)
        depths:       预测位置的水深 (m)
        velocities:   预测流速 (m/s)
        is_predicted: 哪些点是模型预测的 (True) vs 直接回传的已知点 (False)
        diagnostics:  诊断信息, 如裁剪掉的负值数量等
    """

    distances: List[float]
    depths: List[float]
    velocities: List[float]
    is_predicted: List[bool]
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_predicted(self) -> int:
        return sum(self.is_predicted)

    def to_dict(self) -> dict:
        """序列化为 JSON 友好的 dict"""
        return {
            "distances": list(self.distances),
            "depths": list(self.depths),
            "velocities": list(self.velocities),
            "is_predicted": list(self.is_predicted),
            "diagnostics": dict(self.diagnostics),
        }


def make_query_from_missing(section: SectionData) -> QuerySpec:
    """由 SectionData 中缺测点 (velocity is None/NaN) 构建 QuerySpec。"""
    mask = section.known_mask()
    miss_idx = [i for i, m in enumerate(mask) if not m]
    if not miss_idx:
        raise ValueError("断面无缺测点, 无法构建 impute_missing 的 query")
    distances = [section.distances[i] for i in miss_idx]
    depths = [section.depths[i] for i in miss_idx]
    return QuerySpec(distances=distances, depths=depths)
