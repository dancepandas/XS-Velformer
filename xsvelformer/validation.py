"""
xs_velformer.validation — 原始输入的校验与物理合理性检查

生产接口必须在进入模型前拦截非法输入, 避免静默错误。
所有检查抛出带明确信息的 ValueError。
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .schemas import QuerySpec, SectionData

# 业务约束阈值 (可按现场调整)
MIN_CONTEXT_POINTS = 3          # 至少需要的已知测点数
MAX_CONTEXT_POINTS = 2000       # 防止异常大输入
MIN_WIDTH = 1e-3                # m
MAX_WIDTH = 1e4                 # m (论文数据上限 404.7m, 留余量)
MIN_WATER_STAGE = -1e3          # m (海拔可为负, 留余量)
MAX_WATER_STAGE = 1e4
MIN_DEPTH = 0.0                 # 水深非负
MAX_VELOCITY = 50.0             # m/s, 物理上限保护


class ValidationError(ValueError):
    """输入校验错误"""


def _is_finite(x: float) -> bool:
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def validate_section(section: SectionData, *, require_known: bool = True) -> None:
    """完整校验一条 SectionData。

    Args:
        section:        原始断面数据
        require_known:  是否要求至少有 MIN_CONTEXT_POINTS 个已知流速点
                         (impute_missing 需要; reconstruct/densify 也需要 context)
    """
    if section.K == 0:
        raise ValidationError("断面测点数为 0")

    if section.K > MAX_CONTEXT_POINTS:
        raise ValidationError(f"断面测点数 {section.K} 超过上限 {MAX_CONTEXT_POINTS}")

    # width
    if not _is_finite(section.width) or section.width <= 0:
        raise ValidationError(f"width 非法: {section.width}")
    if section.width < MIN_WIDTH or section.width > MAX_WIDTH:
        raise ValidationError(f"width {section.width} 越界 [{MIN_WIDTH}, {MAX_WIDTH}]")

    # water_stage
    if not _is_finite(section.water_stage):
        raise ValidationError(f"water_stage 非法: {section.water_stage}")
    if not (MIN_WATER_STAGE <= section.water_stage <= MAX_WATER_STAGE):
        raise ValidationError(f"water_stage {section.water_stage} 越界")

    # 逐点检查
    _check_point_sequence(section.distances, section.depths, section.velocities, section.width)

    # 已知点数
    if require_known:
        n_known = sum(section.known_mask())
        if n_known < MIN_CONTEXT_POINTS:
            raise ValidationError(
                f"已知流速点数 {n_known} < 下限 {MIN_CONTEXT_POINTS}, 无法可靠插补"
            )


def validate_query(query: QuerySpec, section: SectionData) -> None:
    """校验 QuerySpec: 位置有限、非空, 且落在断面实测横向范围内。

    query 位置必须落在 [min(section.distances), max(section.distances)] 内,
    否则 x_norm 会落到 [0,1] 之外 → bank_norm 被静默裁剪 → 分布外输入。
    超出实测范围的预测 (外推) 不在本接口支持范围内。
    """
    if query.M == 0:
        raise ValidationError("QuerySpec 为空 (无预测位置)")
    sec_x = section.distances
    x_lo, x_hi = min(sec_x), max(sec_x)
    eps = 1e-6
    for i, x in enumerate(query.distances):
        if not _is_finite(x):
            raise ValidationError(f"query 第 {i} 个起点距非法: {x}")
        if x < x_lo - eps or x > x_hi + eps:
            raise ValidationError(
                f"query 第 {i} 个起点距 {x} 越出断面实测范围 [{x_lo}, {x_hi}]"
            )
    if query.depths is not None:
        for i, h in enumerate(query.depths):
            if not _is_finite(h):
                raise ValidationError(f"query 第 {i} 个水深非法: {h}")
            if h < MIN_DEPTH:
                raise ValidationError(f"query 第 {i} 个水深 {h} 为负")


def _check_point_sequence(
    distances: List[float],
    depths: List[float],
    velocities: List,
    width: float,
) -> None:
    """逐点 + 序列一致性检查"""
    x_min = min(distances)
    x_max = max(distances)

    for i, (x, h) in enumerate(zip(distances, depths)):
        if not _is_finite(x):
            raise ValidationError(f"第 {i} 个起点距非法: {x}")
        if not _is_finite(h):
            raise ValidationError(f"第 {i} 个水深非法: {h}")
        if h < MIN_DEPTH:
            raise ValidationError(f"第 {i} 个水深 {h} 为负")
        if x < 0 or x > width + 1e-6:
            raise ValidationError(f"第 {i} 个起点距 {x} 越界 [0, width={width}]")

    # 升序非降
    for i in range(1, len(distances)):
        if distances[i] < distances[i - 1] - 1e-9:
            raise ValidationError(
                f"起点距必须升序非降, 第 {i-1}({distances[i-1]}) > 第 {i}({distances[i]})"
            )

    # 已知流速合法性
    for i, v in enumerate(velocities):
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        if isinstance(v, float) and math.isinf(v):
            raise ValidationError(f"第 {i} 个流速为 inf")
        if v < 0:
            raise ValidationError(f"第 {i} 个流速 {v} 为负")
        if v > MAX_VELOCITY:
            raise ValidationError(f"第 {i} 个流速 {v} 超过物理上限 {MAX_VELOCITY}")

    # 断面几何一致性: 最大起点距 - 最小起点距 应约等于 width (容许测量误差)
    if (x_max - x_min) > width + 1e-6:
        raise ValidationError(
            f"起点距跨度 ({x_max - x_min}) 超过 width ({width})"
        )
    # 横向跨度必须 > 0: 全部起点距相同 (退化断面) 会使 x_norm 全零、grad 退化
    if (x_max - x_min) <= 0:
        raise ValidationError(
            "断面无横向跨度 (所有起点距相同), 无法归一化"
        )
