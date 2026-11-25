"""
xs_velformer.normalizer — 归一化的单一事实来源

修复研究期 predict.py 的泄漏抽象: 把 global_stats (v/Z/h_max/W 的 mean/std)
从 checkpoint 里取出, 封装为可独立检视、可持久化的 Normalizer。

归一化方案 (与训练期 data/preprocess.py 完全一致):
    v_norm   = (v   - v_mean)   / v_std         # 全局 StandardScaler
    Z_norm   = (Z   - Z_mean)   / Z_std
    h_max_norm = (h_max - h_max_mean) / h_max_std
    W_norm   = (W   - W_mean)   / W_std
    x_norm   = (x - x_min) / W                   # 逐断面左对齐, [0,1]
    d_norm   = h / h_max                         # 逐断面, [0,1]
    bank_norm, grad_norm 由 features.compute_bank_and_grad_features 计算
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class StatPair:
    """单个量的 mean/std"""
    mean: float
    std: float

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "StatPair":
        return cls(mean=float(d["mean"]), std=float(d["std"]))


@dataclass
class Normalizer:
    """流速 / 断面全局量的归一化器。

    bank_norm / grad_norm 是逐断面局部特征, 由 FeatureBuilder 基于 x_norm/d_norm 计算,
    不在此处; 这里只管全局 StandardScaler (v/Z/h_max/W)。
    """

    v: StatPair = field(default_factory=lambda: StatPair(0.0, 1.0))
    Z: StatPair = field(default_factory=lambda: StatPair(0.0, 1.0))
    h_max: StatPair = field(default_factory=lambda: StatPair(0.0, 1.0))
    W: StatPair = field(default_factory=lambda: StatPair(0.0, 1.0))

    # ---- 全局量 (v/Z/h_max/W) ----
    def normalize_velocity(self, v_raw: np.ndarray) -> np.ndarray:
        return (v_raw - self.v.mean) / self.v.std

    def denormalize_velocity(self, v_norm: np.ndarray) -> np.ndarray:
        return v_norm * self.v.std + self.v.mean

    def normalize_section_globals(self, Z: float, h_max: float, W: float) -> Dict[str, float]:
        """对水位 / 最大水深 / 河宽做全局标准化"""
        return {
            "Z_norm": (Z - self.Z.mean) / self.Z.std,
            "h_max_norm": (h_max - self.h_max.mean) / self.h_max.std,
            "W_norm": (W - self.W.mean) / self.W.std,
        }

    # 逐断面量 (x_norm / d_norm) 的归一化属于特征工程, 由 features.FeatureBuilder 持有,
    # 不放在此处, 避免两份公式漂移。

    # ---- 持久化 ----
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "v": self.v.to_dict(),
            "Z": self.Z.to_dict(),
            "h_max": self.h_max.to_dict(),
            "W": self.W.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Dict[str, float]]) -> "Normalizer":
        """从 {v:{mean,std}, Z:{...}, ...} 构建 (兼容旧 checkpoint 的 global_stats 格式)"""
        return cls(
            v=StatPair.from_dict(d["v"]),
            Z=StatPair.from_dict(d["Z"]),
            h_max=StatPair.from_dict(d["h_max"]),
            W=StatPair.from_dict(d["W"]),
        )

    def to_json_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, path: str) -> "Normalizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
