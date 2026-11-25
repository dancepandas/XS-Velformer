"""
xs_velformer.artifact — 版本化产物包的加载与校验

标准产物包结构:
    artifacts/xsvelformer-v1/
    ├── model.pt           # state_dict (torch)
    ├── config.json        # ModelConfig 架构参数
    ├── normalizer.json    # global_stats (v/Z/h_max/W 的 mean/std)
    └── manifest.json      # 版本 / 来源 / 指标 / IO schema

加载时校验四件套齐备且版本兼容。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ._config import ModelConfig
from .normalizer import Normalizer

ARTIFACT_VERSION = "1.0"

# 产物包内文件名约定
F_MODEL = "model.pt"
F_CONFIG = "config.json"
F_NORMALIZER = "normalizer.json"
F_MANIFEST = "manifest.json"

REQUIRED_FILES = [F_MODEL, F_CONFIG, F_NORMALIZER, F_MANIFEST]

# 输入/输出 schema (写进 manifest, 供下游系统对齐)
IO_SCHEMA = {
    "input": {
        "section": {
            "distances": "List[float], m, ascending in [0, width]",
            "depths": "List[float], m, >=0",
            "velocities": "List[Optional[float]], m/s; None=missing",
            "water_stage": "float, m",
            "width": "float, m",
        },
        "query": "QuerySpec (distances, optional depths), m",
    },
    "output": {
        "velocities": "List[float], m/s",
        "is_predicted": "List[bool]",
    },
}


@dataclass
class Manifest:
    version: str = ARTIFACT_VERSION
    model_name: str = "XS-VelFormer"
    source_checkpoint: Optional[str] = None
    trained_on: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "artifact_version": self.version,
            "model_name": self.model_name,
            "source_checkpoint": self.source_checkpoint,
            "trained_on": self.trained_on,
            "metrics": self.metrics,
            "notes": self.notes,
            "io_schema": IO_SCHEMA,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Manifest":
        return cls(
            version=d.get("artifact_version", ARTIFACT_VERSION),
            model_name=d.get("model_name", "XS-VelFormer"),
            source_checkpoint=d.get("source_checkpoint"),
            trained_on=d.get("trained_on"),
            metrics=d.get("metrics", {}),
            notes=d.get("notes"),
        )


class ArtifactError(RuntimeError):
    """产物包缺失或损坏"""


def _paths_in(dirpath: str) -> Dict[str, str]:
    return {name: os.path.join(dirpath, name) for name in REQUIRED_FILES}


def write_artifact(
    dirpath: str,
    *,
    state_dict: Dict[str, torch.Tensor],
    model_cfg: ModelConfig,
    normalizer: Normalizer,
    manifest: Manifest,
) -> None:
    """把一组组件写成标准产物包"""
    os.makedirs(dirpath, exist_ok=True)
    p = _paths_in(dirpath)
    torch.save(state_dict, p[F_MODEL])
    with open(p[F_CONFIG], "w", encoding="utf-8") as f:
        json.dump(model_cfg.__dict__, f, indent=2, ensure_ascii=False)
    normalizer.to_json_file(p[F_NORMALIZER])
    with open(p[F_MANIFEST], "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)


def load_artifact(dirpath: str) -> Dict:
    """加载并校验标准产物包。

    Returns:
        dict: {state_dict, model_cfg, normalizer, manifest, dirpath}
    """
    if not os.path.isdir(dirpath):
        raise ArtifactError(f"产物目录不存在: {dirpath}")

    p = _paths_in(dirpath)
    for name, path in p.items():
        if not os.path.isfile(path):
            raise ArtifactError(f"产物包缺文件: {name} ({path})")

    state_dict = torch.load(p[F_MODEL], map_location="cpu", weights_only=False)
    with open(p[F_CONFIG], "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    model_cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if hasattr(ModelConfig, k)})
    normalizer = Normalizer.from_json_file(p[F_NORMALIZER])
    with open(p[F_MANIFEST], "r", encoding="utf-8") as f:
        manifest = Manifest.from_dict(json.load(f))

    return {
        "state_dict": state_dict,
        "model_cfg": model_cfg,
        "normalizer": normalizer,
        "manifest": manifest,
        "dirpath": dirpath,
    }
