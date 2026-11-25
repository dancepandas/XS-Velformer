"""
xs_velformer.convert_legacy — 旧 checkpoint → 标准产物包 (一次性转换)

把训练期 train.py 保存的旧格式 checkpoint:
    {model_state_dict, optimizer_state_dict, model_config, train_config,
     global_stats, best_val_loss, artifacts}
拆分为标准产物包:
    artifacts/<name>/{model.pt, config.json, normalizer.json, manifest.json}

只读旧 checkpoint, 不修改它。用法:
    python -m xsvelformer.convert_legacy checkpoints/full.pt artifacts/xsvelformer-v1
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from typing import Optional

import torch

from ._config import ModelConfig

from .artifact import Manifest, write_artifact
from .normalizer import Normalizer

# 论文报告的主模型指标, 仅作追溯备注写进 manifest.notes;
# 不写入 manifest.metrics —— metrics 只放 checkpoint 派生量, 避免不同 checkpoint 误用论文值。
PAPER_METRICS_NOTE = "paper(full.pt): RMSE=0.102 m/s, MAE=0.042 m/s, NSE=0.978, KGE=0.969"


def convert(legacy_ckpt: str, out_dir: str, *, name: Optional[str] = None) -> str:
    """转换旧 checkpoint 为标准产物包。

    Args:
        legacy_ckpt: 旧 checkpoint 路径 (如 checkpoints/full.pt)
        out_dir:     输出目录; 若 name 给定则写到 out_dir/<name>/
        name:        产物包名 (如 'xsvelformer-v1')
    Returns:
        实际写入的产物包目录
    """
    if not os.path.isfile(legacy_ckpt):
        raise FileNotFoundError(f"旧 checkpoint 不存在: {legacy_ckpt}")

    ckpt = torch.load(legacy_ckpt, map_location="cpu", weights_only=False)

    state_dict = ckpt["model_state_dict"]
    cfg_dict = ckpt.get("model_config", {})
    global_stats = ckpt.get("global_stats", {})
    if not global_stats:
        raise ValueError("旧 checkpoint 缺 global_stats, 无法构建 Normalizer")

    model_cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if hasattr(ModelConfig, k)})
    normalizer = Normalizer.from_dict(global_stats)

    target = os.path.join(out_dir, name) if name else out_dir
    trained_on = datetime.datetime.fromtimestamp(os.path.getmtime(legacy_ckpt)).isoformat(timespec="seconds")

    metrics = {}
    bvl = ckpt.get("best_val_loss")
    if bvl is not None:
        metrics["best_val_loss"] = float(bvl)

    manifest = Manifest(
        source_checkpoint=os.path.abspath(legacy_ckpt),
        trained_on=trained_on,
        metrics=metrics,
        notes=(
            "Converted from legacy train.py checkpoint by xsvelformer.convert_legacy. "
            + PAPER_METRICS_NOTE
        ),
    )

    write_artifact(target, state_dict=state_dict, model_cfg=model_cfg,
                   normalizer=normalizer, manifest=manifest)

    print(f"[convert_legacy] {legacy_ckpt}")
    print(f"               -> {target}")
    print(f"  model: {model_cfg.d_model}d, ctx={model_cfg.context_encoder_layers}, "
          f"qry={model_cfg.query_encoder_layers}, cross={model_cfg.cross_attention_layers}")
    print(f"  normalizer: v~N({normalizer.v.mean:.3f},{normalizer.v.std:.3f}), "
          f"W~N({normalizer.W.mean:.1f},{normalizer.W.std:.1f})")
    return target


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="旧 checkpoint -> XS-VelFormer 标准产物包")
    p.add_argument("legacy_ckpt", help="旧 checkpoint 路径 (如 checkpoints/full.pt)")
    p.add_argument("out_dir", help="输出目录")
    p.add_argument("--name", default=None, help="产物包名 (如 xsvelformer-v1)")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    convert(a.legacy_ckpt, a.out_dir, name=a.name)
