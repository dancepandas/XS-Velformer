"""随机推理适配器的评估可复现测试 (协议 §3)

同 checkpoint、同评估配置下重复 predict 结果必须一致:
随机推理适配器 (Informer/Diffusion/GAN) 在固定评估随机流
(BaseModelAdapter._fixed_rng, eval_seed) 下重放, 评估入口落盘 eval_seed。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baseline.informer.model import InformerBaseline
from config import ModelConfig
from evaluation.model_adapters import InformerAdapter


def _make_adapter(eval_seed: int = 20261112) -> InformerAdapter:
    ad = InformerAdapter(device="cpu", eval_seed=eval_seed)
    cfg = ModelConfig(d_model=64, ffn_dim=128)
    ad.model = InformerBaseline(cfg).to(ad.device).eval()
    return ad


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ctx_mask = torch.zeros(2, 10, dtype=torch.bool)
    ctx_mask[:, :8] = True
    return {
        "context": torch.randn(2, 10, 8, generator=g),
        "query": torch.randn(2, 5, 7, generator=g),
        "context_mask": ctx_mask,
        "query_mask": torch.ones(2, 5, dtype=torch.bool),
        "target": torch.randn(2, 5, generator=g),
    }


def test_informer_adapter_repeat_eval_identical():
    ad = _make_adapter()
    batch = _batch()
    y1 = ad.predict(batch)
    y2 = ad.predict(batch)
    assert torch.equal(y1, y2), "同权重同配置重复评估结果不一致"


def test_adapter_eval_seed_changes_sampling():
    y1 = _make_adapter(eval_seed=1).predict(_batch())
    y2 = _make_adapter(eval_seed=2).predict(_batch())
    assert not torch.equal(y1, y2), "不同评估种子输出相同, 采样未生效"


def test_stochastic_adapters_carry_eval_seed():
    # 三个随机推理适配器都必须携带评估种子与固定随机流入口 (落盘依据)
    from evaluation.model_adapters import DiffusionMeanAdapter, GANMeanAdapter
    for ad in (InformerAdapter(), DiffusionMeanAdapter(), GANMeanAdapter()):
        assert isinstance(ad.eval_seed, int)
        with ad._fixed_rng():
            torch.rand(3)
        # 上下文退出后 RNG 状态恢复 (与进入前一致)
        s = torch.get_rng_state()
        with ad._fixed_rng():
            torch.rand(3)
        assert torch.equal(torch.get_rng_state(), s)
