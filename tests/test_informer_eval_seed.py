"""随机推理的验证随机流固定 (fixed_torch_rng) 行为测试

协议 §3: Informer ProbSparse 采样等随机推理不承诺逐样本批不变, 但同一权重
下验证分数必须可复现 —— 由"固定评估种子 + 退出恢复 RNG 状态"保证。这里守护:
1. 固定种子下同权重两次前向逐位相等;
2. 上下文退出后全局 RNG 状态恢复;
3. 不同种子下输出确实不同 (证明采样在起作用, 确定性不是平凡成立)。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baseline.informer.model import ProbSparseAttention
from config import ModelConfig
from train import fixed_torch_rng


def _make_attn() -> ProbSparseAttention:
    cfg = ModelConfig(d_model=64, ffn_dim=128, num_heads=4)
    attn = ProbSparseAttention(cfg, factor=5)
    attn.eval()
    return attn


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 25, 64, generator=g)
    mask = torch.zeros(2, 25, dtype=torch.bool)
    mask[0, :20] = True
    mask[1, :12] = True
    return x, mask


def test_fixed_seed_replay_is_bitwise_deterministic():
    attn = _make_attn()
    x, mask = _inputs()
    with torch.no_grad(), fixed_torch_rng(123):
        y1 = attn(x, mask)
    with torch.no_grad(), fixed_torch_rng(123):
        y2 = attn(x, mask)
    assert torch.equal(y1, y2), "同种子重放结果不一致"


def test_rng_state_restored_after_context():
    attn = _make_attn()
    x, mask = _inputs()
    state_before = torch.get_rng_state()
    with torch.no_grad(), fixed_torch_rng(7):
        attn(x, mask)
    assert torch.equal(torch.get_rng_state(), state_before), \
        "fixed_torch_rng 退出后未恢复全局 RNG 状态"


def test_different_seeds_give_different_outputs():
    attn = _make_attn()
    x, mask = _inputs()
    with torch.no_grad(), fixed_torch_rng(1):
        y1 = attn(x, mask)
    with torch.no_grad(), fixed_torch_rng(2):
        y2 = attn(x, mask)
    assert not torch.equal(y1, y2), \
        "不同种子输出相同 —— 随机采样未生效, 确定性测试平凡通过"


def test_all_padding_sample_rejected():
    attn = _make_attn()
    x, _ = _inputs()
    mask = torch.zeros(2, 25, dtype=torch.bool)
    try:
        with torch.no_grad(), fixed_torch_rng(1):
            attn(x, mask)
    except ValueError:
        return
    raise AssertionError("全 padding 样本未被拒绝")
