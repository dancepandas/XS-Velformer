"""跨站 × 跨时双重留出划分 (split_by_station_and_time) 的不变量测试

语料约定: sections.pkl 中同一测站的断面已按测流时间排序 (SectionSample
没有时间字段, 输入顺序是唯一时间信息) —— 与 split_by_time 的约定一致。
"""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess import SectionPreprocessor, SectionSample


def make_sample(station: str, K: int = 12) -> SectionSample:
    return SectionSample(
        station_id=station,
        x_raw=np.linspace(0, 10, K),
        h_raw=np.ones(K) * 2.0,
        v_raw=np.ones(K) * 0.5,
        Z=1.0,
        x_norm=np.linspace(0, 1, K),
        d_norm=np.ones(K) * 0.5,
        bank_norm=np.abs(np.linspace(-1, 1, K)),
        grad_norm=np.zeros(K),
        v_norm=np.ones(K) * 0.5,
        h_max=2.0,
        W=10.0,
        x_min=0.0,
    )


def build_corpus() -> tuple[list, dict]:
    """站间交错、站内时间序 (合法输入); seq 记录每条样本的全局时间序号"""
    samples, seq = [], {}
    g = 0
    caps = {"A": 120, "B": 100, "C": 80, "D": 60,
            "E": 40, "F": 30, "G": 20, "H": 10}
    for i in range(max(caps.values())):
        for st, cap in caps.items():
            if i < cap:
                s = make_sample(st)
                samples.append(s)
                seq[id(s)] = (st, g)
                g += 1
    return samples, seq


def test_dual_holdout_invariants():
    samples, seq = build_corpus()
    p = SectionPreprocessor()
    tr, va, te = p.split_by_station_and_time(
        samples, train_ratio=0.7, val_ratio=0.15, seed=42)

    tr_st = {s.station_id for s in tr}
    va_st = {s.station_id for s in va}
    te_st = {s.station_id for s in te}
    holdout = te_st - tr_st

    # 保留站的 train/val 同站集合; 留出站整体只出现在 test
    assert tr_st == va_st
    assert holdout, "必须存在站点留出"
    assert not (holdout & tr_st) and not (holdout & va_st)
    # 双重留出的固有代价: test 占比 ~28%
    assert 0.25 <= len(te) / len(samples) <= 0.32

    # 时间留出: 保留站的 test 样本必须严格晚于该站全部 train 样本
    max_train_seq = defaultdict(int)
    for s in tr:
        st, g = seq[id(s)]
        max_train_seq[st] = max(max_train_seq[st], g)
    for s in te:
        st, g = seq[id(s)]
        if st in tr_st:
            assert g > max_train_seq[st], "保留站的 test 块混入了早期断面"

    # 恰好划分一次
    assert len(tr) + len(va) + len(te) == len(samples)


def test_ratio_robustness_small_stations():
    """小站 (n<10) 也必须产出非空 train/val 且不崩溃"""
    samples, seq = build_corpus()
    p = SectionPreprocessor()
    tr, va, te = p.split_by_station_and_time(
        samples, train_ratio=0.6, val_ratio=0.2, seed=7)
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0
    assert len(tr) + len(va) + len(te) == len(samples)
