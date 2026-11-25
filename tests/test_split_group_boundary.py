"""split_by_station_and_time 的测次分组不变量测试

同一测站同一时刻 (同一测次) 的全部记录必须落入同一集合:
逐记录切分会让同测次断面跨越 train/val/test 边界泄漏 —— 这里守护分组语义。
同时验证 require_time 对旧语料回退的开关行为。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess import SectionPreprocessor, SectionSample


def make_sample(station: str, time: object, K: int = 12) -> SectionSample:
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
        time=time,
    )


def _times_in(samples):
    return {s.time for s in samples}


def test_same_timestamp_never_straddles_sets():
    # 6 个测次, 每测次 3 条记录; 逐记录切分时边界测次必被劈开, 分组切分则不许
    stamps = [f"2025-0{m}-01 08:00:00" for m in range(1, 7)]
    samples = [make_sample("st", t) for t in stamps for _ in range(3)]

    pre = SectionPreprocessor()
    train, val, test = pre.split_by_station_and_time(
        samples, train_ratio=0.5, val_ratio=0.2)

    t_tr, t_va, t_te = _times_in(train), _times_in(val), _times_in(test)
    assert not (t_tr & t_va), f"同一测次同时出现在 train/val: {t_tr & t_va}"
    assert not (t_tr & t_te), f"同一测次同时出现在 train/test: {t_tr & t_te}"
    assert not (t_va & t_te), f"同一测次同时出现在 val/test: {t_va & t_te}"
    # 18 条记录全部分配
    assert len(train) + len(val) + len(test) == len(samples)
    # 同测次的 3 条记录不被拆散
    from collections import Counter
    for subset, name in ((train, "train"), (val, "val"), (test, "test")):
        c = Counter(s.time for s in subset)
        assert set(c.values()) <= {3}, f"{name} 内同测次被拆散: {c}"


def test_equivalent_utc_offsets_group_together():
    # 同一时刻的不同 offset 写法必须视为同一测次组 (08:00+08:00 == 00:00Z)
    samples = [
        make_sample("st", "2025-01-01T00:00:00Z"),
        make_sample("st", "2025-01-01T08:00:00+08:00"),
        make_sample("st", "2025-02-01T00:00:00Z"),
        make_sample("st", "2025-03-01T00:00:00Z"),
    ]
    pre = SectionPreprocessor()
    train, val, test = pre.split_by_station_and_time(
        samples, train_ratio=0.5, val_ratio=0.25)

    all_sets = [set(_times_in(x)) for x in (train, val, test)]
    pair = {"2025-01-01T00:00:00Z", "2025-01-01T08:00:00+08:00"}
    # 这两条必须同组 (同集合), 且两个时间戳都不与对方所在集合的其他元素分隔
    together = any(pair <= s for s in all_sets)
    assert together, f"等价 UTC 时刻未成组: {all_sets}"


def test_require_time_rejects_all_missing():
    samples = [make_sample("st", None) for _ in range(6)]
    pre = SectionPreprocessor()
    with pytest.raises(ValueError, match="require_time"):
        pre.split_by_station_and_time(samples, 0.5, 0.25, require_time=True)


def test_legacy_all_missing_still_allowed_without_flag():
    samples = [make_sample("st", None) for _ in range(6)]
    pre = SectionPreprocessor()
    train, val, test = pre.split_by_station_and_time(
        samples, 0.5, 0.25, require_time=False)
    assert len(train) + len(val) + len(test) == len(samples)
