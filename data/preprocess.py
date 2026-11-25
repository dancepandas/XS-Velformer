"""
数据预处理模块 —— 基于 sections.pkl 的变长断面特征工程与归一化

本模块从 sections.pkl 加载变长断面列表 (每条断面 K_i 个测速线, 不同断面可不同),
计算归一化特征:
    - x_norm = (x - x_min) / W, 逐断面左对齐归一化, 值域 [0, 1]
    - d_norm = h / h_max, 逐断面水深归一化 (h 直接是水深, 非河底高程)
    - bank_norm = 2 * min(x_norm, 1 - x_norm), 相对岸距归一化 [0, 1]
    - grad_norm = 局部水深梯度归一化 [0, 1]
    - v_norm = (v - v_mean_train) / v_std_train, 全局 StandardScaler 标准化
    - Z_norm = (Z - Z_mean_train) / Z_std_train, 全局 StandardScaler 标准化
    - h_max_norm = (h_max - h_max_mean_train) / h_max_std_train, 全局 StandardScaler 标准化
    - W_norm = (W - W_mean_train) / W_std_train, 全局 StandardScaler 标准化

bank_norm 和 grad_norm 是完整断面形态特征, 在预处理阶段一次性计算并保存,
不依赖训练时的 mask 策略, 保证同一测点在不同缺失场景下具有固定形态属性。

全局统计量仅用训练集拟合, 验证集和测试集复用, 避免数据泄漏。

数据集划分支持两种模式:
    - 按测站划分 (多站点, 测试跨站泛化能力)
    - 按时间划分 (单站点, 避免未来数据泄漏)
"""

import pickle
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional


def _parse_sample_time(ts: str) -> datetime:
    """解析样本时间戳为可比对 datetime, 格式/时区不可比时直接拒绝

    - 接受 ISO 8601 (含 "Z" 后缀, 统一替换为 +00:00)
    - 拒绝不可解析字符串 (含空串/占位符)
    - 排序键: naive 原样; aware 统一折算到 UTC —— 二者混用会 raise (拒绝)
    """
    t = ts.strip()
    if t.endswith(("Z", "z")):
        t = t[:-1] + "+00:00"
    return datetime.fromisoformat(t)


@dataclass
class SectionSample:
    """
    单条变长断面的完整数据

    Attributes:
        station_id:  测站编号 (如 "dz", "xt")
        x_raw:       [K] 原始起点距 (m), K 为该断面的测速线数量
        h_raw:       [K] 原始水深 (m)
        v_raw:       [K] 原始表面流速 (m/s)
        Z:           水位 (m)
        x_norm:      [K] 逐断面归一化起点距 [0, 1]
        d_norm:      [K] 逐断面归一化水深 [0, 1]
        bank_norm:   [K] 相对岸距归一化 [0, 1], 0=岸边, 1=断面中心
        grad_norm:   [K] 局部水深梯度归一化 [0, 1], 值越大说明形态变化越剧烈
        v_norm:      [K] 全局 StandardScaler 标准化流速
        h_max:       最大水深 (m)
        W:           水面宽度 (m)
        x_min:       最左测点起点距 (m)
        Z_norm:      全局 StandardScaler 标准化水位
        h_max_norm:  全局 StandardScaler 标准化最大水深
        W_norm:      全局 StandardScaler 标准化河宽
    """
    station_id: str
    x_raw: np.ndarray
    h_raw: np.ndarray
    v_raw: np.ndarray
    Z: float
    x_norm: np.ndarray
    d_norm: np.ndarray
    bank_norm: np.ndarray
    grad_norm: np.ndarray
    v_norm: np.ndarray
    h_max: float
    W: float
    x_min: float
    Z_norm: float = 0.0
    h_max_norm: float = 0.0
    W_norm: float = 0.0
    # 测流时间 (ISO 字符串, 如 "2025-06-01 08:30:00")。旧语料为 None;
    # 新语料装配时必须写入 —— split_by_station_and_time 有真实时间时按时间排序,
    # 否则退回"站内输入顺序 = 时间顺序"的语料约定。
    time: Optional[str] = None

    @property
    def K(self) -> int:
        """该断面的测速线数量"""
        return len(self.x_raw)


def compute_bank_and_grad_features(x_norm: np.ndarray, d_norm: np.ndarray, eps: float = 1e-6):
    """
    基于完整断面的 x_norm 和 d_norm 计算局部形态特征

    Args:
        x_norm:  [K] 归一化横向位置 [0, 1]
        d_norm:  [K] 归一化水深 [0, 1]
        eps:     防止除零的小常数

    Returns:
        bank_norm: [K] 相对岸距归一化 [0, 1]
        grad_norm: [K] 局部水深梯度归一化 [0, 1]
    """
    bank_norm = 2.0 * np.minimum(x_norm, 1.0 - x_norm)
    bank_norm = np.clip(bank_norm, 0.0, 1.0)

    K = len(x_norm)
    grad = np.zeros(K, dtype=np.float64)
    if K >= 2:
        grad[0] = abs(d_norm[1] - d_norm[0]) / (abs(x_norm[1] - x_norm[0]) + eps)
        grad[-1] = abs(d_norm[-1] - d_norm[-2]) / (abs(x_norm[-1] - x_norm[-2]) + eps)
    if K >= 3:
        dx = np.abs(x_norm[2:] - x_norm[:-2]) + eps
        dd = np.abs(d_norm[2:] - d_norm[:-2])
        grad[1:-1] = dd / dx

    grad_max = grad.max()
    if grad_max > 0:
        grad_norm = grad / (grad_max + eps)
    else:
        grad_norm = grad
    grad_norm = np.clip(grad_norm, 0.0, 1.0)

    return bank_norm.astype(np.float32), grad_norm.astype(np.float32)


class SectionPreprocessor:
    """
    基于 sections.pkl 的变长断面数据预处理器

    职责:
    1. 从 pkl 加载变长断面列表
    2. 计算 x_norm, d_norm (逐断面)
    3. fit_global_stats: 用训练集拟合 v, Z, h_max, W 的全局统计量
    4. apply_global_norm: 应用全局 StandardScaler 标准化
    5. split_by_station: 按测站划分 train/val/test
    6. split_by_time: 按时间顺序划分 train/val/test
    """

    def __init__(self, std_threshold: float = 1e-6):
        self.std_threshold = std_threshold
        self.global_stats: Dict[str, Dict[str, float]] = {}

    def load_sections(self, pkl_path: str) -> List[SectionSample]:
        """
        从 sections.pkl 加载变长断面列表并计算基础特征

        计算:
        - x_norm, d_norm (逐断面)
        - bank_norm, grad_norm (基于完整断面的局部形态特征)
        - v_norm 占位 (apply_global_norm 时用全局 StandardScaler 填充)
        - h_max, W, x_min
        """
        with open(pkl_path, "rb") as f:
            raw_sections = pickle.load(f)

        samples = []
        for sec in raw_sections:
            x = sec["x"]
            h = sec["h"]
            v = sec["v"]
            Z = sec["Z"]
            station_id = sec["station_id"]

            W = float(x.max() - x.min())
            x_min = float(x.min())
            h_max = float(h.max()) if h.max() > 0 else 1.0

            x_norm = (x - x_min) / W if W > 0 else np.zeros_like(x)
            d_norm = h / h_max
            v_norm = np.zeros_like(v)

            bank_norm, grad_norm = compute_bank_and_grad_features(x_norm, d_norm)

            samples.append(SectionSample(
                station_id=station_id,
                x_raw=x, h_raw=h, v_raw=v, Z=Z,
                x_norm=x_norm, d_norm=d_norm,
                bank_norm=bank_norm, grad_norm=grad_norm,
                v_norm=v_norm,
                h_max=h_max, W=W, x_min=x_min,
                time=sec.get("time"),  # 新语料装配必写; 旧语料 dict 无该键 → None
            ))
        return samples

    def fit_global_stats(self, samples: List[SectionSample]):
        """
        用训练集拟合 v, Z, h_max, W 的全局统计量

        所有 v_raw 拉平后计算全局均值和标准差, 用于 StandardScaler 标准化。
        统计量保存后用于对所有数据集 (train/val/test) 做相同的标准化,
        避免数据泄漏。
        """
        v_all = np.concatenate([s.v_raw for s in samples])
        Z_vals = np.array([s.Z for s in samples])
        h_max_vals = np.array([s.h_max for s in samples])
        W_vals = np.array([s.W for s in samples])
        self.global_stats = {
            "v": {"mean": float(v_all.mean()), "std": float(v_all.std()) + 1e-8},
            "Z": {"mean": float(Z_vals.mean()), "std": float(Z_vals.std()) + 1e-8},
            "h_max": {"mean": float(h_max_vals.mean()), "std": float(h_max_vals.std()) + 1e-8},
            "W": {"mean": float(W_vals.mean()), "std": float(W_vals.std()) + 1e-8},
        }

    def apply_global_norm(self, sample: SectionSample) -> SectionSample:
        """
        应用全局 StandardScaler 标准化到 v, Z, h_max, W

        使用 fit_global_stats 拟合的训练集统计量, 验证集和测试集复用。
        v_norm = (v - v_mean_train) / v_std_train
        """
        if not self.global_stats:
            return sample
        sample.v_norm = (sample.v_raw - self.global_stats["v"]["mean"]) / self.global_stats["v"]["std"]
        sample.Z_norm = (sample.Z - self.global_stats["Z"]["mean"]) / self.global_stats["Z"]["std"]
        sample.h_max_norm = (sample.h_max - self.global_stats["h_max"]["mean"]) / self.global_stats["h_max"]["std"]
        sample.W_norm = (sample.W - self.global_stats["W"]["mean"]) / self.global_stats["W"]["std"]
        return sample

    def split_stratified_by_station(
        self,
        samples: List[SectionSample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[SectionSample], List[SectionSample], List[SectionSample]]:
        """
        按测站分层随机划分数据集

        对每个 station_id 内部随机打乱后按比例切分, 再合并所有测站的
        train/val/test。保证每个集合都覆盖所有测站, 同时同一条完整断面
        只会属于其中一个集合。

        Args:
            samples: 全部断面列表
            train_ratio: 训练集占比
            val_ratio: 验证集占比
            seed: 随机种子
        """
        from collections import defaultdict

        by_station = defaultdict(list)
        for s in samples:
            by_station[s.station_id].append(s)

        rng = np.random.RandomState(seed)
        train_list, val_list, test_list = [], [], []

        for sid in sorted(by_station.keys()):
            station_samples = by_station[sid]
            indices = list(range(len(station_samples)))
            rng.shuffle(indices)

            n = len(indices)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))

            train_list.extend(station_samples[i] for i in indices[:n_train])
            val_list.extend(station_samples[i] for i in indices[n_train:n_train + n_val])
            test_list.extend(station_samples[i] for i in indices[n_train + n_val:])

        return train_list, val_list, test_list

    def split_by_station(
        self,
        samples: List[SectionSample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[SectionSample], List[SectionSample], List[SectionSample]]:
        """
        按测站划分数据集 (测试跨站泛化能力), 按断面数量比例分配测站

        贪心策略: 将测站按断面数降序排列, 依次将每个测站分配到当前断面数
        占比最低于目标比例的集合中, 使得 train/val/test 的断面数尽量接近
        目标比例, 同时保证同一测站的所有断面只出现在一个集合中。

        Args:
            samples: 全部断面列表
            train_ratio: 训练集断面数目标占比
            val_ratio: 验证集断面数目标占比
        """
        from collections import Counter

        station_counts = Counter(s.station_id for s in samples)
        total = len(samples)
        target_train = total * train_ratio
        target_val = total * val_ratio

        # 按断面数降序排列, 大站优先分配
        sorted_stations = sorted(station_counts.keys(),key=lambda sid: station_counts[sid], reverse=True)
        train_stations = set()
        val_stations = set()
        test_stations = set()
        n_train = 0
        n_val = 0
        n_test = 0

        rng = np.random.RandomState(42)

        for sid in sorted_stations:
            cnt = station_counts[sid]
            # 计算分配到每个集合后与目标比例的差距
            deficit_train = target_train - n_train
            deficit_val = target_val - n_val
            deficit_test = (total - target_train - target_val) - n_test

            # 选择差距最大的集合 (最需要补充断面的集合)
            deficits = {"train": deficit_train, "val": deficit_val, "test": deficit_test}
            chosen = max(deficits, key=lambda k: deficits[k])

            if chosen == "train":
                train_stations.add(sid)
                n_train += cnt
            elif chosen == "val":
                val_stations.add(sid)
                n_val += cnt
            else:
                test_stations.add(sid)
                n_test += cnt

        train_samples = [s for s in samples if s.station_id in train_stations]
        val_samples = [s for s in samples if s.station_id in val_stations]
        test_samples = [s for s in samples if s.station_id in test_stations]

        return train_samples, val_samples, test_samples

    def split_by_time(
        self,
        samples: List[SectionSample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[SectionSample], List[SectionSample], List[SectionSample]]:
        """
        按时间顺序划分数据集 (避免未来数据泄漏)

        sections.pkl 中同一测站的断面已按测流时间排序,
        直接按索引顺序切分:
        - 训练集: 前 70%
        - 验证集: 中间 15%
        - 测试集: 后 15%

        适用于单测站数据或需要保持时间顺序的场景。
        """
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            samples[:n_train],
            samples[n_train:n_train + n_val],
            samples[n_train + n_val:],
        )

    def split_by_station_and_time(
        self,
        samples: List[SectionSample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        require_time: bool = False,
    ) -> Tuple[List[SectionSample], List[SectionSample], List[SectionSample]]:
        """
        跨站 × 跨时双重留出划分 (最严格的泛化协议)

        两步:
        1. 站点留出: 用与 split_by_station 相同的贪心策略, 将断面数占比约
           (1 - train_ratio - val_ratio) 的测站整体留出 —— 这些站的全部断面
           (任意时刻) 都进测试集, 模型训练从未见过这些站。
        2. 时间留出: 对其余测站, 按各站断面已按测流时间排序的约定
           (同 split_by_time), 每站依次切 train_ratio / val_ratio /
           其余, 最后一块进测试集 —— 模型从未见过这些时刻。

        最终 test = 留出站全部断面 ∪ 保留站的时间留出块, 是两类泛化测试的并集:
        留出站样本考核跨站泛化, 保留站的时间留出块考核跨时泛化 —— 论文报告时
        必须按"并集"表述, 不得声称每个测试样本同时满足双重留出。
        注意: 由于两个维度同时留出, test 占比会高于单维协议
        (默认参数下约 28%), train/val 相应缩小; 比例漂移是双重留出的
        固有代价, 论文中报告时需说明。

        Args:
            samples: 全部断面列表
            train_ratio: 保留站内部时间切分的训练占比 (按测次组计)
            val_ratio:   保留站内部时间切分的验证占比 (按测次组计)
            seed:        保留参数兼容; 站点分配为确定性贪心 (按断面数降序), 未用到随机
            require_time: True 时拒绝"全部缺 time"的旧语料兼容回退 (新语料协议必须 True;
                旧语料复现保持 False)
        """
        from collections import Counter, defaultdict

        station_counts = Counter(s.station_id for s in samples)
        total = len(samples)
        holdout_target = total * (1.0 - train_ratio - val_ratio)

        # 阶段 1: 贪心留出整站 (大站优先, 与 split_by_station 同策略)
        sorted_stations = sorted(station_counts.keys(),
                                 key=lambda sid: station_counts[sid], reverse=True)
        rng = np.random.RandomState(seed)
        candidates = sorted_stations[:]
        holdout_stations = set()
        n_holdout = 0
        # 从小站开始逐个累加会在只剩大站时比例失控; 与 split_by_station 一致,
        # 每步把下一个站放进"当前最需要的集合"。这里只需要二选一:
        # 已累积的留出量离目标还差多少 vs 已保留量离 (total-目标) 还差多少。
        n_keep = 0
        keep_stations = set()
        for sid in candidates:
            cnt = station_counts[sid]
            deficit_hold = holdout_target - n_holdout
            deficit_keep = (total - holdout_target) - n_keep
            if deficit_hold >= deficit_keep:
                holdout_stations.add(sid)
                n_holdout += cnt
            else:
                keep_stations.add(sid)
                n_keep += cnt

        # 阶段 2: 保留站内部按时间切
        by_station = defaultdict(list)
        for s in samples:
            if s.station_id in keep_stations:
                by_station[s.station_id].append(s)

        # 时间排序来源: SectionSample.time (新语料装配时写入真实测流时间);
        # 全部缺失时退回旧语料约定 —— 站内输入顺序 = 测流时间顺序。
        has_time = all(getattr(s, "time", None) for s in samples)
        time_keys: Dict[int, datetime] = {}
        if has_time:
            # 全量解析校验 (不只保留站): 格式异常必须在这里暴露, 而非静默字典序
            for s in samples:
                try:
                    time_keys[id(s)] = _parse_sample_time(s.time)
                except ValueError as exc:
                    raise ValueError(
                        f"测站 {s.station_id} 的时间戳不可解析: {s.time!r} ({exc}); "
                        "新语料装配必须统一写出可解析的 ISO 8601 时间") from exc
            aware = [dt.tzinfo is not None for dt in time_keys.values()]
            if any(aware) and not all(aware):
                raise ValueError(
                    f"{sum(aware)}/{len(aware)} 条样本带时区信息, 其余为 naive 时间: "
                    "混用时区语义不可排序, 装配阶段必须统一 (全 naive 或全 aware)")
            # 相同时间戳允许出现 (同一测次多条记录): 成组切分保证同组不跨集合
        else:
            n_missing = sum(1 for s in samples if not getattr(s, "time", None))
            if 0 < n_missing < len(samples):
                raise ValueError(
                    f"{n_missing}/{len(samples)} 条断面缺少 time 字段: "
                    "跨时划分要求全部样本携带真实测流时间 "
                    "(或旧语料保证站内输入序=时间序且 time 全为 None)")
            if n_missing == len(samples) and require_time:
                raise ValueError(
                    "require_time=True: 全部样本缺 time, 旧语料\"输入序=时间序\""
                    "回退被禁止 —— 新语料协议要求真实测流时间")
            # n_missing == len(samples) 且 require_time=False: 旧语料回退

        train_list, val_list, test_list = [], [], []
        for sid, secs in by_station.items():
            if has_time:
                # 按站内规范化时间戳成组再切: 同一时刻 (同一测次) 的全部记录必须
                # 落入同一集合, 否则同测次信息会跨越 train/val/test 边界泄漏
                def _sort_key(s, _tk=time_keys):
                    dt = _tk[id(s)]
                    return dt.astimezone(timezone.utc) if dt.tzinfo else dt
                groups: Dict[object, List[SectionSample]] = defaultdict(list)
                for s in sorted(secs, key=_sort_key):
                    groups[_sort_key(s)].append(s)
                g_list = list(groups.values())       # 时间序的测次组
                g = len(g_list)
                g_train = max(1, int(g * train_ratio))
                g_val = max(1, int(g * val_ratio))
                g_val = min(g_val, g - g_train)      # 至少给时间留出留 1 组
                for grp in g_list[:g_train]:
                    train_list.extend(grp)
                for grp in g_list[g_train:g_train + g_val]:
                    val_list.extend(grp)
                for grp in g_list[g_train + g_val:]:
                    test_list.extend(grp)
            else:
                # 无 time 旧语料: 保持输入序 (与 split_by_time 约定一致), 按记录数切
                n = len(secs)
                n_train = max(1, int(n * train_ratio))
                n_val = max(1, int(n * val_ratio))
                n_val = min(n_val, n - n_train)  # 至少给时间留出留 1 条
                train_list.extend(secs[:n_train])
                val_list.extend(secs[n_train:n_train + n_val])
                test_list.extend(secs[n_train + n_val:])

        # 阶段 1 留出的站: 全部进测试集
        test_list.extend(s for s in samples if s.station_id in holdout_stations)
        _ = rng  # 保留 seed 参数以备平局随机化
        return train_list, val_list, test_list
