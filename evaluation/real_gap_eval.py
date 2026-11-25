"""Track B 真实缺失断面合理性评估 —— 锚点定理 D1–D10 (A1–A10)

设计文档: docs/revision/track-b-hydraulic-anchors.md

数据基础: aether 分片的逐垂线溯源字段
    raw_valid [K,3]: 每线 3 个原始视频片段的有效观测标记 (bool)
    is_algo    [K]:  平台算法改写线标记
    raw_v    [K,3]:  每线原始片段速度 (未过终端 QC, 用前必须门控)
实测语义 (2026-11-11, 全语料 687,244 线): 硬缺失线为 0 (无原始片段的线
入库前已被丢弃); is_algo 改写线占 45.7%, 其上平台值与自身 raw 均值的
偏差 p90 = 1.03 m/s —— 平台内置值不可作真值的直接证据。缺失场景由
改写线承担, raw 观测与平台值构成双参照。

关键设计决定 (用户确认 2026-11-11):
    1. 真实缺失线上没有 ground truth —— 本模块只做合理性诊断, 不报 RMSE
    2. 平台自身插补值 (is_algo 线的 v_surface) 不可当真值, 仅可作为
       一个对比方法 (platform_builtin) 的输出

诊断层只依赖 numpy (物理量纲, 无需训练管线); 模型接入函数
impute_with_model 是桩, 待公平重跑的 checkpoint 落地后经归一化管线
(SectionPreprocessor + adapters) 填充。

用法:
    python evaluation/real_gap_eval.py                 # 合成自测
    python evaluation/real_gap_eval.py --shards <dir>  # 扫描真实缺失断面清单
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# 真实缺失断面提取
# ---------------------------------------------------------------------------

@dataclass
class RealGapSection:
    """一条含"真实缺失"线的断面 (物理量纲)

    2026-11-11 实测: 平台语料硬缺失线 (raw_valid 全 False) 为 0 —— 无原始
    片段的线入库前已被丢弃。缺失场景由 is_algo 改写线承担: 仪器有原始
    观测但终值被平台算法替换 (全语料 45.7%), 对模型按缺失线处理, 其
    raw_v 均值 (QC 门控) 与平台改写值构成双参照 (见 compare_references)。
    """
    station: str
    device: str
    time: str
    x: np.ndarray            # [K] 起点距 (m)
    depth: np.ndarray        # [K] 水深 (m)
    v_obs: np.ndarray        # [K] 流速, 缺失线为 np.nan (m/s)
    gap_idx: np.ndarray      # 缺失线索引 (改写线; 未来含硬缺失)
    obs_idx: np.ndarray      # 观测线索引
    width: float             # 水面宽 (m)
    Z: float                 # 水位 (m)
    platform_v: Optional[np.ndarray] = None  # [K] 平台改写值 (仅缺失线有效)
    v_raw: Optional[np.ndarray] = None  # [K] QC 门控的原始观测均值 (全线)
    section_area: float = float("nan")
    surface_avg_velocity: float = float("nan")

    @property
    def bank_norm(self) -> np.ndarray:
        """[K] 相对岸距 [0,1], 0=岸边 (与训练特征同口径)"""
        xn = (self.x - self.x.min()) / max(self.width, 1e-9)
        return 2.0 * np.minimum(xn, 1.0 - xn)


def _load_shard(path: str) -> list[dict]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_real_gap_sections(
    shards_dir: str,
    min_observed: int = 8,
    max_gap_frac: float = 0.5,
    max_sections_per_device: Optional[int] = None,
    raw_v_max: float = 5.633,
) -> List[RealGapSection]:
    """从 aether 分片目录扫描"真实缺失"断面 (is_algo 改写线为缺失场景)

    缺失线 = is_algo 为真 (平台算法改写; 未来数据含 raw_valid 全 False
    的硬缺失线时同样纳入)。raw_v 门控: 线内有效片段均值 ∈ (0, raw_v_max]
    才作参照 (raw_v 未过终端 QC, 分片实测最大偏差 20 m/s, 必须设界)。

    选择标准 (排除退化情形):
        - 观测线数 >= min_observed
        - 缺失占比 <= max_gap_frac
        - 至少存在 1 条缺失线
    """
    import warnings as _warnings

    out: List[RealGapSection] = []
    per_device: collections.Counter = collections.Counter()
    for path in sorted(glob.glob(str(Path(shards_dir) / "sections" / "*.pkl"))):
        for sec in _load_shard(path):
            x = np.asarray(sec["x"], dtype=float)
            depth = np.asarray(sec["depth"], dtype=float)
            v = np.asarray(sec["v_surface"], dtype=float)
            raw_valid = np.asarray(sec.get("raw_valid"), dtype=bool)
            if raw_valid is None or raw_valid.ndim != 2:
                continue
            observed = raw_valid.any(axis=1)
            is_algo = np.asarray(sec.get("is_algo"))
            if is_algo is None or is_algo.shape != v.shape:
                gap = ~observed
            else:
                gap = np.asarray(is_algo, dtype=bool) | (~observed)
            n_obs, n_gap = int(observed.sum()), int(gap.sum())
            if n_gap < 1 or n_obs < min_observed:
                continue
            if n_gap / max(n_gap + n_obs, 1) > max_gap_frac:
                continue
            device = str(sec.get("device") or "")
            if max_sections_per_device is not None:
                if per_device[device] >= max_sections_per_device:
                    continue
                per_device[device] += 1

            # raw_v 门控均值 (双参照之一; 未过终端 QC 必须设界)
            v_raw = np.full(len(v), np.nan)
            raw_v = np.asarray(sec.get("raw_v"), dtype=float)
            if raw_v.ndim == raw_valid.ndim:
                masked = np.where(raw_valid, raw_v, np.nan)
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", category=RuntimeWarning)
                    means = np.nanmean(masked, axis=1)
                ok = np.isfinite(means) & (means > 0) & (means <= raw_v_max)
                v_raw[ok] = means[ok]

            v_full = np.where(observed & ~gap, v, np.nan)
            platform_v = np.where(gap, v, np.nan) if n_gap else None
            out.append(RealGapSection(
                station=str(sec.get("station") or ""),
                device=device,
                time=str(sec.get("time") or ""),
                x=x, depth=depth, v_obs=v_full,
                gap_idx=np.where(gap)[0],
                obs_idx=np.where(observed & ~gap)[0],
                width=float(sec.get("water_width") or (x.max() - x.min())),
                Z=float(sec.get("water_level") or np.nan),
                platform_v=platform_v,
                v_raw=v_raw,
                section_area=float(sec.get("section_area") or np.nan),
                surface_avg_velocity=float(sec.get("surface_avg_velocity") or np.nan),
            ))
    return out


# ---------------------------------------------------------------------------
# A4/D9 Chiu 熵理论参照 (最强锚点)
# ---------------------------------------------------------------------------

def isovel_coord(x: np.ndarray, x_star: float, width: float) -> np.ndarray:
    """Chiu 等流速线坐标 ξ/ξ_max ∈ [0,1]: 0 在两岸 (u=0), 1 在最大流速处 x*

    Chiu (1988) 的 ξ 从边界 (ξ₀=0, u=0) 沿等流速线增至最大流速点
    (ξ_max, u=umax); 表面流操作化取两岸到 x* 的线性归一化距离。
    """
    left = (x - x.min()) / max(x_star - x.min(), width * 1e-6)
    right = (x.max() - x) / max(x.max() - x_star, width * 1e-6)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def chiu_fit(v_obs: np.ndarray, xi_obs: np.ndarray) -> tuple[float, float]:
    """从观测线拟合 Chiu 熵参数 M (umax 取观测最大值)

    u = (umax/M) ln(1 + (e^M − 1) ξ)   —— ξ∈[0,1]
    网格搜索 M ∈ [0.05, 20], 最小二乘。M→∞ 时退化为线性分布。
    """
    umax = float(np.nanmax(v_obs))
    grid = np.concatenate([np.linspace(0.05, 8, 400), np.linspace(8, 20, 60)])
    best_m, best_err = 20.0, np.inf
    for M in grid:
        pred = umax / M * np.log1p(np.expm1(M) * xi_obs)
        err = float(np.mean((pred - v_obs) ** 2))
        if err < best_err:
            best_m, best_err = float(M), err
    return best_m, umax


def chiu_profile(M: float, umax: float, xi_all: np.ndarray) -> np.ndarray:
    return umax / M * np.log1p(np.expm1(M) * xi_all)


def chiu_mean_ratio(M: float) -> float:
    """Chiu 解析的平均/最大流速比 ū/umax = e^M/(e^M−1) − 1/M (A9/D9)"""
    if M < 1e-6:
        return 1.0
    return float(np.exp(M) / (np.expm1(M)) - 1.0 / M)


# ---------------------------------------------------------------------------
# 诊断器 D1–D10 (物理量纲, 输入完整剖面 = 观测值 + 插补值)
# ---------------------------------------------------------------------------

def d1_noslip_bank_violation(x, v, width, gap_idx, obs_idx,
                             bank_frac=0.05, rel_thr=0.2) -> dict:
    """A1 无滑移: 岸线 (bank_norm<bank_frac) 的**插补**点 v > rel_thr×v_max_obs
    记违例; 观测线违例单列 (仪器侧参考, 不计入模型指标)"""
    bank = 2.0 * np.minimum((x - x.min()) / max(width, 1e-9),
                            1.0 - (x - x.min()) / max(width, 1e-9))
    vmax_obs = float(np.nanmax(v[obs_idx])) if len(obs_idx) else np.nan
    lines = np.arange(len(x))
    near_gap = (bank < bank_frac) & np.isin(lines, gap_idx)
    near_obs = (bank < bank_frac) & np.isin(lines, obs_idx)
    if not np.isfinite(vmax_obs) or vmax_obs <= 0:
        return {"violations": 0, "rate": np.nan, "n_imputed": int(near_gap.sum()),
                "n_observed": int(near_obs.sum()), "observed_violations": 0}
    viol_gap = near_gap & (v > rel_thr * vmax_obs)
    viol_obs = near_obs & (v > rel_thr * vmax_obs)
    n_gap = int(near_gap.sum())
    return {"violations": int(viol_gap.sum()),
            "rate": float(viol_gap.sum() / n_gap) if n_gap else np.nan,
            "n_imputed": n_gap, "n_observed": int(near_obs.sum()),
            "observed_violations": int(viol_obs.sum())}


def d2_wall_decay(x, v, width, gap_idx, obs_idx, band=0.25, n_min=4) -> dict:
    """A2/A3 壁面剪切层: 近岸观测带拟合 ln v = ln a + d/λ (d=离岸距离,
    SKM 侧向剪切层的指数形态; λ 为 e 折长度, v 趋岸指数衰减), 报告
    插补点残差 RMS (按带内观测 std 归一) 与两岸 λ 对称性。
    横向表面数据不能直接检验垂向对数律, 故以剪切层指数衰减为形状参照。
    """
    bank = 2.0 * np.minimum((x - x.min()) / max(width, 1e-9),
                            1.0 - (x - x.min()) / max(width, 1e-9))
    lines = np.arange(len(x))
    mid = 0.5 * (x.min() + x.max())
    vstd_all = []
    out = {"lambda_left": np.nan, "lambda_right": np.nan,
           "imputed_resid_rms_norm": np.nan, "n_obs_band": 0, "n_imp_band": 0}
    resids = []
    for side, sel_side in (("left", x < mid), ("right", ~((x < mid)))):
        obs_band = sel_side & (bank < band) & np.isin(lines, obs_idx) & (v > 0)
        if int(obs_band.sum()) < n_min:
            continue
        slope, intercept = np.polyfit(bank[obs_band], np.log(v[obs_band]), 1)
        if slope <= 0:  # 带内非"趋岸衰减"形态, 参照不成立
            continue
        out[f"lambda_{side}"] = float(1.0 / slope)
        vstd_all.append(float(np.std(v[obs_band])))
        imp_band = sel_side & (bank < band) & np.isin(lines, gap_idx)
        out["n_obs_band"] += int(obs_band.sum())
        out["n_imp_band"] += int(imp_band.sum())
        if imp_band.any():
            pred = np.exp(intercept + slope * bank[imp_band])
            resids.extend((v[imp_band] - pred).tolist())
    scale = max(vstd_all) if vstd_all else np.nan
    if resids and np.isfinite(scale) and scale > 0:
        out["imputed_resid_rms_norm"] = float(np.sqrt(np.mean(np.square(resids))) / scale)
    return out


def d3_lateral_variation_rate(x, v, gap_idx: np.ndarray) -> dict:
    """A3 SKM 动机的一致性: 插补段 |dv/dx| 均值 / 观测段 |dv/dx| 均值

    SKM 解在均匀区为指数形态 —— 变化率量级突变 (比值远超 1) 提示插补
    段与断面的侧向动量结构不符。均匀/突变分层留给上游按几何调用。
    """
    dv = np.abs(np.diff(v))
    dx = np.abs(np.diff(x))
    rate = dv / np.maximum(dx, 1e-6)
    seg_of = np.arange(len(rate))
    obs_mask = ~np.isin(seg_of, gap_idx) & ~np.isin(seg_of, gap_idx - 1)
    imp_mask = np.isin(seg_of, gap_idx) | np.isin(seg_of, gap_idx - 1)
    obs_rate = rate[obs_mask & np.isfinite(rate)]
    imp_rate = rate[imp_mask & np.isfinite(rate)]
    if len(obs_rate) < 2 or len(imp_rate) < 1:
        return {"ratio": np.nan, "obs_rate": np.nan, "imp_rate": np.nan}
    ro, ri = float(np.mean(obs_rate)), float(np.mean(imp_rate))
    return {"ratio": ri / max(ro, 1e-9), "obs_rate": ro, "imp_rate": ri}


def d5_depth_velocity_direction(v, depth, gap_idx: np.ndarray) -> dict:
    """A5 水力几何: 观测线与插补线的 depth–velocity 相关方向一致性 + 失效清单"""
    obs = np.isfinite(v)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 and np.std(a) > 1e-9 and np.std(b) > 1e-9 else np.nan
    c_obs = corr(v[obs], depth[obs])
    imp = np.zeros(len(v), dtype=bool)
    imp[gap_idx] = True
    c_imp = corr(v[imp], depth[imp]) if int(imp.sum()) >= 3 else np.nan
    consistent = (np.sign(c_obs) == np.sign(c_imp)) if np.isfinite(c_obs) and np.isfinite(c_imp) else np.nan
    return {"corr_obs": c_obs, "corr_imputed": c_imp, "direction_consistent": consistent}


def d6_magnitude_bounds(v, depth, obs_idx, gap_idx, fr_thr=1.0, out_thr=1.5) -> dict:
    """A6 Manning/Froude: **插补**点局部 Fr>=1 标记 + 超观测最大值
    out_thr 倍的离群; 观测线同类计数单列 (仪器侧参考)"""
    vmax_obs = float(np.nanmax(v[obs_idx])) if len(obs_idx) else np.nan
    fr = v / np.sqrt(GRAVITY * np.maximum(depth, 1e-3))
    lines = np.arange(len(v))
    on_gap = np.isin(lines, gap_idx)
    supercritical = int(np.sum(on_gap & (fr >= fr_thr)))
    outliers = 0
    if np.isfinite(vmax_obs) and vmax_obs > 0:
        outliers = int(np.sum(on_gap & (v > out_thr * vmax_obs)))
    obs_outliers = int(np.sum(v[obs_idx] > out_thr * vmax_obs)) if np.isfinite(vmax_obs) and vmax_obs > 0 else 0
    return {"supercritical_pts": supercritical, "outlier_pts": outliers,
            "observed_outlier_pts": obs_outliers,
            "v_max_obs": vmax_obs, "v_max_full": float(np.nanmax(v))}


def d7_argmax_thalweg_offset(x, v, depth, width, gap_idx: np.ndarray) -> dict:
    """A7/A10 dip: 全剖面 argmax 与最深点 (thalweg) 的偏移 (以河宽归一);
    argmax 落在插补线上属模型影响, 单列标记"""
    i_v = int(np.nanargmax(v))
    i_d = int(np.nanargmax(depth))
    return {"offset_norm": float(abs(x[i_v] - x[i_d]) / max(width, 1e-9)),
            "argmax_on": "imputed" if i_v in set(gap_idx.tolist()) else "observed",
            "x_argmax": float(x[i_v]), "x_thalweg": float(x[i_d])}


def d9_surface_mean_ratio(M: float) -> dict:
    """A9 Chiu 解析 ū/umax —— 替代固定 0.85 系数的逐断面定理化系数"""
    return {"chiu_mean_ratio": chiu_mean_ratio(M), "entropy_M": M}


# ---------------------------------------------------------------------------
# D8 连续性 (邻近实测流量参照) 与 D10 跨模型共识
# ---------------------------------------------------------------------------

def velocity_area_discharge(x, v, depth, obs_only=False) -> float:
    """ISO 748 velocity-area: Q = Σ v_i · d_i · Δx_i (中点法)"""
    if obs_only:
        m = np.isfinite(v)
        x, v, depth = x[m], v[m], depth[m]
    if len(x) < 2:
        return float("nan")
    dx = np.diff(x)
    mid_v = 0.5 * (v[1:] + v[:-1])
    mid_d = 0.5 * (depth[1:] + depth[:-1])
    return float(np.nansum(mid_v * mid_d * dx))


def section_key(s: RealGapSection) -> str:
    return f"{s.device}|{s.time}"


def d8_discharge_coherence(sections: Sequence[RealGapSection],
                           imputed: Dict[str, np.ndarray],
                           dt_max_hours: float = 6.0,
                           dz_max: float = 0.2) -> dict:
    """A8 连续性: 插补断面 Q vs 同设备邻近实测断面 Q (时间线性插值参照)

    参照 Q 来自邻近断面的观测线 (非插补), 要求 |ΔZ| < dz_max 且 Δt < dt_max,
    且邻近断面自身前后都有观测。返回逐断面相对偏差与汇总分布。
    """
    import datetime as _dt

    def parse(t: str):
        return _dt.datetime.fromisoformat(t.split(".")[0])

    by_device: Dict[str, list] = collections.defaultdict(list)
    for s in sections:
        by_device[s.device].append(s)

    rows = []
    for s in sections:
        if section_key(s) not in imputed:
            continue
        v_full = s.v_obs.copy()
        v_full[s.gap_idx] = imputed[section_key(s)]
        q_imp = velocity_area_discharge(s.x, v_full, s.depth)
        t0 = parse(s.time)
        neighbors = []
        for o in by_device[s.device]:
            if o is s:
                continue
            dt = abs((parse(o.time) - t0).total_seconds()) / 3600.0
            if dt > dt_max_hours or not np.isfinite(o.Z) or not np.isfinite(s.Z):
                continue
            if abs(o.Z - s.Z) > dz_max:
                continue
            neighbors.append((dt, o))
        if not neighbors:
            rows.append({"station": s.station, "time": s.time, "q_imputed": q_imp,
                         "q_ref": np.nan, "rel_dev": np.nan, "n_ref": 0})
            continue
        qs, ws = [], []
        for dt, o in neighbors:
            q_ref = velocity_area_discharge(o.x, o.v_obs, o.depth, obs_only=True)
            if np.isfinite(q_ref) and q_ref > 0:
                qs.append(q_ref)
                ws.append(1.0 / max(dt, 0.1))
        if not qs:
            rows.append({"station": s.station, "time": s.time, "q_imputed": q_imp,
                         "q_ref": np.nan, "rel_dev": np.nan, "n_ref": 0})
            continue
        q_ref = float(np.average(qs, weights=ws))
        rel = (q_imp - q_ref) / q_ref
        rows.append({"station": s.station, "time": s.time, "q_imputed": q_imp,
                     "q_ref": q_ref, "rel_dev": rel, "n_ref": len(qs)})
    devs = np.array([r["rel_dev"] for r in rows if np.isfinite(r["rel_dev"])])
    return {"per_section": rows,
            "summary": {"n": len(devs),
                        "median_rel_dev": float(np.median(devs)) if len(devs) else np.nan,
                        "iqr": [float(np.percentile(devs, 25)), float(np.percentile(devs, 75))] if len(devs) else [np.nan, np.nan],
                        "rmse_rel": float(np.sqrt(np.mean(devs ** 2))) if len(devs) else np.nan}}


def d10_consensus_spread(imputed_by_model: Dict[str, np.ndarray]) -> dict:
    """跨模型共识: 每条缺失线的模型间标准差 + 离群模型 (MAD>3)"""
    names = list(imputed_by_model)
    if len(names) < 2:
        return {"per_line_std": None, "outlier_models": []}
    stack = np.stack([imputed_by_model[n] for n in names])  # [M, K_gap]
    std = stack.std(axis=0)
    med = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - med), axis=0) + 1e-9
    z = np.abs(stack - med) / mad
    outlier_models = [n for i, n in enumerate(names)
                      if float(np.mean(z[i] > 3.0)) > 0.3]
    return {"per_line_std": std.tolist(), "median_std": float(np.median(std)),
            "outlier_models": outlier_models}


# ---------------------------------------------------------------------------
# 单断面汇总
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# B-α 双参照对比 (仅改写线场景有: raw 观测 + 平台改写值)
# ---------------------------------------------------------------------------

def compare_references(sec: RealGapSection, v_imputed_gap: np.ndarray) -> dict:
    """缺失线上模型插补值 vs (a) QC 门控 raw 观测 (b) 平台改写值

    raw 与平台值都不是真值: raw 是仪器测量但被平台改写流程拒绝的输入
    (含噪声), 平台值是另一种算法的输出 —— 双参照的**一致性模式**才是
    诊断量 (R2-8: 平台内置值不可作为评估基准, 本函数正是量化这一点)。
    """
    imp = np.asarray(v_imputed_gap, float)
    n = len(imp)
    raw = sec.v_raw[sec.gap_idx] if sec.v_raw is not None else np.full(n, np.nan)
    plat = (sec.platform_v[sec.gap_idx] if sec.platform_v is not None
            else np.full(n, np.nan))
    d_raw, d_plat = imp - raw, imp - plat

    def stats(d: np.ndarray) -> dict:
        ok = np.isfinite(d)
        if not ok.any():
            return {"n": 0, "bias": np.nan, "mae": np.nan, "p90_absdev": np.nan}
        ad = np.abs(d[ok])
        return {"n": int(ok.sum()), "bias": float(np.median(d[ok])),
                "mae": float(np.mean(ad)),
                "p90_absdev": float(np.percentile(ad, 90))}

    return {"vs_raw": stats(d_raw), "vs_platform": stats(d_plat),
            "raw_platform_gap": stats(raw - plat)}


def evaluate_section(sec: RealGapSection, v_imputed_gap: np.ndarray,
                     model_name: str = "model") -> dict:
    """对一条真实缺失断面 + 一个模型的插补输出, 计算 D1–D7, D9 + 双参照

    v_imputed_gap: [len(gap_idx)] 缺失线上的插补值 (物理量纲 m/s)
    """
    v = sec.v_obs.copy()
    v[sec.gap_idx] = v_imputed_gap
    obs_max_idx = int(np.nanargmax(sec.v_obs))
    xi_all = isovel_coord(sec.x, float(sec.x[obs_max_idx]), sec.width)
    M, umax = chiu_fit(sec.v_obs[sec.obs_idx], xi_all[sec.obs_idx])

    out = {"station": sec.station, "time": sec.time, "model": model_name,
           "n_gap": int(len(sec.gap_idx)), "n_obs": int(len(sec.obs_idx))}
    out["D1_noslip"] = d1_noslip_bank_violation(sec.x, v, sec.width,
                                                sec.gap_idx, sec.obs_idx)
    out["D2_wall_decay"] = d2_wall_decay(sec.x, v, sec.width,
                                         sec.gap_idx, sec.obs_idx)
    out["D3_skm_rate"] = d3_lateral_variation_rate(sec.x, v, sec.gap_idx)
    out["D4_chiu"] = {
        "M": M, "umax_obs": umax,
        "resid_rms_norm": float(np.sqrt(np.mean(
            (v[sec.gap_idx] - chiu_profile(M, umax, xi_all[sec.gap_idx])) ** 2))
            / max(umax, 1e-9)),
    }
    out["D5_dv_direction"] = d5_depth_velocity_direction(v, sec.depth, sec.gap_idx)
    out["D6_bounds"] = d6_magnitude_bounds(v, sec.depth, sec.obs_idx, sec.gap_idx)
    out["D7_thalweg"] = d7_argmax_thalweg_offset(sec.x, v, sec.depth, sec.width,
                                                 sec.gap_idx)
    out["D9_ratio"] = d9_surface_mean_ratio(M)
    out["ref_compare"] = compare_references(sec, v_imputed_gap)
    return out


# ---------------------------------------------------------------------------
# 模型接入桩 (公平重跑 checkpoint 落地后实现)
# ---------------------------------------------------------------------------

def impute_with_model(model_type: str, sections: Sequence[RealGapSection],
                      ckpt_path: str, device: str = "cpu") -> Dict[str, np.ndarray]:
    """经归一化管线 (SectionPreprocessor + adapter.predict) 生成缺失线插补值。

    TODO(重跑后): 与 evaluation/model_adapters.py 对接 —— context=观测线,
    query=缺失线, 反归一化后按 section_key(device|time) 返回 [K_gap]
    物理量纲数组。
    """
    raise NotImplementedError("等待公平重跑 checkpoint; 诊断层已可独立使用")


# ---------------------------------------------------------------------------
# 合成自测
# ---------------------------------------------------------------------------

def _selftest() -> None:
    rng = np.random.default_rng(0)
    # 合成断面: 60 线, 抛物型水深, Chiu 型剖面 (M=3), 8 条真实缺失
    x = np.linspace(0, 50, 60)
    depth = 3.0 * np.exp(-((x - 28) ** 2) / (2 * 12 ** 2)) + 0.3
    x_star = float(x[np.argmax(depth)])
    xi = isovel_coord(x, x_star, x.max() - x.min())
    v_true = chiu_profile(3.0, 1.6, xi)  # 物理合理剖面
    # 缺失限中段 (避开近岸 6 线), 更贴近仪器漏测的实际分布
    gap = np.sort(rng.choice(np.arange(6, 54), size=8, replace=False))
    v_obs = v_true.copy()
    platform_v = v_true.copy()
    v_obs[gap] = np.nan
    # 平台内置插补 = 线性插补 (近似平台行为, 仅作对比方法)
    platform_v[gap] = np.interp(x[gap], x[np.setdiff1d(np.arange(60), gap)],
                                v_true[np.setdiff1d(np.arange(60), gap)])
    sec = RealGapSection(station="SYN", device="D0", time="2026-01-01 12:00:00",
                         x=x, depth=depth, v_obs=v_obs, gap_idx=np.sort(gap),
                         obs_idx=np.setdiff1d(np.arange(60), np.sort(gap)),
                         width=float(x.max() - x.min()), Z=100.0,
                         platform_v=platform_v, v_raw=v_true.copy())
    good = evaluate_section(sec, v_true[gap], model_name="plausible")
    wild = v_true[gap].copy()
    wild[:3] = 5.0  # 岸边/主槽异常高值
    bad = evaluate_section(sec, wild, model_name="violating")

    print("== self-test: plausible profile ==")
    print(f"  D1 rate={good['D1_noslip']['rate']:.2f}"
          f"  D2 resid_norm={good['D2_wall_decay']['imputed_resid_rms_norm']:.3f}"
          f" (lL={good['D2_wall_decay']['lambda_left']:.2f})"
          f"  D4 resid_norm={good['D4_chiu']['resid_rms_norm']:.3f} (M={good['D4_chiu']['M']:.2f})"
          f"  D5 consistent={good['D5_dv_direction']['direction_consistent']}"
          f"  D6 outliers={good['D6_bounds']['outlier_pts']}"
          f"  D7 offset={good['D7_thalweg']['offset_norm']:.3f} on={good['D7_thalweg']['argmax_on']}"
          f"  D9 ratio={good['D9_ratio']['chiu_mean_ratio']:.3f}")
    rc = good["ref_compare"]
    print(f"  refs: mae_raw={rc['vs_raw']['mae']:.4f} (n={rc['vs_raw']['n']})"
          f"  mae_plat={rc['vs_platform']['mae']:.4f}"
          f"  raw-plat gap median={rc['raw_platform_gap']['bias']:.4f}")
    print("== self-test: violating profile ==")
    rc = bad["ref_compare"]
    print(f"  D1 rate={bad['D1_noslip']['rate']:.2f}"
          f"  D2 resid_norm={bad['D2_wall_decay']['imputed_resid_rms_norm']:.3f}"
          f"  D4 resid_norm={bad['D4_chiu']['resid_rms_norm']:.3f}"
          f"  D6 outliers={bad['D6_bounds']['outlier_pts']}"
          f"  refs: mae_raw={rc['vs_raw']['mae']:.4f} mae_plat={rc['vs_platform']['mae']:.4f}")
    # D8/D10
    g2 = np.sort(rng.choice(60, size=5, replace=False))
    sec2 = RealGapSection("SYN", "D0", "2026-01-01 14:30:00", x, depth,
                          v_true.copy(), g2, np.setdiff1d(np.arange(60), g2),
                          float(x.max() - x.min()), 100.02)
    res = d8_discharge_coherence(
        [sec, sec2],
        {section_key(sec): v_true[sec.gap_idx], section_key(sec2): v_true[g2]},
        dt_max_hours=6.0)
    print(f"  D8 median_rel_dev={res['summary']['median_rel_dev']:.4f} (n={res['summary']['n']})")
    spread = d10_consensus_spread({"a": v_true[gap], "b": v_true[gap] + 0.02, "c": v_true[gap] * 2})
    print(f"  D10 outlier_models={spread['outlier_models']} median_std={spread['median_std']:.4f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", default="data_fetch/aether/data/revision",
                        help="aether 输出目录 (含 sections/*.pkl); 不传 --scan 时仅自测")
    parser.add_argument("--scan", action="store_true", help="扫描真实缺失断面清单")
    parser.add_argument("--out", default="artifacts/real_gap_sections.json")
    parser.add_argument("--min-observed", type=int, default=8)
    parser.add_argument("--max-gap-frac", type=float, default=0.5)
    args = parser.parse_args(argv)

    _selftest()
    if args.scan:
        sections = load_real_gap_sections(args.shards, args.min_observed,
                                          args.max_gap_frac)
        by_station = collections.Counter(s.station for s in sections)
        gaps = [len(s.gap_idx) for s in sections]
        print(f"\n== scan: {len(sections)} real-gap sections across {len(by_station)} stations ==")
        for st, n in by_station.most_common():
            print(f"  {st}: {n}")
        if gaps:
            print(f"  gap lines/section: median={int(np.median(gaps))} p90={int(np.percentile(gaps, 90))}")
        if sections:
            cov = float(np.mean([np.isfinite(s.v_raw[s.gap_idx]).mean() for s in sections]))
            print(f"  raw-ref coverage on gap lines: {cov * 100:.1f}%")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            [{"station": s.station, "device": s.device, "time": s.time,
              "n_gap": len(s.gap_idx), "n_obs": len(s.obs_idx),
              "gap_frac": len(s.gap_idx) / (len(s.gap_idx) + len(s.obs_idx))}
             for s in sections], ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
