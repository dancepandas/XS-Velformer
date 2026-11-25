"""
固定测试任务生成与保存

从 sections.pkl 加载数据, 按训练一致的划分逻辑得到 test_samples,
为每条 test section 生成固定 mask 的 test_case, 保存为 pickle 文件。

用法:
    python -m evaluation.test_cases \
        --data v_data/processed/sections.pkl \
        --split stratified \
        --seed 42 \
        --output result/fixed_test_cases.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import argparse
import logging
import numpy as np
from typing import List, Dict

from data.preprocess import SectionPreprocessor, SectionSample

logger = logging.getLogger(__name__)

RANDOM_RATIOS = [0.2, 0.3, 0.4]
CONTINUOUS_RATIOS = [0.2, 0.3, 0.4]
EDGE_SIDES_RATIOS = [("left", 0.3), ("right", 0.3)]
SPARSE_RESTRUCT_CONTEXT_RATIOS = [0.1, 0.2, 0.3]
DENSIFY_CONTEXT_RATIOS = [0.2, 0.3, 0.4]


def _build_features(
    sample: SectionSample,
    context_idx: np.ndarray,
    query_idx: np.ndarray,
):
    global_feat = np.array([sample.Z_norm, sample.h_max_norm, sample.W_norm])

    context = np.column_stack([
        sample.x_norm[context_idx],
        sample.d_norm[context_idx],
        sample.bank_norm[context_idx],
        sample.grad_norm[context_idx],
        sample.v_norm[context_idx],
        np.tile(global_feat, (len(context_idx), 1)),
    ])

    query = np.column_stack([
        sample.x_norm[query_idx],
        sample.d_norm[query_idx],
        sample.bank_norm[query_idx],
        sample.grad_norm[query_idx],
        np.tile(global_feat, (len(query_idx), 1)),
    ])

    target = sample.v_norm[query_idx]

    return context, query, target


def _random_mask(K: int, mask_ratio: float, rng: np.random.Generator):
    n_query = max(1, int(K * mask_ratio))
    perm = rng.permutation(K)
    query_idx = np.sort(perm[:n_query])
    context_idx = np.sort(perm[n_query:])
    return context_idx, query_idx


def _continuous_mask(K: int, mask_ratio: float, rng: np.random.Generator):
    block_len = max(1, int(K * mask_ratio))
    start = int(rng.integers(0, max(K - block_len + 1, 1)))
    query_idx = np.arange(start, start + block_len)
    context_idx = np.array([j for j in range(K) if j not in set(query_idx)])
    if len(context_idx) == 0:
        context_idx = np.array([query_idx[0]])
        query_idx = query_idx[1:]
    return context_idx, query_idx


def _edge_mask(K: int, side: str, mask_ratio: float):
    n_query = max(1, int(K * mask_ratio))
    if side == "left":
        query_idx = np.arange(n_query)
    else:
        query_idx = np.arange(K - n_query, K)
    context_idx = np.array([j for j in range(K) if j not in set(query_idx)])
    if len(context_idx) == 0:
        context_idx = np.array([0])
        query_idx = np.array([1])
    return context_idx, query_idx


def _sparse_reconstruct(K: int, context_ratio: float, rng: np.random.Generator):
    n_context = max(3, int(K * context_ratio))
    context_idx = np.sort(rng.choice(K, n_context, replace=False))
    query_idx = np.array([j for j in range(K) if j not in set(context_idx)])
    return context_idx, query_idx


def _densify(K: int, context_ratio: float):
    n_context = max(3, int(K * context_ratio))
    step = max(1, K // n_context)
    context_idx = np.arange(0, K, step)
    if len(context_idx) < 2:
        context_idx = np.array([0, K // 2])
    query_idx = np.array([j for j in range(K) if j not in set(context_idx)])
    return context_idx, query_idx


def generate_test_cases(
    test_samples: List[SectionSample],
    seed: int = 42,
    max_sections: int = 0,
    max_cases_per_section: int = 0,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    test_cases = []

    sections = test_samples
    if max_sections > 0:
        sections = sections[:max_sections]

    for sec_idx, sample in enumerate(sections):
        K = sample.K
        if K < 4:
            continue

        cases_for_section = []
        case_index = 0

        for ratio in RANDOM_RATIOS:
            context_idx, query_idx = _random_mask(K, ratio, rng)
            context, query, target = _build_features(sample, context_idx, query_idx)
            cases_for_section.append({
                "case_id": f"{sample.station_id}_{sec_idx}_random_{ratio:.1f}_{case_index}",
                "station_id": sample.station_id,
                "time": "",
                "strategy": "random",
                "mask_ratio": ratio,
                "context_idx": context_idx.tolist(),
                "query_idx": query_idx.tolist(),
                "context": context,
                "query": query,
                "target": target,
                "section_length": K,
            })
            case_index += 1

        for ratio in CONTINUOUS_RATIOS:
            context_idx, query_idx = _continuous_mask(K, ratio, rng)
            context, query, target = _build_features(sample, context_idx, query_idx)
            cases_for_section.append({
                "case_id": f"{sample.station_id}_{sec_idx}_continuous_{ratio:.1f}_{case_index}",
                "station_id": sample.station_id,
                "time": "",
                "strategy": "continuous",
                "mask_ratio": ratio,
                "context_idx": context_idx.tolist(),
                "query_idx": query_idx.tolist(),
                "context": context,
                "query": query,
                "target": target,
                "section_length": K,
            })
            case_index += 1

        for side, ratio in EDGE_SIDES_RATIOS:
            context_idx, query_idx = _edge_mask(K, side, ratio)
            context, query, target = _build_features(sample, context_idx, query_idx)
            cases_for_section.append({
                "case_id": f"{sample.station_id}_{sec_idx}_edge_{side}_{ratio:.1f}_{case_index}",
                "station_id": sample.station_id,
                "time": "",
                "strategy": "edge",
                "mask_ratio": ratio,
                "context_idx": context_idx.tolist(),
                "query_idx": query_idx.tolist(),
                "context": context,
                "query": query,
                "target": target,
                "section_length": K,
            })
            case_index += 1

        for ctx_ratio in SPARSE_RESTRUCT_CONTEXT_RATIOS:
            context_idx, query_idx = _sparse_reconstruct(K, ctx_ratio, rng)
            mask_ratio = round(1.0 - ctx_ratio, 2)
            context, query, target = _build_features(sample, context_idx, query_idx)
            cases_for_section.append({
                "case_id": f"{sample.station_id}_{sec_idx}_sparse_reconstruct_{mask_ratio:.1f}_{case_index}",
                "station_id": sample.station_id,
                "time": "",
                "strategy": "sparse_reconstruct",
                "mask_ratio": mask_ratio,
                "context_idx": context_idx.tolist(),
                "query_idx": query_idx.tolist(),
                "context": context,
                "query": query,
                "target": target,
                "section_length": K,
            })
            case_index += 1

        for ctx_ratio in DENSIFY_CONTEXT_RATIOS:
            context_idx, query_idx = _densify(K, ctx_ratio)
            mask_ratio = round(1.0 - ctx_ratio, 2)
            context, query, target = _build_features(sample, context_idx, query_idx)
            cases_for_section.append({
                "case_id": f"{sample.station_id}_{sec_idx}_densify_{mask_ratio:.1f}_{case_index}",
                "station_id": sample.station_id,
                "time": "",
                "strategy": "densify",
                "mask_ratio": mask_ratio,
                "context_idx": context_idx.tolist(),
                "query_idx": query_idx.tolist(),
                "context": context,
                "query": query,
                "target": target,
                "section_length": K,
            })
            case_index += 1

        if max_cases_per_section > 0:
            cases_for_section = cases_for_section[:max_cases_per_section]

        test_cases.extend(cases_for_section)

    return test_cases


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="生成固定测试任务")
    parser.add_argument("--data", type=str, default="v_data/processed/sections.pkl")
    parser.add_argument("--split", type=str, default="stratified",
                        choices=["stratified", "station", "time", "station_and_time"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_sections", type=int, default=0)
    parser.add_argument("--max_cases_per_section", type=int, default=0)
    parser.add_argument("--output", type=str, default="result/fixed_test_cases.pkl")
    args = parser.parse_args()

    preprocessor = SectionPreprocessor()
    samples = preprocessor.load_sections(args.data)
    logger.info(f"Loaded {len(samples)} sections from {args.data}")

    if args.split == "stratified":
        train_s, val_s, test_s = preprocessor.split_stratified_by_station(
            samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed,
        )
    elif args.split == "station":
        train_s, val_s, test_s = preprocessor.split_by_station(
            samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        )
    elif args.split == "station_and_time":
        train_s, val_s, test_s = preprocessor.split_by_station_and_time(
            samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            seed=args.seed, require_time=True,
        )
    else:
        train_s, val_s, test_s = preprocessor.split_by_time(
            samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        )

    logger.info(f"Split ({args.split}): train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    preprocessor.fit_global_stats(train_s)
    for s in train_s:
        preprocessor.apply_global_norm(s)
    for s in val_s:
        preprocessor.apply_global_norm(s)
    for s in test_s:
        preprocessor.apply_global_norm(s)

    test_cases = generate_test_cases(
        test_s,
        seed=args.seed,
        max_sections=args.max_sections,
        max_cases_per_section=args.max_cases_per_section,
    )
    logger.info(f"Generated {len(test_cases)} test cases")

    strategies = sorted(set(c["strategy"] for c in test_cases))
    payload = {
        "meta": {
            "data_path": args.data,
            "split_mode": args.split,
            "seed": args.seed,
            "n_cases": len(test_cases),
            "strategies": strategies,
        },
        "global_stats": preprocessor.global_stats,
        "test_cases": test_cases,
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Saved fixed test cases to {args.output}")


if __name__ == "__main__":
    main()
