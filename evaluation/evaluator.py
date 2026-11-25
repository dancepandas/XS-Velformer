"""
统一评估器

遍历固定测试集, 调用模型预测, 汇总指标:
    - overall: 全局汇总
    - by_strategy: 按任务类型汇总
    - by_mask_ratio: 按缺失比例汇总
    - by_station: 按测站汇总
    - per_case: 每个 test_case 的详细预测与指标 (用于后续绘图)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from collections import defaultdict

from evaluation.datasets import FixedTestDataset, fixed_collate_fn
from evaluation.metrics import MetricsAccumulator, compute_all_metrics
from evaluation.model_adapters import BaseModelAdapter

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        test_cases_path: str,
        adapter: BaseModelAdapter,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.adapter = adapter
        self.batch_size = batch_size
        self.device = device

        with open(test_cases_path, "rb") as f:
            payload = pickle.load(f)
        self.test_cases = payload["test_cases"]
        self.global_stats = payload.get("global_stats", {})
        self.meta = payload.get("meta", {})

        self.v_mean = self.global_stats.get("v", {}).get("mean", 0.0)
        self.v_std = self.global_stats.get("v", {}).get("std", 1.0)

    def run(self) -> Dict:
        dataset = FixedTestDataset(self.test_cases)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=fixed_collate_fn,
        )

        overall_acc = MetricsAccumulator(v_mean=self.v_mean, v_std=self.v_std)
        by_strategy_accs: Dict[str, MetricsAccumulator] = {}
        by_mask_ratio_accs: Dict[str, MetricsAccumulator] = {}
        by_station_accs: Dict[str, MetricsAccumulator] = {}

        per_case_results: List[Dict] = []
        n_cases = 0

        for batch in loader:
            pred_norm = self.adapter.predict(batch)
            target = batch["target"]
            query_mask = batch["query_mask"]

            B = target.shape[0]
            for i in range(B):
                mask_i = query_mask[i].cpu().numpy()
                n_valid = int(mask_i.sum())
                if n_valid < 1:
                    continue

                pred_i = pred_norm[i].cpu().numpy()
                target_i = target[i].cpu().numpy()

                valid_pred = pred_i[mask_i.astype(bool)]
                valid_target = target_i[mask_i.astype(bool)]
                full_mask = np.ones(n_valid, dtype=np.float32)

                strategy = batch["strategy"][i]
                mask_ratio = batch["mask_ratio"][i]
                station_id = batch["station_id"][i]
                case_id = batch["case_id"][i]
                section_length = batch["section_length"][i]

                overall_acc.update(valid_pred, valid_target, full_mask)

                if strategy not in by_strategy_accs:
                    by_strategy_accs[strategy] = MetricsAccumulator(v_mean=self.v_mean, v_std=self.v_std)
                by_strategy_accs[strategy].update(valid_pred, valid_target, full_mask)

                mr_key = str(mask_ratio)
                if mr_key not in by_mask_ratio_accs:
                    by_mask_ratio_accs[mr_key] = MetricsAccumulator(v_mean=self.v_mean, v_std=self.v_std)
                by_mask_ratio_accs[mr_key].update(valid_pred, valid_target, full_mask)

                if station_id not in by_station_accs:
                    by_station_accs[station_id] = MetricsAccumulator(v_mean=self.v_mean, v_std=self.v_std)
                by_station_accs[station_id].update(valid_pred, valid_target, full_mask)

                case_metrics = compute_all_metrics(
                    valid_pred, valid_target, full_mask,
                    v_mean=self.v_mean, v_std=self.v_std,
                )

                per_case_results.append({
                    "case_id": case_id,
                    "strategy": strategy,
                    "mask_ratio": mask_ratio,
                    "station_id": station_id,
                    "section_length": section_length,
                    "n_query_points": n_valid,
                    "pred_norm": valid_pred.tolist(),
                    "target_norm": valid_target.tolist(),
                    **{k: case_metrics[k] for k in case_metrics},
                })

                n_cases += 1

        result = {
            "overall": overall_acc.compute(),
            "by_strategy": {k: v.compute() for k, v in by_strategy_accs.items()},
            "by_mask_ratio": {k: v.compute() for k, v in by_mask_ratio_accs.items()},
            "by_station": {k: v.compute() for k, v in by_station_accs.items()},
            "per_case": per_case_results,
            "n_cases": n_cases,
        }

        return result
