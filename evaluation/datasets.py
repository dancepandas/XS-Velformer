"""
FixedTestDataset + fixed_collate_fn

将 fixed_test_cases.pkl 中的 test_case 列表转为 DataLoader batch,
保留元信息 (case_id, strategy, mask_ratio, station_id 等) 用于分组统计。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import Dataset
from typing import Dict, List

from data.dataset import collate_fn


class FixedTestDataset(Dataset):
    def __init__(self, test_cases: List[Dict]):
        self.test_cases = test_cases

    def __len__(self):
        return len(self.test_cases)

    def __getitem__(self, idx: int) -> Dict:
        case = self.test_cases[idx]
        return {
            "context": torch.tensor(case["context"], dtype=torch.float32),
            "query": torch.tensor(case["query"], dtype=torch.float32),
            "target": torch.tensor(case["target"], dtype=torch.float32),
            "case_id": case["case_id"],
            "strategy": case["strategy"],
            "mask_ratio": case["mask_ratio"],
            "station_id": case["station_id"],
            "time": case.get("time", ""),
            "section_length": case["section_length"],
        }


def fixed_collate_fn(batch: List[Dict]) -> Dict:
    tensor_batch = collate_fn([
        {
            "context": item["context"],
            "query": item["query"],
            "target": item["target"],
        }
        for item in batch
    ])
    tensor_batch["case_id"] = [item["case_id"] for item in batch]
    tensor_batch["strategy"] = [item["strategy"] for item in batch]
    tensor_batch["mask_ratio"] = [item["mask_ratio"] for item in batch]
    tensor_batch["station_id"] = [item["station_id"] for item in batch]
    tensor_batch["time"] = [item["time"] for item in batch]
    tensor_batch["section_length"] = [item["section_length"] for item in batch]
    return tensor_batch
