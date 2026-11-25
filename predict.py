"""
推理模块 —— Hydro-Query Transformer 统一推理接口

三种推理场景 (共享同一个模型):
    - 缺失插补: context=已知点, query=缺失位置
    - 标准断面重构: context=已知点, query=标准网格点
    - 测线加密: context=稀疏实测点, query=高密度目标点

特征结构:
    Context 8 维: [x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm]
    Query   7 维: [x_norm, d_norm, bank_norm, grad_norm, Z_norm, h_max_norm, W_norm]
"""

import numpy as np
import torch
from typing import Dict, Optional
from models.query_transformer import HydroQueryTransformer
from data.preprocess import compute_bank_and_grad_features
from config import ModelConfig


class VelocityPredictor:
    """河道断面流速统一推理器"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_cfg: Optional[ModelConfig] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cfg = model_cfg or ModelConfig()
        self.model = HydroQueryTransformer(self.model_cfg).to(self.device)
        self.global_stats: Dict[str, Dict[str, float]] = {}
        if model_path:
            self.load(model_path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        if "global_stats" in checkpoint:
            self.global_stats = checkpoint["global_stats"]

    def predict_velocity(
        self,
        context_points: np.ndarray,
        query_points: np.ndarray,
        section_features: Dict[str, float],
    ) -> np.ndarray:
        """
        统一流速预测接口

        Args:
            context_points: [N, 3] -> (x_norm, d_norm, v_raw)
            query_points:   [M, 2] -> (x_norm, d_norm)
            section_features: {Z_norm, h_max_norm, W_norm}

        Returns:
            [M] 预测的原始流速 (m/s)
        """
        global_feat = np.array([
            section_features.get("Z_norm", 0.0),
            section_features.get("h_max_norm", 0.0),
            section_features.get("W_norm", 0.0),
        ])

        v_mean = self.global_stats.get("v", {}).get("mean", 0.0)
        v_std = self.global_stats.get("v", {}).get("std", 1.0)
        v_norm_context = (context_points[:, 2] - v_mean) / v_std

        # 基于完整 context 的 x_norm 和 d_norm 计算局部形态特征
        all_x = np.concatenate([context_points[:, 0], query_points[:, 0]])
        all_d = np.concatenate([context_points[:, 1], query_points[:, 1]])
        all_bank, all_grad = compute_bank_and_grad_features(
            np.sort(all_x), np.interp(np.sort(all_x), np.sort(all_x), all_d[np.argsort(all_x)])
        )
        # 按原始顺序恢复
        sort_idx = np.argsort(all_x)
        unsort_idx = np.argsort(sort_idx)
        all_bank = all_bank[unsort_idx]
        all_grad = all_grad[unsort_idx]

        ctx_bank = all_bank[:len(context_points)]
        ctx_grad = all_grad[:len(context_points)]
        qry_bank = all_bank[len(context_points):]
        qry_grad = all_grad[len(context_points):]

        N = len(context_points)
        M = len(query_points)

        context = np.column_stack([
            context_points[:, 0],
            context_points[:, 1],
            ctx_bank,
            ctx_grad,
            v_norm_context,
            np.tile(global_feat, (N, 1)),
        ])
        query = np.column_stack([
            query_points[:, 0],
            query_points[:, 1],
            qry_bank,
            qry_grad,
            np.tile(global_feat, (M, 1)),
        ])

        batch = {
            "context": torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device),
            "context_mask": torch.ones(1, N, dtype=torch.bool).to(self.device),
            "query": torch.tensor(query, dtype=torch.float32).unsqueeze(0).to(self.device),
            "query_mask": torch.ones(1, M, dtype=torch.bool).to(self.device),
        }

        with torch.no_grad():
            output = self.model(batch)
        v_norm_pred = output["v_norm_pred"].cpu().numpy()[0]

        v_pred = v_norm_pred * v_std + v_mean

        return v_pred

    def impute_missing(
        self,
        known_points: np.ndarray,
        missing_positions: np.ndarray,
        section_features: Dict[str, float],
    ) -> np.ndarray:
        """
        缺失插补 (§7.1)

        Args:
            known_points:     [N, 3] -> (x_norm, d_norm, v_raw)
            missing_positions: [M, 2] -> (x_norm, d_norm)
        Returns:
            [M] 预测流速
        """
        return self.predict_velocity(known_points, missing_positions, section_features)

    def reconstruct_standard_section(
        self,
        known_points: np.ndarray,
        n_lines: int = 100,
        section_features: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        标准断面重构 (§7.2)

        Args:
            known_points: [N, 3] -> (x_norm, d_norm, v_raw)
            n_lines: 标准测线数量
        Returns:
            [n_lines] 预测流速
        """
        x_targets = np.linspace(0, 1, n_lines)
        d_targets = np.interp(x_targets, known_points[:, 0], known_points[:, 1])
        query_points = np.column_stack([x_targets, d_targets])
        return self.predict_velocity(known_points, query_points, section_features or {})

    def densify_section(
        self,
        known_points: np.ndarray,
        n_target_lines: int = 200,
        section_features: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        测线加密 (§7.3)

        Args:
            known_points: [N, 3] -> (x_norm, d_norm, v_raw)
            n_target_lines: 加密后测线数量
        Returns:
            [n_target_lines] 预测流速
        """
        return self.reconstruct_standard_section(
            known_points, n_target_lines, section_features,
        )
