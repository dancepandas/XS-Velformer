"""
Enhanced traditional baselines for fair comparison with Hydro-QT.

Classes:
    MultiDimIDW: 4D inverse distance weighting (x, d, bank, grad)
    OrdinaryKriging: Ordinary Kriging with x + d covariates

These use the same feature space as Hydro-QT (x_norm, d_norm, bank_norm, grad_norm)
but via non-parametric interpolation, providing a "same-feature" baseline.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from typing import Optional, Tuple


class MultiDimIDW:
    """Multi-dimensional Inverse Distance Weighting.

    Uses 4D distance metric combining lateral position, depth, bank distance,
    and depth gradient:
        dist = sqrt(w_x*(dx)^2 + w_d*(dd)^2 + w_b*(db)^2 + w_g*(dg)^2)

    The weights w_x, w_d, w_b, w_g control the relative influence of each
    dimension. They can be set manually or tuned via cross-validation.

    Power p controls the decay rate: weight = 1 / dist^p.
    """

    def __init__(
        self,
        power: float = 2.0,
        w_x: float = 1.0,
        w_d: float = 0.5,
        w_b: float = 0.5,
        w_g: float = 0.25,
        epsilon: float = 1e-8,
    ):
        self.power = power
        self.w_x = w_x
        self.w_d = w_d
        self.w_b = w_b
        self.w_g = w_g
        self.epsilon = epsilon

    def _weighted_distance(
        self,
        query_points: np.ndarray,   # [M, 4] (x, d, bank, grad)
        context_points: np.ndarray,  # [N, 4]
    ) -> np.ndarray:                # [M, N]
        """Compute weighted Euclidean distance between query and context points."""
        weights = np.array([self.w_x, self.w_d, self.w_b, self.w_g])
        # Scale each dimension by sqrt(w_i)
        q_scaled = query_points * np.sqrt(weights)
        c_scaled = context_points * np.sqrt(weights)
        return cdist(q_scaled, c_scaled, metric='euclidean')

    def interpolate(
        self,
        context_points: np.ndarray,  # [N, 4] (x_norm, d_norm, bank_norm, grad_norm)
        context_values: np.ndarray,  # [N,] velocity
        query_points: np.ndarray,    # [M, 4]
    ) -> np.ndarray:                # [M,] predicted velocity
        """Interpolate values at query positions."""
        if len(context_points) == 0:
            return np.zeros(len(query_points))

        dists = self._weighted_distance(query_points, context_points)
        # Clamp to avoid division by zero at exact match points
        dists = np.maximum(dists, self.epsilon)
        weights = 1.0 / (dists ** self.power)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights @ context_values


class OrdinaryKriging:
    """Ordinary Kriging with x_norm and d_norm as covariates.

    Models the spatial correlation using an exponential variogram:
        gamma(h) = sill * (1 - exp(-h / range)) + nugget

    where h is the Euclidean distance in the (scaled) 2D or 4D feature space.
    """

    def __init__(
        self,
        sill: float = 1.0,
        range_param: float = 0.3,
        nugget: float = 0.01,
        use_4d: bool = True,
        w_x: float = 1.0,
        w_d: float = 0.5,
        w_b: float = 0.5,
        w_g: float = 0.25,
    ):
        self.sill = sill
        self.range_param = range_param
        self.nugget = nugget
        self.use_4d = use_4d
        self.w_x = w_x
        self.w_d = w_d
        self.w_b = w_b
        self.w_g = w_g

    def _variogram(self, h: np.ndarray) -> np.ndarray:
        """Exponential variogram model."""
        return self.sill * (1.0 - np.exp(-h / self.range_param)) + self.nugget

    def _covariance(self, h: np.ndarray) -> np.ndarray:
        """Covariance from variogram: C(h) = sill + nugget - gamma(h)"""
        return self.sill * np.exp(-h / self.range_param)

    def _distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Weighted Euclidean distance matrix between all pairs."""
        if self.use_4d:
            weights = np.array([self.w_x, self.w_d, self.w_b, self.w_g])
            scaled = points * np.sqrt(weights)
        else:
            weights = np.array([self.w_x, self.w_d])
            scaled = points[:, :2] * np.sqrt(weights)
        return cdist(scaled, scaled, metric='euclidean')

    def _cross_distance(
        self, query: np.ndarray, context: np.ndarray
    ) -> np.ndarray:
        """Weighted distance between query and context points."""
        if self.use_4d:
            weights = np.array([self.w_x, self.w_d, self.w_b, self.w_g])
            q = query * np.sqrt(weights)
            c = context * np.sqrt(weights)
        else:
            weights = np.array([self.w_x, self.w_d])
            q = query[:, :2] * np.sqrt(weights)
            c = context[:, :2] * np.sqrt(weights)
        return cdist(q, c, metric='euclidean')

    def interpolate(
        self,
        context_points: np.ndarray,  # [N, D]
        context_values: np.ndarray,  # [N,]
        query_points: np.ndarray,    # [M, D]
    ) -> np.ndarray:                # [M,]
        """Kriging interpolation at query positions."""
        N = len(context_points)
        M = len(query_points)

        if N == 0:
            return np.zeros(M)
        if N == 1:
            return np.full(M, context_values[0])

        # Build kriging matrix: K = [C 1; 1^T 0]
        ctx_dists = self._distance_matrix(context_points)
        K = np.zeros((N + 1, N + 1))
        K[:N, :N] = self._covariance(ctx_dists)
        K[:N, N] = 1.0
        K[N, :N] = 1.0
        K[N, N] = 0.0

        # Add small nugget to diagonal for numerical stability
        K[:N, :N] += np.eye(N) * self.nugget * 1e-3

        # Right-hand side: interpolate each query point
        cross_dists = self._cross_distance(query_points, context_points)
        cross_cov = self._covariance(cross_dists)  # [M, N]

        predictions = np.zeros(M)
        for i in range(M):
            rhs = np.zeros(N + 1)
            rhs[:N] = cross_cov[i]
            rhs[N] = 1.0
            try:
                # 增广矩阵 [C 1; 1^T 0] 是对称不定 KKT 矩阵, 不能按正定 (Cholesky) 解
                w = solve(K, rhs, assume_a='sym')
                predictions[i] = np.dot(w[:N], context_values)
            except (np.linalg.LinAlgError, ValueError):
                # 求解失败回退 IDW (此前误回退为 context 均值, 等价于放弃该基线)
                d = np.maximum(cross_dists[i], 1e-12)
                w_idw = 1.0 / d ** 2
                predictions[i] = float((w_idw * context_values).sum() / w_idw.sum())

        return predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else -999
    bias = float(np.mean(y_pred - y_true))
    return {"rmse": rmse, "mae": mae, "nse": nse, "bias": bias}
