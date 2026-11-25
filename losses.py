"""
损失函数模块 —— L_mse + λ_smooth * L_smooth + λ_corr * L_depth_corr + λ_bank * L_bank + λ_bank_zero * L_bank_zero + λ_grad * L_grad

所有损失在标准化流速空间 (v_norm) 中计算, 不需要反归一化。

六个损失项:
    1. L_mse:        逐点均方误差, 主损失, 保证预测精度
    2. L_smooth:     相邻 query 点预测值的一阶差分平方, 鼓励预测平滑
    3. L_depth_corr: 单断面内流速与水深的负相关惩罚 (per-section), 物理一致性
    4. L_bank:       近岸浅水区域异常高流速惩罚, 物理约束
    5. L_bank_zero:  岸边零速衰减损失, 约束两岸浅水趋近 0 m/s
    6. L_grad:       形态自适应平滑损失, 平缓区域强平滑, 形态突变区域弱平滑

所有损失均使用 query_mask 忽略 padding 位置,
只对有效 query 点计算, 避免不同长度断面的统计偏差。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class VelocityLoss(nn.Module):
    """多目标流速损失函数

    总损失 = L_mse + λ_smooth * L_smooth
                   + λ_corr * L_depth_corr
                   + λ_bank * L_bank
                   + λ_bank_zero * L_bank_zero
                   + λ_grad * L_grad

    默认: λ_smooth=0.05, λ_corr=0.02, λ_bank=0.01, λ_bank_zero=0.02, λ_grad=0.01

    Args:
        lambda_smooth:    平滑损失权重
        lambda_depth_corr: per-section 水深-流速负相关惩罚权重
        lambda_bank:      近岸异常高流速惩罚权重
        lambda_bank_zero: 岸边零速衰减损失权重
        lambda_grad:      形态自适应平滑损失权重
        bank_threshold:   近岸判断阈值 (bank_norm < threshold 视为近岸)
        depth_threshold:  浅水判断阈值 (d_norm < threshold 视为浅水)
        v_bank_limit:     近岸流速上限 (标准化空间)
        grad_smooth_gamma: 形态自适应平滑的梯度衰减系数
        bank_zero_tau:    岸边零速衰减的岸距衰减系数
        depth_zero_tau:   岸边零速衰减的水深衰减系数
        v_zero_norm:      原始流速 0 m/s 对应的标准化值
    """

    def __init__(
        self,
        lambda_smooth: float = 0.05,
        lambda_depth_corr: float = 0.02,
        lambda_bank: float = 0.01,
        lambda_bank_zero: float = 0.02,
        lambda_grad: float = 0.01,
        bank_threshold: float = 0.2,
        depth_threshold: float = 0.3,
        v_bank_limit: float = 0.5,
        grad_smooth_gamma: float = 2.0,
        bank_zero_tau: float = 0.15,
        depth_zero_tau: float = 0.25,
        v_zero_norm: float = 0.0,
        corr_use_softplus: bool = True,
        corr_softplus_beta: float = 5.0,
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_depth_corr = lambda_depth_corr
        self.lambda_bank = lambda_bank
        self.lambda_bank_zero = lambda_bank_zero
        self.lambda_grad = lambda_grad
        self.bank_threshold = bank_threshold
        self.depth_threshold = depth_threshold
        self.v_bank_limit = v_bank_limit
        self.grad_smooth_gamma = grad_smooth_gamma
        self.bank_zero_tau = bank_zero_tau
        self.depth_zero_tau = depth_zero_tau
        self.v_zero_norm = v_zero_norm
        self.corr_use_softplus = corr_use_softplus
        self.corr_softplus_beta = corr_softplus_beta

    def mse_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """带 mask 的均方误差

        只对 mask=True 的位置计算 MSE:
            L_mse = Σ(pred - target)² * mask / Σ(mask)
        """
        diff = (pred - target) ** 2
        if mask is not None:
            return (diff * mask).sum() / (mask.sum() + 1e-8)
        return diff.mean()

    def smooth_loss(self, pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """相邻 query 点的一阶差分平滑损失

        鼓励相邻 query 点的预测值平滑过渡:
            L_smooth = Σ(pred[i+1] - pred[i])² * pair_mask / Σ(pair_mask)
        """
        if pred.shape[-1] < 2:
            return torch.tensor(0.0, device=pred.device)
        if mask is not None:
            valid = mask.float()
            if valid.sum() < 2:
                return torch.tensor(0.0, device=pred.device)
            diff = (pred[..., 1:] - pred[..., :-1]) ** 2
            pair_mask = valid[..., 1:] * valid[..., :-1]
            return (diff * pair_mask).sum() / (pair_mask.sum() + 1e-8)
        diff = (pred[..., 1:] - pred[..., :-1]) ** 2
        return diff.mean()

    def depth_corr_per_section_loss(
        self,
        pred: torch.Tensor,
        d_norm: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-section 水深-流速负相关惩罚

        对每条断面单独计算预测流速与水深的 Pearson 相关系数,
        只惩罚明显负相关, 不强制相关性必须很高。
        batch 损失 = mean(relu(-corr_per_section))

        Args:
            pred:   [B, N_qry] 预测流速
            d_norm: [B, N_qry] 归一化水深
            mask:   [B, N_qry] bool, True=有效
        """
        B = pred.shape[0]
        losses = []
        eps = 1e-6

        for b in range(B):
            if mask is not None:
                valid = mask[b]
                if valid.sum() < 3:
                    continue
                v = pred[b, valid]
                d = d_norm[b, valid]
            else:
                v = pred[b]
                d = d_norm[b]
                if v.shape[0] < 3:
                    continue

            if v.std() < eps or d.std() < eps:
                continue

            v_c = v - v.mean()
            d_c = d - d.mean()
            corr = (v_c * d_c).mean() / (v.std() * d.std() + eps)
            if self.corr_use_softplus:
                # SoftPlus: smooth, continuous gradient even when r >= 0
                # beta=5 gives sharp transition near 0, approximating ReLU with continuous gradient
                losses.append(torch.nn.functional.softplus(-corr, beta=self.corr_softplus_beta))
            else:
                # Legacy ReLU: gradient cliff at r >= 0
                losses.append(torch.relu(-corr))

        if losses:
            return torch.stack(losses).mean()
        return pred.new_tensor(0.0)

    def bank_velocity_loss(
        self,
        pred: torch.Tensor,
        bank_norm: torch.Tensor,
        d_norm: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """近岸浅水异常高流速惩罚

        对 bank_norm < threshold 且 d_norm < depth_threshold 的 query 点,
        惩罚明显偏高的预测流速。

        L_bank = mean(relu(v_pred - v_bank_limit)^2 over bank_mask)

        Args:
            pred:      [B, N_qry] 预测流速
            bank_norm: [B, N_qry] 归一化岸距
            d_norm:    [B, N_qry] 归一化水深
            mask:      [B, N_qry] bool, True=有效
        """
        bank_mask = (bank_norm < self.bank_threshold) & (d_norm < self.depth_threshold)
        if mask is not None:
            bank_mask = bank_mask & mask

        if bank_mask.sum() < 1:
            return pred.new_tensor(0.0)

        excess = torch.relu(pred - self.v_bank_limit) ** 2
        return (excess * bank_mask.float()).sum() / (bank_mask.float().sum() + 1e-8)

    def bank_zero_loss(
        self,
        pred: torch.Tensor,
        bank_norm: torch.Tensor,
        d_norm: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """岸边零速衰减损失

        约束两岸浅水区域预测流速趋近 0 m/s。
        bank_norm 越小 (越近岸)、d_norm 越小 (越浅), 权重越大。
        离岸或深水区域权重快速衰减, 不干扰主槽预测。

        weight = exp(-bank_norm / bank_zero_tau) * exp(-d_norm / depth_zero_tau)
        L_bank_zero = sum(weight * (pred - v_zero_norm)^2 * mask) / sum(weight * mask)

        Args:
            pred:      [B, N_qry] 预测流速 (标准化空间)
            bank_norm: [B, N_qry] 归一化岸距
            d_norm:    [B, N_qry] 归一化水深
            mask:      [B, N_qry] bool, True=有效
        """
        bank_weight = torch.exp(-bank_norm / self.bank_zero_tau)
        depth_weight = torch.exp(-d_norm / self.depth_zero_tau)
        weight = bank_weight * depth_weight

        diff_sq = (pred - self.v_zero_norm) ** 2

        if mask is not None:
            mask_float = mask.float()
            weighted_sum = (diff_sq * weight * mask_float).sum()
            weight_sum = (weight * mask_float).sum()
        else:
            weighted_sum = (diff_sq * weight).sum()
            weight_sum = weight.sum()

        if weight_sum < 1e-8:
            return pred.new_tensor(0.0)

        return weighted_sum / (weight_sum + 1e-8)

    def gradient_adaptive_smooth_loss(
        self,
        pred: torch.Tensor,
        grad_norm: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """形态自适应平滑损失

        平滑强度随局部水深梯度调整:
        - 梯度小的平缓区域: 强平滑
        - 梯度大的形态突变区域: 弱平滑, 允许合理变化

        w_i = exp(-γ * max(grad_i, grad_{i+1}))
        L_grad = mean(w_i * (pred[i+1] - pred[i])²)

        Args:
            pred:      [B, N_qry] 预测流速
            grad_norm: [B, N_qry] 归一化局部水深梯度
            mask:      [B, N_qry] bool, True=有效
        """
        if pred.shape[-1] < 2:
            return pred.new_tensor(0.0)

        diff_sq = (pred[..., 1:] - pred[..., :-1]) ** 2

        # 相邻点取较大梯度
        grad_max = torch.max(grad_norm[..., 1:], grad_norm[..., :-1])
        weight = torch.exp(-self.grad_smooth_gamma * grad_max)

        if mask is not None:
            valid = mask.float()
            pair_mask = valid[..., 1:] * valid[..., :-1]
            if pair_mask.sum() < 1:
                return pred.new_tensor(0.0)
            return (diff_sq * weight * pair_mask).sum() / (pair_mask.sum() + 1e-8)

        return (diff_sq * weight).mean()

    def forward(
        self,
        v_norm_pred: torch.Tensor,
        target: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        d_norm: Optional[torch.Tensor] = None,
        bank_norm: Optional[torch.Tensor] = None,
        grad_norm: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            v_norm_pred: [B, N_qry] 预测的标准化流速
            target:      [B, N_qry] 真实的标准化流速
            query_mask:  [B, N_qry] bool, True=有效
            d_norm:      [B, N_qry] 归一化水深 (用于 depth_corr 和 bank loss)
            bank_norm:   [B, N_qry] 归一化岸距 (用于 bank loss)
            grad_norm:   [B, N_qry] 归一化局部水深梯度 (用于 grad smooth loss)

        Returns:
            loss:        总损失 (标量, 可反向传播)
            mse:         MSE 损失 (detached)
            smooth:      平滑损失 (detached)
            depth_corr:  水深相关性损失 (detached)
            bank:        近岸损失 (detached)
            bank_zero:   岸边零速损失 (detached)
            grad_smooth: 形态自适应平滑损失 (detached)
        """
        l_mse = self.mse_loss(v_norm_pred, target, query_mask)
        l_smooth = self.smooth_loss(v_norm_pred, query_mask)
        total = l_mse + self.lambda_smooth * l_smooth

        l_depth_corr = v_norm_pred.new_tensor(0.0)
        if d_norm is not None and self.lambda_depth_corr > 0:
            l_depth_corr = self.depth_corr_per_section_loss(v_norm_pred, d_norm, query_mask)
            total = total + self.lambda_depth_corr * l_depth_corr

        l_bank = v_norm_pred.new_tensor(0.0)
        if bank_norm is not None and d_norm is not None and self.lambda_bank > 0:
            l_bank = self.bank_velocity_loss(v_norm_pred, bank_norm, d_norm, query_mask)
            total = total + self.lambda_bank * l_bank

        l_bank_zero = v_norm_pred.new_tensor(0.0)
        if bank_norm is not None and d_norm is not None and self.lambda_bank_zero > 0:
            l_bank_zero = self.bank_zero_loss(v_norm_pred, bank_norm, d_norm, query_mask)
            total = total + self.lambda_bank_zero * l_bank_zero

        l_grad = v_norm_pred.new_tensor(0.0)
        if grad_norm is not None and self.lambda_grad > 0:
            l_grad = self.gradient_adaptive_smooth_loss(v_norm_pred, grad_norm, query_mask)
            total = total + self.lambda_grad * l_grad

        return {
            "loss": total,
            "mse": l_mse.detach(),
            "smooth": l_smooth.detach(),
            "depth_corr": l_depth_corr.detach(),
            "bank": l_bank.detach(),
            "bank_zero": l_bank_zero.detach(),
            "grad_smooth": l_grad.detach(),
        }
