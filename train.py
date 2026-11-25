"""
训练模块 —— Query Transformer 训练循环

本文件实现完整的训练管理流程:
    1. StepLogger:     逐 step 记录训练/验证损失, 保存为 JSON 便于后续绘图
    2. TrainingHistory: 逐 epoch 记录指标并绘制训练曲线
    3. Trainer:        训练管理器, 包含数据准备、训练循环、验证、保存/加载

数据流:
    sections.pkl -> SectionPreprocessor -> SectionDataset -> DataLoader -> Trainer

训练结束后:
    - 自动绘制 loss 曲线 (training_curves.png)
    - 保存逐 step 训练日志 (training_log.json)
    - 保存 checkpoint (包含 model_config, global_stats, artifacts 路径等)
"""

import os
import json
import math
import logging
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from config import ModelConfig, TrainConfig, DataConfig
from data.preprocess import SectionPreprocessor, SectionSample
from data.dataset import SectionDataset, collate_fn
from models.query_transformer import HydroQueryTransformer
from losses import VelocityLoss

logger = logging.getLogger(__name__)


class StepLogger:
    """
    逐 step 记录训练/验证损失, 保存为 JSON 便于后续绘图

    每条记录包含:
        global_step:     全局步数 (从 1 递增, 仅 train 步递增)
        epoch:           当前 epoch 编号 (0-indexed)
        step_in_epoch:   当前 epoch 内的 step 编号
        phase:           "train" 或 "val"
        lr:              当前学习率 (val 阶段为 None)
        以及各损失分量: loss, mse, smooth, depth 等

    用法:
        logger = StepLogger()
        logger.log_train(epoch=0, step_in_epoch=0, lr=0.0002, loss=0.5, mse=0.48)
        logger.log_val(epoch=0, step_in_epoch=0, loss=0.3, mse=0.28)
        logger.save("training_log.json")
    """

    def __init__(self):
        self.records: List[Dict] = []
        self._global_step = 0

    def log_train(self, epoch: int, step_in_epoch: int, lr: float, **kwargs):
        """记录一个训练 step"""
        self._global_step += 1
        self.records.append({
            "global_step": self._global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "phase": "train",
            "lr": lr,
            **{k: float(v) for k, v in kwargs.items()},
        })

    def log_val(self, epoch: int, step_in_epoch: int, **kwargs):
        """记录一个验证 step (不递增 global_step)"""
        self.records.append({
            "global_step": self._global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "phase": "val",
            "lr": None,
            **{k: float(v) for k, v in kwargs.items()},
        })

    def save(self, path: str):
        """将所有记录保存为 JSON"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False)
        logger.info(f"Step log saved to {path} ({len(self.records)} records)")


class TrainingHistory:
    """
    训练过程记录与可视化

    每个 epoch 记录:
    - train_loss, train_mse
    - val_loss, val_mse (如有验证集)
    - lr (当前学习率)

    训练结束后调用 plot() 绘制曲线并保存。
    """

    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.train_mse = []
        self.val_loss = []
        self.val_mse = []
        self.lr_history = []

    def record(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: float = 0.0,
    ):
        """记录一个 epoch 的指标"""
        self.epochs.append(epoch)
        self.train_loss.append(train_metrics["loss"])
        self.train_mse.append(train_metrics["mse"])
        self.val_loss.append(val_metrics["loss"] if val_metrics else None)
        self.val_mse.append(val_metrics["mse"] if val_metrics else None)
        self.lr_history.append(lr)

    def plot(self, save_path: str = "checkpoints/training_curves.png"):
        """绘制训练曲线: loss / mse / lr 三张子图"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = self.epochs

        # --- 子图1: Loss (train + val) ---
        ax = axes[0]
        ax.plot(epochs, self.train_loss, "b-o", markersize=3, label="train_loss")
        val_loss_valid = [(e, v) for e, v in zip(epochs, self.val_loss) if v is not None]
        if val_loss_valid:
            ve, vv = zip(*val_loss_valid)
            ax.plot(ve, vv, "r-o", markersize=3, label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- 子图2: MSE (train + val) ---
        ax = axes[1]
        ax.plot(epochs, self.train_mse, "b-o", markersize=3, label="train_mse")
        val_mse_valid = [(e, v) for e, v in zip(epochs, self.val_mse) if v is not None]
        if val_mse_valid:
            ve, vv = zip(*val_mse_valid)
            ax.plot(ve, vv, "r-o", markersize=3, label="val_mse")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- 子图3: Learning Rate ---
        ax = axes[2]
        ax.plot(epochs, self.lr_history, "g-o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Training curves saved to {save_path}")


class FixedMaskDataset(torch.utils.data.Dataset):
    """Replay validation masks by index without consuming training RNG state."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        state = np.random.get_state()
        try:
            np.random.seed((self.seed + idx) % (2 ** 32))
            return self.dataset[idx]
        finally:
            np.random.set_state(state)


def masked_mse_parts(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> tuple:
    """全局 masked MSE 的累加部件: 返回 (平方和, 有效样本数)

    跨家族统一选模指标的基础: 各训练器在验证循环里累加这两个量,
    最终 mse = 平方和 / 有效样本数, 保证同一聚合口径。
    """
    qm = mask.float()
    diff = (pred - target) ** 2
    return (diff * qm).sum().item(), qm.sum().item()


@contextmanager
def fixed_torch_rng(seed: int):
    """临时固定 torch 全局随机流, 退出时恢复原状态 (上下文管理器)

    用于随机推理的验证/评估重放: ProbSparse 稀疏采样、DDPM 采样等依赖
    全局 RNG, 不固定则验证分数受此前训练 RNG 消耗影响, 跨 epoch 不可比。
    固定后验证分数只随模型权重变化 (协议 docs/revision/rerun-protocol.md §3)。
    """
    rng_state = torch.get_rng_state()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else None)
    torch.manual_seed(seed)
    if cuda_states is not None:
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(rng_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def build_val_loader(val_samples, batch_size: int, num_workers: int,
                     mask_seed: int) -> DataLoader:
    """验证装载器唯一构造路径 (协议 §3 固定验证遮蔽语义)

    固定 mask 比例 (0.2, 0.3) × repeats 5 × FixedMaskDataset(mask_seed) 重放。
    Trainer.fit 与控制元/评估脚本共用本函数, 消除多处独立硬编码的漂移风险
    (Opus 控制元审查两轮均点名)。
    """
    val_dataset = SectionDataset(val_samples, mask_ratio_range=(0.2, 0.3), repeats=5)
    val_dataset = FixedMaskDataset(val_dataset, mask_seed)
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


class Trainer:
    """Hydro-Query Transformer 训练管理器

    负责:
    - 数据加载与划分 (prepare_data)
    - 训练循环 (fit)
    - 验证与 early stopping
    - 模型保存/加载 (save/load)

    checkpoint 包含:
    - model_state_dict:  模型参数
    - optimizer_state_dict: 优化器状态
    - model_config:      模型配置
    - train_config:      训练配置
    - global_stats:      全局归一化统计量
    - best_val_mse:     最优验证 masked MSE (跨家族统一选模指标; 复合损失仅记录不入模)
    - artifacts:         训练产出物路径 (曲线图、step log)
    """

    def __init__(
        self,
        model_cfg: Optional[ModelConfig] = None,
        train_cfg: Optional[TrainConfig] = None,
        data_cfg: Optional[DataConfig] = None,
        device: Optional[str] = None,
    ):
        self.model_cfg = model_cfg or ModelConfig()
        self.train_cfg = train_cfg or TrainConfig()
        self.data_cfg = data_cfg or DataConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HydroQueryTransformer(self.model_cfg).to(self.device)
        self.loss_fn = VelocityLoss(
            lambda_smooth=self.train_cfg.lambda_smooth,
            lambda_depth_corr=self.train_cfg.lambda_depth_corr,
            lambda_bank=self.train_cfg.lambda_bank,
            lambda_bank_zero=self.train_cfg.lambda_bank_zero,
            lambda_grad=self.train_cfg.lambda_grad,
            bank_threshold=self.train_cfg.bank_threshold,
            depth_threshold=self.train_cfg.depth_threshold,
            v_bank_limit=self.train_cfg.v_bank_limit,
            grad_smooth_gamma=self.train_cfg.grad_smooth_gamma,
            bank_zero_tau=self.train_cfg.bank_zero_tau,
            depth_zero_tau=self.train_cfg.depth_zero_tau,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.scheduler = None
        self.best_val_mse = float("inf")
        self.best_state_dict = None
        self.patience_counter = 0

        self.preprocessor = SectionPreprocessor()
        self.history = TrainingHistory()
        self.step_logger = StepLogger()
        self.artifacts = {}

    def prepare_data(self):
        """
        从 sections.pkl 加载变长断面并准备训练/验证数据

        根据 data_cfg.split_mode 选择划分方式:
        - "stratified": 按测站分层随机划分 (每站都有 train/val/test)
        - "station":   按测站划分 (测试跨站泛化能力)
        - "time":      按时间划分 (避免未来数据泄漏)
        - "station_and_time": 跨站 × 跨时双重留出 (修订协议最严格划分)
        """
        samples = self.preprocessor.load_sections(self.data_cfg.data_path)
        logger.info(f"Loaded {len(samples)} sections from {self.data_cfg.data_path}")

        if self.data_cfg.split_mode == "stratified":
            self.train_samples, self.val_samples, self.test_samples = \
                self.preprocessor.split_stratified_by_station(
                    samples,
                    train_ratio=self.data_cfg.train_ratio,
                    val_ratio=self.data_cfg.val_ratio,
                    seed=self.data_cfg.random_seed,
                )
        elif self.data_cfg.split_mode == "station":
            self.train_samples, self.val_samples, self.test_samples = \
                self.preprocessor.split_by_station(
                    samples,
                    train_ratio=self.data_cfg.train_ratio,
                    val_ratio=self.data_cfg.val_ratio,
                )
        elif self.data_cfg.split_mode == "station_and_time":
            self.train_samples, self.val_samples, self.test_samples = \
                self.preprocessor.split_by_station_and_time(
                    samples,
                    train_ratio=self.data_cfg.train_ratio,
                    val_ratio=self.data_cfg.val_ratio,
                    seed=self.data_cfg.random_seed,
                    require_time=self.data_cfg.require_time,
                )
        else:
            self.train_samples, self.val_samples, self.test_samples = \
                self.preprocessor.split_by_time(
                    samples,
                    train_ratio=self.data_cfg.train_ratio,
                    val_ratio=self.data_cfg.val_ratio,
                )

        train_stations = set(s.station_id for s in self.train_samples)
        val_stations = set(s.station_id for s in self.val_samples)
        test_stations = set(s.station_id for s in self.test_samples)
        logger.info(f"Split ({self.data_cfg.split_mode}): "
                     f"train={len(self.train_samples)} [{len(train_stations)} stations], "
                     f"val={len(self.val_samples)} [{len(val_stations)} stations], "
                     f"test={len(self.test_samples)} [{len(test_stations)} stations]")

        # 全局统计量只从训练集拟合, 避免 val/test 泄漏
        self.preprocessor.fit_global_stats(self.train_samples)
        for s in self.train_samples:
            self.preprocessor.apply_global_norm(s)
        for s in self.val_samples:
            self.preprocessor.apply_global_norm(s)
        for s in self.test_samples:
            self.preprocessor.apply_global_norm(s)

        # 原始流速 0 m/s 对应的标准化值, 供 bank_zero_loss 使用
        v_mean = self.preprocessor.global_stats.get("v", {}).get("mean", 0.0)
        v_std = self.preprocessor.global_stats.get("v", {}).get("std", 1.0)
        self.loss_fn.v_zero_norm = (0.0 - v_mean) / v_std

    def fit(self, plot_path: str = "checkpoints/training_curves.png",
            log_path: str = "checkpoints/training_log.json"):
        """
        执行训练

        Args:
            plot_path: 训练曲线保存路径
            log_path:  逐 step 训练日志保存路径
        """
        if not hasattr(self, "train_samples"):
            self.prepare_data()

        # 取证计数器: 实际优化步数 (每个优化器更新 +1; 在 step_logger.log_train
        # 同一批循环内递增 → 与 run_control_cell 由 step_logger train 记录推导
        # 的 n_opt_steps 同源同值)
        self.n_opt_steps = 0

        # 训练集: 动态 mask, repeats 扩增
        train_dataset = SectionDataset(
            self.train_samples,
            mask_ratio_range=tuple(self.train_cfg.mask_ratio_range),
            repeats=self.train_cfg.repeats,
        )
        logger.info(f"Training samples: {len(train_dataset)} "
                     f"({len(self.train_samples)} sections x {self.train_cfg.repeats} repeats)")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )

        # 验证集: 固定 mask 比例, 较少 repeats (构造路径唯一化, 见 build_val_loader)
        val_loader = None
        if self.val_samples:
            val_loader = build_val_loader(
                self.val_samples,
                batch_size=self.train_cfg.batch_size,
                num_workers=self.train_cfg.num_workers,
                mask_seed=self.data_cfg.random_seed,
            )

        total_steps = len(train_loader) * self.train_cfg.max_epochs
        self._build_scheduler(total_steps)

        for epoch in range(self.train_cfg.max_epochs):
            train_metrics = self._train_epoch(train_loader, epoch)
            lr = self.optimizer.param_groups[0]["lr"]

            val_metrics = None
            if val_loader is not None:
                val_metrics = self._validate(val_loader, epoch)
                logger.info(
                    f"Epoch {epoch+1}/{self.train_cfg.max_epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} train_mse={train_metrics['mse']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} val_mse={val_metrics['mse']:.4f} | "
                    f"lr={lr:.2e}"
                )

                # 基于全局验证 masked MSE 选最优模型 (跨家族统一选模指标; 复合损失仅记录)
                # 比较/存储必须同一量纲: 曾出现"MSE 比较、复合 loss 存储"的口径错位
                if val_metrics["mse"] < self.best_val_mse:
                    self.best_val_mse = val_metrics["mse"]
                    self.best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.train_cfg.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.train_cfg.max_epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} train_mse={train_metrics['mse']:.4f} | "
                    f"lr={lr:.2e}"
                )

            self.history.record(epoch + 1, train_metrics, val_metrics, lr)

        # 恢复最优模型参数
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            logger.info(f"Restored best model (val_mse={self.best_val_mse:.4f})")

        # 训练结束后绘制曲线
        self.history.plot(save_path=plot_path)

        # 保存逐 step 训练日志
        self.step_logger.save(log_path)

        # 记录产出物路径, 方便后续追溯
        self.artifacts = {
            "training_curve": plot_path,
            "step_log": log_path,
        }

    def _build_scheduler(self, total_steps: int):
        """构建学习率调度器

        - cosine:  余弦退火 + warmup (默认)
        - onecycle: OneCycleLR
        - None:    不使用调度器
        """
        if self.train_cfg.scheduler == "cosine":
            warmup_epochs = min(5, self.train_cfg.max_epochs // 10)
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                progress = (epoch - warmup_epochs) / max(1, self.train_cfg.max_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress)) * 0.99 + 0.01
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif self.train_cfg.scheduler == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.train_cfg.learning_rate,
                total_steps=total_steps,
            )
        else:
            self.scheduler = None

    def _train_epoch(self, loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """执行一个 epoch 的训练"""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(batch)
            d_norm = batch["query"][:, :, 1]
            bank_norm = batch["query"][:, :, 2]
            grad_norm = batch["query"][:, :, 3]
            loss_dict = self.loss_fn(
                output["v_norm_pred"], batch["target"],
                output["query_mask"], d_norm,
                bank_norm=bank_norm, grad_norm=grad_norm,
            )

            self.optimizer.zero_grad()
            loss_dict["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.n_opt_steps += 1          # 取证计数 (不改变训练语义)
            if self.scheduler is not None and self.train_cfg.scheduler == "onecycle":
                self.scheduler.step()

            total_loss += loss_dict["loss"].item()
            total_mse += loss_dict["mse"].item()
            n_batches += 1

            lr = self.optimizer.param_groups[0]["lr"]
            self.step_logger.log_train(
                epoch=epoch, step_in_epoch=batch_idx, lr=lr,
                loss=loss_dict["loss"].item(),
                mse=loss_dict["mse"].item(),
                smooth=loss_dict["smooth"].item(),
                depth_corr=loss_dict["depth_corr"].item(),
                bank=loss_dict["bank"].item(),
                bank_zero=loss_dict["bank_zero"].item(),
                grad_smooth=loss_dict["grad_smooth"].item(),
            )

        if self.scheduler is not None and self.train_cfg.scheduler == "cosine":
            self.scheduler.step()

        return {
            "loss": total_loss / max(n_batches, 1),
            "mse": total_mse / max(n_batches, 1),
        }

    @torch.no_grad()
    def _validate(self, loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """执行一个 epoch 的验证"""
        self.model.eval()
        total_loss = 0.0
        sq_sum = 0.0
        n_valid = 0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(batch)
            d_norm = batch["query"][:, :, 1]
            bank_norm = batch["query"][:, :, 2]
            grad_norm = batch["query"][:, :, 3]
            loss_dict = self.loss_fn(
                output["v_norm_pred"], batch["target"],
                output["query_mask"], d_norm,
                bank_norm=bank_norm, grad_norm=grad_norm,
            )
            total_loss += loss_dict["loss"].item()
            # 选模指标: 全局 masked MSE (sum/count 口径, 与全部 baseline 家族统一)
            sq, nv = masked_mse_parts(output["v_norm_pred"], batch["target"],
                                      output["query_mask"])
            sq_sum += sq
            n_valid += nv
            n_batches += 1

            self.step_logger.log_val(
                epoch=epoch, step_in_epoch=batch_idx,
                loss=loss_dict["loss"].item(),
                mse=loss_dict["mse"].item(),
                smooth=loss_dict["smooth"].item(),
                depth_corr=loss_dict["depth_corr"].item(),
                bank=loss_dict["bank"].item(),
                bank_zero=loss_dict["bank_zero"].item(),
                grad_smooth=loss_dict["grad_smooth"].item(),
            )

        return {
            "loss": total_loss / max(n_batches, 1),
            "mse": sq_sum / max(n_valid, 1),  # 全局 masked MSE (选模指标)
        }

    def save(self, path: str):
        """
        保存 checkpoint

        包含模型参数、优化器状态、配置、归一化统计量和产出物路径。
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_cfg.__dict__,
            "train_config": self.train_cfg.__dict__,
            "global_stats": self.preprocessor.global_stats,
            "best_val_mse": self.best_val_mse,
            "artifacts": self.artifacts,
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """从 checkpoint 加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # 只认 best_val_mse (统一选模指标)。旧 ReLU 时代 checkpoint 的 best_val_loss
        # 存的是复合损失, 与 MSE 不可比 —— 故意不回退读它, 避免污染新协议的选模。
        self.best_val_mse = checkpoint.get("best_val_mse", float("inf"))
        if "global_stats" in checkpoint:
            self.preprocessor.global_stats = checkpoint["global_stats"]
            v_mean = self.preprocessor.global_stats.get("v", {}).get("mean", 0.0)
            v_std = self.preprocessor.global_stats.get("v", {}).get("std", 1.0)
            self.loss_fn.v_zero_norm = (0.0 - v_mean) / v_std
        if "artifacts" in checkpoint:
            self.artifacts = checkpoint["artifacts"]
        logger.info(f"Model loaded from {path}")
