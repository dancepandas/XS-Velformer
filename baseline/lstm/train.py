"""
LSTM Baseline 训练模块

本文件实现条件式 Encoder-Decoder LSTM 的训练流程。

模型结构:
    Context [B, N_ctx, 6] -> context_proj -> context_lstm -> context_summary [B, H]
    Query   [B, N_qry, 5] -> query_proj  -> query_lstm  -> query_hidden [B, N_qry, H]
    [query_hidden, context_summary_expand] -> MLP -> v_norm_pred [B, N_qry]

训练策略:
    - 损失函数: MSE + lambda_smooth * smooth + lambda_depth * depth (与主模型一致)
    - 验证: 动态 mask, 固定比例 (0.2~0.3)
    - Early stopping: 基于验证 loss

checkpoint 结构:
    - model_state_dict:     模型参数
    - optimizer_state_dict: 优化器状态
    - model_config:         模型配置
    - train_config:         训练配置
    - baseline_config:      结构参数 (num_layers)
    - global_stats:         归一化统计量
    - best_val_loss:        最优验证损失
    - artifacts:            产出物路径

用法:
    python -m baseline.lstm.train
    python -m baseline.lstm.train --epochs 100 --d_model 256
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional

from config import ModelConfig, TrainConfig, DataConfig
from data.preprocess import SectionPreprocessor
from data.dataset import SectionDataset, collate_fn
from losses import VelocityLoss
from train import TrainingHistory, StepLogger, FixedMaskDataset, masked_mse_parts
from baseline.lstm.model import LSTMBaseline

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """LSTM Baseline 训练管理器

    Args:
        model_cfg:   模型配置
        train_cfg:   训练配置
        data_cfg:    数据配置
        num_layers:  LSTM 层数
        device:      计算设备
    """

    def __init__(
        self,
        model_cfg: Optional[ModelConfig] = None,
        train_cfg: Optional[TrainConfig] = None,
        data_cfg: Optional[DataConfig] = None,
        num_layers: int = 2,
        device: Optional[str] = None,
    ):
        self.model_cfg = model_cfg or ModelConfig()
        self.train_cfg = train_cfg or TrainConfig()
        self.data_cfg = data_cfg or DataConfig()
        self.num_layers = num_layers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMBaseline(self.model_cfg, num_layers=num_layers).to(self.device)
        self.loss_fn = VelocityLoss(
            lambda_smooth=self.train_cfg.lambda_smooth,
            lambda_depth_corr=0.0,
            lambda_bank=0.0,
            lambda_grad=0.0,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.scheduler = None
        self.best_val_loss = float("inf")
        self.best_state_dict = None
        self.patience_counter = 0

        self.preprocessor = SectionPreprocessor()
        self.history = TrainingHistory()
        self.step_logger = StepLogger()
        self.artifacts = {}

    def prepare_data(self):
        """加载并划分数据, 应用归一化"""
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

        self.preprocessor.fit_global_stats(self.train_samples)
        for s in self.train_samples:
            self.preprocessor.apply_global_norm(s)
        for s in self.val_samples:
            self.preprocessor.apply_global_norm(s)
        for s in self.test_samples:
            self.preprocessor.apply_global_norm(s)

    def fit(self, plot_path: str = "checkpoints/lstm_training_curves.png",
            log_path: str = "checkpoints/lstm_training_log.json"):
        """执行训练"""
        if not hasattr(self, "train_samples"):
            self.prepare_data()

        # 取证计数器: 实际优化步数 (每个优化器更新 +1; 不改变训练语义)
        self.n_opt_steps = 0

        train_dataset = SectionDataset(
            self.train_samples,
            mask_ratio_range=tuple(self.train_cfg.mask_ratio_range),
            repeats=self.train_cfg.repeats,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )

        val_loader = None
        if self.val_samples:
            val_dataset = SectionDataset(
                self.val_samples,
                mask_ratio_range=(0.2, 0.3),
                repeats=5,
            )
            # 固定验证遮蔽 (跨家族统一协议)
            val_dataset = FixedMaskDataset(val_dataset, self.data_cfg.random_seed)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.train_cfg.batch_size,
                shuffle=False,
                num_workers=self.train_cfg.num_workers,
                collate_fn=collate_fn,
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

                # 全局 masked MSE 选模 (跨家族统一指标; 复合损失仅记录)
                if val_metrics["mse"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["mse"]
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

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            logger.info(f"Restored best model (val_loss={self.best_val_loss:.4f})")

        self.history.plot(save_path=plot_path)
        self.step_logger.save(log_path)
        self.artifacts = {
            "training_curve": plot_path,
            "step_log": log_path,
        }

    def _build_scheduler(self, total_steps: int):
        """构建学习率调度器"""
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
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(batch)
            loss_dict = self.loss_fn(
                output["v_norm_pred"], batch["target"],
                output["query_mask"],
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
            )

        if self.scheduler is not None and self.train_cfg.scheduler == "cosine":
            self.scheduler.step()

        return {
            "loss": total_loss / max(n_batches, 1),
            "mse": total_mse / max(n_batches, 1),
        }

    @torch.no_grad()
    def _validate(self, loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """验证一个 epoch"""
        self.model.eval()
        total_loss = 0.0
        sq_sum, n_valid = 0.0, 0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(batch)
            loss_dict = self.loss_fn(
                output["v_norm_pred"], batch["target"],
                output["query_mask"],
            )
            total_loss += loss_dict["loss"].item()
            # 全局 masked MSE (sum/count, 与主模型统一口径)
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
            )

        return {
            "loss": total_loss / max(n_batches, 1),
            "mse": sq_sum / max(n_valid, 1),
        }

    def save(self, path: str):
        """保存 checkpoint, 包含结构参数"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_cfg.__dict__,
            "train_config": self.train_cfg.__dict__,
            "baseline_config": {
                "num_layers": self.num_layers,
            },
            "global_stats": self.preprocessor.global_stats,
            "best_val_loss": self.best_val_loss,
            "artifacts": self.artifacts,
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """从 checkpoint 加载, 自动恢复结构参数"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if "global_stats" in checkpoint:
            self.preprocessor.global_stats = checkpoint["global_stats"]
        if "baseline_config" in checkpoint:
            self.num_layers = checkpoint["baseline_config"].get("num_layers", self.num_layers)
        if "artifacts" in checkpoint:
            self.artifacts = checkpoint["artifacts"]
        logger.info(f"Model loaded from {path}")


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="LSTM Baseline 训练")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["stratified", "station", "time"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--save", type=str, default="checkpoints/lstm_baseline.pt")
    parser.add_argument("--plot_path", type=str, default="checkpoints/lstm_training_curves.png")
    parser.add_argument("--log_path", type=str, default="checkpoints/lstm_training_log.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    model_kw = {}
    if args.d_model is not None:
        model_kw["d_model"] = args.d_model

    train_kw = {}
    if args.epochs is not None:
        train_kw["max_epochs"] = args.epochs
    if args.batch_size is not None:
        train_kw["batch_size"] = args.batch_size
    if args.lr is not None:
        train_kw["learning_rate"] = args.lr

    data_kw = {}
    if args.data is not None:
        data_kw["data_path"] = args.data
    if args.split is not None:
        data_kw["split_mode"] = args.split

    model_cfg = ModelConfig(**model_kw)
    train_cfg = TrainConfig(**train_kw)
    data_cfg = DataConfig(**data_kw)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model:  LSTM Baseline (d_model={model_cfg.d_model}, layers={args.num_layers})")
    print(f"Train:  epochs={train_cfg.max_epochs}, batch={train_cfg.batch_size}, lr={train_cfg.learning_rate}")
    print(f"Device: {device}")
    print()

    trainer = LSTMTrainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        num_layers=args.num_layers,
        device=device,
    )
    trainer.prepare_data()
    trainer.fit(plot_path=args.plot_path, log_path=args.log_path)
    trainer.save(args.save)


if __name__ == "__main__":
    main()
