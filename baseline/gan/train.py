"""
Conditional GAN Baseline 训练模块 (LSGAN)

本文件实现条件 GAN (Least Squares GAN) 的训练流程。

训练流程:
    1. 更新 Discriminator:
       L_D = 0.5 * MSE(D(real), 1) + 0.5 * MSE(D(fake.detach()), 0)
    2. 更新 Generator:
       L_G = L_rec + lambda_adv * L_adv + lambda_smooth * L_smooth
       L_adv = MSE(D(fake), 1)
       L_rec = MSE(v_fake, target) (带 mask)

损失项说明:
    - L_rec:    重建损失, 保证生成流速接近真实流速 (主损失)
    - L_adv:    对抗损失, 驱动生成器产生更逼真的流速分布
    - L_smooth: 平滑损失, 鼓励相邻 query 点预测平滑

注意: GAN baseline 不使用 depth_velocity_penalty,
      保持生成式 baseline 简洁, 避免过多先验约束。

checkpoint 结构:
    - generator_state_dict:     生成器参数
    - discriminator_state_dict: 判别器参数
    - opt_G_state_dict:         生成器优化器状态
    - opt_D_state_dict:         判别器优化器状态
    - model_config:             模型配置
    - train_config:             训练配置
    - baseline_config:          结构参数 (num_layers, z_dim, lambda_adv)
    - global_stats:             归一化统计量
    - best_val_loss:            最优验证损失
    - artifacts:                产出物路径

用法:
    python -m baseline.gan.train
    python -m baseline.gan.train --epochs 200 --lambda_adv 0.01
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional

from config import ModelConfig, TrainConfig, DataConfig
from data.preprocess import SectionPreprocessor
from data.dataset import SectionDataset, collate_fn
from losses import VelocityLoss
from train import TrainingHistory, StepLogger, FixedMaskDataset, masked_mse_parts
from baseline.gan.model import ConditionalGANGenerator, ConditionalGANDiscriminator

logger = logging.getLogger(__name__)


class GANTrainer:
    """LSGAN 训练管理器

    Args:
        model_cfg:   模型配置
        train_cfg:   训练配置
        data_cfg:    数据配置
        num_layers:  LSTM 编码器层数
        z_dim:       噪声维度
        lambda_adv:  对抗损失权重
        device:      计算设备
    """

    def __init__(
        self,
        model_cfg: Optional[ModelConfig] = None,
        train_cfg: Optional[TrainConfig] = None,
        data_cfg: Optional[DataConfig] = None,
        num_layers: int = 2,
        z_dim: int = 16,
        lambda_adv: float = 0.01,
        device: Optional[str] = None,
    ):
        self.model_cfg = model_cfg or ModelConfig()
        self.train_cfg = train_cfg or TrainConfig()
        self.data_cfg = data_cfg or DataConfig()
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.lambda_adv = lambda_adv
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = ConditionalGANGenerator(
            self.model_cfg, num_layers=num_layers, z_dim=z_dim,
        ).to(self.device)
        self.discriminator = ConditionalGANDiscriminator(
            self.model_cfg, num_layers=num_layers,
        ).to(self.device)

        self.loss_fn = VelocityLoss(
            lambda_smooth=self.train_cfg.lambda_smooth,
            lambda_depth_corr=0.0,
            lambda_bank=0.0,
            lambda_grad=0.0,
        )

        self.opt_G = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.opt_D = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

        self.best_val_loss = float("inf")
        self.best_G_state = None
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

    def fit(self, plot_path: str = "checkpoints/gan_training_curves.png",
            log_path: str = "checkpoints/gan_training_log.json"):
        """执行训练

        Early stopping 基于验证 MSE (z=0 确定性推理),
        不是基于对抗损失。
        """
        if not hasattr(self, "train_samples"):
            self.prepare_data()

        # 取证计数器: 实际优化迭代数 (每 batch 一次 D 更新 + 一次 G 更新 = +1;
        # 不改变训练语义)
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

        for epoch in range(self.train_cfg.max_epochs):
            train_metrics = self._train_epoch(train_loader, epoch)

            val_metrics = None
            if val_loader is not None:
                val_metrics = self._validate(val_loader, epoch)
                logger.info(
                    f"Epoch {epoch+1}/{self.train_cfg.max_epochs} | "
                    f"G_loss={train_metrics['g_loss']:.4f} D_loss={train_metrics['d_loss']:.4f} "
                    f"rec={train_metrics['mse']:.4f} | "
                    f"val_mse={val_metrics['mse']:.4f}"
                )

                # 基于 z=0 验证 MSE 选择最优生成器
                if val_metrics["mse"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["mse"]
                    self.best_G_state = {k: v.clone() for k, v in self.generator.state_dict().items()}
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.train_cfg.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.train_cfg.max_epochs} | "
                    f"G_loss={train_metrics['g_loss']:.4f} D_loss={train_metrics['d_loss']:.4f} "
                    f"rec={train_metrics['mse']:.4f}"
                )

            train_metrics_for_history = {
                "loss": train_metrics["g_loss"],
                "mse": train_metrics["mse"],
            }
            self.history.record(epoch + 1, train_metrics_for_history, val_metrics, 0.0)

        if self.best_G_state is not None:
            self.generator.load_state_dict(self.best_G_state)
            logger.info(f"Restored best generator (val_mse={self.best_val_loss:.4f})")

        self.history.plot(save_path=plot_path)
        self.step_logger.save(log_path)
        self.artifacts = {
            "training_curve": plot_path,
            "step_log": log_path,
        }

    def _compute_masked_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """带 mask 的 MSE"""
        diff = (pred - target) ** 2
        if mask is not None:
            return (diff * mask).sum() / (mask.sum() + 1e-8)
        return diff.mean()

    def _train_epoch(self, loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """训练一个 epoch: 先更新 D, 再更新 G"""
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        total_rec = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            target = batch["target"]
            query_mask = batch.get("query_mask")
            mask_float = query_mask.float() if query_mask is not None else None
            n_valid = mask_float.sum() + 1e-8 if mask_float is not None else 1.0

            # --- Step 1: 更新 Discriminator ---
            z = torch.randn(
                batch["query"].shape[0],
                batch["query"].shape[1],
                self.generator.z_dim,
                device=self.device,
            )
            with torch.no_grad():
                fake_output = self.generator(batch, z=z)
                v_fake_detached = fake_output["v_norm_pred"]

            real_scores = self.discriminator(batch, target)
            fake_scores = self.discriminator(batch, v_fake_detached)

            if mask_float is not None:
                l_d_real = ((real_scores - 1) ** 2 * mask_float).sum() / n_valid
                l_d_fake = (fake_scores ** 2 * mask_float).sum() / n_valid
            else:
                l_d_real = F.mse_loss(real_scores, torch.ones_like(real_scores))
                l_d_fake = F.mse_loss(fake_scores, torch.zeros_like(fake_scores))

            l_d = 0.5 * (l_d_real + l_d_fake)

            self.opt_D.zero_grad()
            l_d.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.opt_D.step()

            # --- Step 2: 更新 Generator ---
            z = torch.randn(
                batch["query"].shape[0],
                batch["query"].shape[1],
                self.generator.z_dim,
                device=self.device,
            )
            fake_output = self.generator(batch, z=z)
            v_fake = fake_output["v_norm_pred"]

            fake_scores_for_g = self.discriminator(batch, v_fake)

            if mask_float is not None:
                l_adv = ((fake_scores_for_g - 1) ** 2 * mask_float).sum() / n_valid
            else:
                l_adv = F.mse_loss(fake_scores_for_g, torch.ones_like(fake_scores_for_g))

            l_rec = self._compute_masked_mse(v_fake, target, mask_float)

            # 平滑损失 (GAN 不使用 depth penalty)
            smooth_dict = self.loss_fn.smooth_loss(v_fake, query_mask)
            if isinstance(smooth_dict, torch.Tensor):
                l_smooth = smooth_dict
            else:
                l_smooth = torch.tensor(0.0, device=self.device)

            l_g = l_rec + self.lambda_adv * l_adv + self.loss_fn.lambda_smooth * l_smooth

            self.opt_G.zero_grad()
            l_g.backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.opt_G.step()
            self.n_opt_steps += 1          # 取证计数 (不改变训练语义)

            total_g_loss += l_g.item()
            total_d_loss += l_d.item()
            total_rec += l_rec.item()
            n_batches += 1

            # 记录真实学习率
            lr_g = self.opt_G.param_groups[0]["lr"]
            lr_d = self.opt_D.param_groups[0]["lr"]
            self.step_logger.log_train(
                epoch=epoch, step_in_epoch=batch_idx, lr=lr_g,
                lr_g=lr_g,
                lr_d=lr_d,
                g_loss=l_g.item(),
                d_loss=l_d.item(),
                rec=l_rec.item(),
                adv=l_adv.item(),
                smooth=l_smooth.item(),
            )

        return {
            "g_loss": total_g_loss / max(n_batches, 1),
            "d_loss": total_d_loss / max(n_batches, 1),
            "mse": total_rec / max(n_batches, 1),
        }

    @torch.no_grad()
    def _validate(self, loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """验证: z=0 确定性推理, 计算 MSE"""
        self.generator.eval()
        sq_sum, n_valid = 0.0, 0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            z = torch.zeros(
                batch["query"].shape[0],
                batch["query"].shape[1],
                self.generator.z_dim,
                device=self.device,
            )
            output = self.generator(batch, z=z)
            query_mask = batch.get("query_mask")
            mask_float = query_mask.float() if query_mask is not None else None

            # 全局 masked MSE (sum/count, 跨家族统一选模口径)
            sq, nv = masked_mse_parts(output["v_norm_pred"], batch["target"], query_mask)
            sq_sum += sq
            n_valid += nv
            n_batches += 1

            self.step_logger.log_val(
                epoch=epoch, step_in_epoch=batch_idx,
                mse=sq / max(nv, 1),
            )

        val_mse = sq_sum / max(n_valid, 1)
        return {
            "loss": val_mse,
            "mse": val_mse,
        }

    def save(self, path: str):
        """保存 checkpoint, 包含结构参数"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "opt_G_state_dict": self.opt_G.state_dict(),
            "opt_D_state_dict": self.opt_D.state_dict(),
            "model_config": self.model_cfg.__dict__,
            "train_config": self.train_cfg.__dict__,
            "baseline_config": {
                "num_layers": self.num_layers,
                "z_dim": self.z_dim,
                "lambda_adv": self.lambda_adv,
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
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.opt_G.load_state_dict(checkpoint["opt_G_state_dict"])
        self.opt_D.load_state_dict(checkpoint["opt_D_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if "global_stats" in checkpoint:
            self.preprocessor.global_stats = checkpoint["global_stats"]
        if "baseline_config" in checkpoint:
            self.num_layers = checkpoint["baseline_config"].get("num_layers", self.num_layers)
            self.z_dim = checkpoint["baseline_config"].get("z_dim", self.z_dim)
            self.lambda_adv = checkpoint["baseline_config"].get("lambda_adv", self.lambda_adv)
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

    parser = argparse.ArgumentParser(description="Conditional GAN Baseline 训练 (LSGAN)")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["stratified", "station", "time"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--lambda_adv", type=float, default=0.01)
    parser.add_argument("--save", type=str, default="checkpoints/gan_baseline.pt")
    parser.add_argument("--plot_path", type=str, default="checkpoints/gan_training_curves.png")
    parser.add_argument("--log_path", type=str, default="checkpoints/gan_training_log.json")
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

    print(f"Model:  Conditional GAN (d_model={model_cfg.d_model}, z_dim={args.z_dim}, lambda_adv={args.lambda_adv})")
    print(f"Train:  epochs={train_cfg.max_epochs}, batch={train_cfg.batch_size}, lr={train_cfg.learning_rate}")
    print(f"Device: {device}")
    print()

    trainer = GANTrainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        num_layers=args.num_layers,
        z_dim=args.z_dim,
        lambda_adv=args.lambda_adv,
        device=device,
    )
    trainer.prepare_data()
    trainer.fit(plot_path=args.plot_path, log_path=args.log_path)
    trainer.save(args.save)


if __name__ == "__main__":
    main()
