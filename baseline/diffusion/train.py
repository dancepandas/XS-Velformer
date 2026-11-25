"""
Conditional Diffusion Baseline 训练模块 (DDPM)

本文件实现 DDPM 扩散模型的训练流程:
    - 训练损失: L = MSE(epsilon_pred, epsilon) (带 mask)
    - 验证: 使用 DDPM 采样得到 v_pred, 计算 MSE
    - 保存: checkpoint 包含 denoiser, scheduler_config, baseline_config

训练/验证策略:
    - 训练: 每步随机采样时间步 t, 加噪后预测噪声, 计算损失
    - 验证: 每 5 个 epoch 执行一次完整采样 (慢), 第 1 个 epoch 必做验证
    - Early stopping: 基于验证 MSE, patience 为设定值的 1/5 (因验证频率低)

checkpoint 结构:
    - denoiser_state_dict: 去噪网络参数
    - optimizer_state_dict: 优化器状态
    - model_config:         模型配置
    - train_config:         训练配置
    - scheduler_config:     扩散调度器配置 (num_timesteps, beta_start, beta_end, type)
    - baseline_config:      baseline 结构参数 (num_layers, n_samples)
    - global_stats:         归一化统计量
    - best_val_loss:        最优验证损失
    - artifacts:            产出物路径

用法:
    python -m baseline.diffusion.train
    python -m baseline.diffusion.train --epochs 300 --num_timesteps 200
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
from baseline.diffusion.model import ConditionalDiffusionDenoiser, DiffusionScheduler

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """DDPM 扩散模型训练管理器

    Args:
        model_cfg:      模型配置
        train_cfg:      训练配置
        data_cfg:       数据配置
        num_layers:     LSTM 编码器层数
        num_timesteps:  扩散步数
        n_samples:      验证时采样次数 (取均值)
        device:         计算设备
    """

    def __init__(
        self,
        model_cfg: Optional[ModelConfig] = None,
        train_cfg: Optional[TrainConfig] = None,
        data_cfg: Optional[DataConfig] = None,
        num_layers: int = 2,
        num_timesteps: int = 200,
        n_samples: int = 5,
        device: Optional[str] = None,
    ):
        self.model_cfg = model_cfg or ModelConfig()
        self.train_cfg = train_cfg or TrainConfig()
        self.data_cfg = data_cfg or DataConfig()
        self.num_layers = num_layers
        self.n_samples = n_samples
        # 验证采样噪声种子: 每 epoch 验证用同一噪声序列重放, 分数只随权重变化
        self.val_sample_seed = int(self.data_cfg.random_seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.denoiser = ConditionalDiffusionDenoiser(
            self.model_cfg, num_layers=num_layers,
        ).to(self.device)
        self.scheduler = DiffusionScheduler(num_timesteps=num_timesteps).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

        self.loss_fn = VelocityLoss(
            lambda_smooth=self.train_cfg.lambda_smooth,
            lambda_depth_corr=0.0,
            lambda_bank=0.0,
            lambda_grad=0.0,
        )

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

    def fit(self, plot_path: str = "checkpoints/diffusion_training_curves.png",
            log_path: str = "checkpoints/diffusion_training_log.json"):
        """执行训练

        验证策略: 第 1 个 epoch 必做验证, 之后每 5 个 epoch 验证一次。
        扩散模型验证需要完整采样循环, 比较慢, 所以降低验证频率。
        """
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
                repeats=3,
            )
            # 固定验证遮蔽 (跨家族统一协议); 采样均值验证的频率保持每 5 epoch
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
            # 第 1 个 epoch 必做验证, 之后每 5 个 epoch 验证一次
            should_validate = val_loader is not None and (
                (epoch + 1) == 1 or (epoch + 1) % 5 == 0
            )

            if should_validate:
                val_metrics = self._validate(val_loader, epoch)
                logger.info(
                    f"Epoch {epoch+1}/{self.train_cfg.max_epochs} | "
                    f"noise_mse={train_metrics['mse']:.4f} | "
                    f"val_vel_mse={val_metrics['mse']:.4f}"
                )

                if val_metrics["mse"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["mse"]
                    self.best_state_dict = {k: v.clone() for k, v in self.denoiser.state_dict().items()}
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.train_cfg.early_stopping_patience // 5:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.train_cfg.max_epochs} | "
                    f"noise_mse={train_metrics['mse']:.4f}"
                )

            train_metrics_for_history = {
                "loss": train_metrics["loss"],
                "mse": train_metrics["mse"],
            }
            self.history.record(epoch + 1, train_metrics_for_history, val_metrics, 0.0)

        if self.best_state_dict is not None:
            self.denoiser.load_state_dict(self.best_state_dict)
            logger.info(f"Restored best denoiser (val_mse={self.best_val_loss:.4f})")

        self.history.plot(save_path=plot_path)
        self.step_logger.save(log_path)
        self.artifacts = {
            "training_curve": plot_path,
            "step_log": log_path,
        }

    def _train_epoch(self, loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """训练一个 epoch: 随机采样时间步, 加噪, 预测噪声, 计算损失"""
        self.denoiser.train()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            x_0 = batch["target"]  # [B, N_qry]
            query_mask = batch.get("query_mask")
            B, N_qry = x_0.shape

            # 随机采样时间步
            t = torch.randint(0, self.scheduler.num_timesteps, (B,), device=self.device)
            noise = torch.randn_like(x_0)
            x_t = self.scheduler.q_sample(x_0, t, noise=noise)

            # 预测噪声
            epsilon_pred = self.denoiser(x_t, t, batch)

            # 计算损失 (带 mask)
            diff = (epsilon_pred - noise) ** 2
            if query_mask is not None:
                mask_float = query_mask.float()
                loss = (diff * mask_float).sum() / (mask_float.sum() + 1e-8)
            else:
                loss = diff.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
            self.optimizer.step()
            self.n_opt_steps += 1          # 取证计数 (不改变训练语义)

            total_loss += loss.item()
            total_mse += loss.item()
            n_batches += 1

            self.step_logger.log_train(
                epoch=epoch, step_in_epoch=batch_idx, lr=self.optimizer.param_groups[0]["lr"],
                noise_mse=loss.item(),
            )

        return {
            "loss": total_loss / max(n_batches, 1),
            "mse": total_mse / max(n_batches, 1),
        }

    @torch.no_grad()
    def _validate(self, loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """验证: 完整 DDPM 采样得到 v_pred, 计算 MSE

        采样噪声按固定种子 (self.val_sample_seed) 重放并在结束时恢复全局 RNG 状态:
        DDPM 采样是随机过程, 不固定则早停分数含 Monte Carlo 波动, 跨 epoch 不可比。
        """
        self.denoiser.eval()
        rng_state = torch.get_rng_state()
        cuda_states = (torch.cuda.get_rng_state_all()
                       if self.device.startswith("cuda") else None)
        torch.manual_seed(self.val_sample_seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(self.val_sample_seed)
        try:
            return self._validate_sampled(loader, epoch)
        finally:
            torch.set_rng_state(rng_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _validate_sampled(self, loader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """在已固定的随机流下执行 DDPM 采样验证 (由 _validate 调用)"""
        sq_sum, n_valid = 0.0, 0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            target = batch["target"]
            query_mask = batch.get("query_mask")
            B = target.shape[0]
            N_qry = target.shape[1]

            # 多次采样取均值
            sample_sum = torch.zeros_like(target)
            for _ in range(self.n_samples):
                v_pred = self.scheduler.sample(
                    self.denoiser, batch, (B, N_qry),
                )
                sample_sum += v_pred

            v_pred_mean = sample_sum / self.n_samples

            # 全局 masked MSE (sum/count, 跨家族统一选模口径)
            sq, nv = masked_mse_parts(v_pred_mean, target, query_mask)
            sq_sum += sq
            n_valid += nv
            n_batches += 1

            self.step_logger.log_val(
                epoch=epoch, step_in_epoch=batch_idx,
                vel_mse=sq / max(nv, 1),
            )

        val_mse = sq_sum / max(n_valid, 1)
        return {
            "loss": val_mse,
            "mse": val_mse,
        }

    def save(self, path: str):
        """保存 checkpoint, 包含结构参数和调度器配置"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "denoiser_state_dict": self.denoiser.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_cfg.__dict__,
            "train_config": self.train_cfg.__dict__,
            "scheduler_config": {
                "type": "ddpm",
                "num_timesteps": self.scheduler.num_timesteps,
                "beta_start": self.scheduler.beta_start,
                "beta_end": self.scheduler.beta_end,
            },
            "baseline_config": {
                "num_layers": self.num_layers,
                "n_samples": self.n_samples,
            },
            "global_stats": self.preprocessor.global_stats,
            "best_val_loss": self.best_val_loss,
            "artifacts": self.artifacts,
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """从 checkpoint 加载, 自动恢复调度器配置"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if "global_stats" in checkpoint:
            self.preprocessor.global_stats = checkpoint["global_stats"]
        if "scheduler_config" in checkpoint:
            cfg = checkpoint["scheduler_config"]
            self.scheduler = DiffusionScheduler(
                num_timesteps=cfg.get("num_timesteps", self.scheduler.num_timesteps),
                beta_start=cfg.get("beta_start", 1e-4),
                beta_end=cfg.get("beta_end", 0.02),
                sampler=cfg.get("type", "ddpm"),
            ).to(self.device)
        if "baseline_config" in checkpoint:
            self.num_layers = checkpoint["baseline_config"].get("num_layers", self.num_layers)
            self.n_samples = checkpoint["baseline_config"].get("n_samples", self.n_samples)
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

    parser = argparse.ArgumentParser(description="Conditional Diffusion Baseline 训练 (DDPM)")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["stratified", "station", "time"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--save", type=str, default="checkpoints/diffusion_baseline.pt")
    parser.add_argument("--plot_path", type=str, default="checkpoints/diffusion_training_curves.png")
    parser.add_argument("--log_path", type=str, default="checkpoints/diffusion_training_log.json")
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

    print(f"Model:  Conditional Diffusion DDPM (d_model={model_cfg.d_model}, T={args.num_timesteps}, samples={args.n_samples})")
    print(f"Train:  epochs={train_cfg.max_epochs}, batch={train_cfg.batch_size}, lr={train_cfg.learning_rate}")
    print(f"Device: {device}")
    print()

    trainer = DiffusionTrainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        n_samples=args.n_samples,
        device=device,
    )
    trainer.prepare_data()
    trainer.fit(plot_path=args.plot_path, log_path=args.log_path)
    trainer.save(args.save)


if __name__ == "__main__":
    main()
