"""
统一模型适配器

不同模型的 checkpoint 和推理方式不同, 通过适配器统一接口:
    - load(checkpoint_path, device): 加载模型, 自动读取 checkpoint 中的结构参数
    - predict(batch): 返回 v_norm_pred

支持的模型类型:
    - hydro_qt:             Hydro-Query Transformer 主模型
    - query_transformer:    同上 (兼容旧名)
    - lstm:                 LSTM Baseline
    - gan_z0:              GAN z=0 确定性推理
    - gan_mean5:           GAN 5 次采样取均值
    - diffusion_mean5:     Diffusion DDPM 5 次采样取均值
    - diffusion_ddpm_mean5: 同上 (显式命名别名)
    - nearest_x:           按 x_norm 最近邻 (传统插值, 无 checkpoint)
    - linear_x:            基于 x_norm 线性插值 (传统插值, 无 checkpoint)
    - idw_x:               按横向距离 IDW (传统插值, 无 checkpoint)
    - idw_xd:              按 [x_norm, d_norm] 二维 IDW (传统插值, 无 checkpoint)

适配器会自动从 checkpoint 中读取 baseline_config/scheduler_config,
无需手动指定 num_layers/z_dim/num_timesteps 等结构参数。

传统插值基线不需要 checkpoint, requires_checkpoint = False。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from abc import ABC, abstractmethod

from config import ModelConfig
from train import fixed_torch_rng
from models.query_transformer import HydroQueryTransformer
from baseline.lstm.model import LSTMBaseline
from baseline.gan.model import ConditionalGANGenerator
from baseline.diffusion.model import ConditionalDiffusionDenoiser, DiffusionScheduler
from baseline.tft.model import TFTBaseline
from baseline.informer.model import InformerBaseline


class BaseModelAdapter(ABC):
    """模型适配器基类

    子类必须实现 load() 和 predict() 方法。
    load() 从 checkpoint 加载模型, predict() 执行推理。

    Attributes:
        requires_checkpoint: 是否需要 checkpoint 文件, 传统插值基线设为 False
    """

    requires_checkpoint: bool = True

    def __init__(self, device: str = "cpu", eval_seed: int = 20261112):
        self.device = device
        # 随机推理适配器 (Informer ProbSparse / Diffusion DDPM / GAN 采样) 的
        # 评估种子: predict 在固定随机流下重放, 同权重同配置重复评估结果一致
        # (协议 docs/revision/rerun-protocol.md §3); 评估入口落盘此值
        self.eval_seed = eval_seed

    def _fixed_rng(self):
        """随机推理的固定评估随机流上下文 (可复现评估)"""
        return fixed_torch_rng(self.eval_seed)

    @abstractmethod
    def load(self, checkpoint_path: str):
        """从 checkpoint 加载模型"""
        pass

    @abstractmethod
    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行推理, 返回 v_norm_pred [B, N_qry]"""
        pass

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将 batch 中所有张量移到指定设备"""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class HydroQueryTransformerAdapter(BaseModelAdapter):
    """HydroQueryTransformer 主模型适配器

    从 checkpoint 读取 model_config 恢复模型结构。
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.model: Optional[HydroQueryTransformer] = None

    def load(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})
        cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
        self.model = HydroQueryTransformer(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._batch_to_device(batch)
        output = self.model(batch)
        return output["v_norm_pred"]

class LSTMAdapter(BaseModelAdapter):
    """LSTM Baseline 适配器

    自动从 checkpoint 的 baseline_config 读取 num_layers。
    """

    def __init__(self, device: str = "cpu", num_layers: int = 2):
        super().__init__(device)
        self.num_layers = num_layers
        self.model: Optional[LSTMBaseline] = None

    def load(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})
        cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
        # 自动读取 baseline_config 中的结构参数
        baseline_cfg = ckpt.get("baseline_config", {})
        self.num_layers = baseline_cfg.get("num_layers", self.num_layers)
        self.model = LSTMBaseline(cfg, num_layers=self.num_layers).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._batch_to_device(batch)
        output = self.model(batch)
        return output["v_norm_pred"]


class GANZ0Adapter(BaseModelAdapter):
    """GAN z=0 确定性推理适配器

    自动从 checkpoint 的 baseline_config 读取 num_layers 和 z_dim。
    """

    def __init__(self, device: str = "cpu", num_layers: int = 2, z_dim: int = 16):
        super().__init__(device)
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.model: Optional[ConditionalGANGenerator] = None

    def load(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})
        cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
        # 自动读取 baseline_config
        baseline_cfg = ckpt.get("baseline_config", {})
        self.num_layers = baseline_cfg.get("num_layers", self.num_layers)
        self.z_dim = baseline_cfg.get("z_dim", self.z_dim)
        self.model = ConditionalGANGenerator(cfg, num_layers=self.num_layers, z_dim=self.z_dim).to(self.device)
        self.model.load_state_dict(ckpt["generator_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._batch_to_device(batch)
        B, N_qry, _ = batch["query"].shape
        z = torch.zeros(B, N_qry, self.z_dim, device=self.device)
        output = self.model(batch, z=z)
        return output["v_norm_pred"]


class GANMeanAdapter(BaseModelAdapter):
    """GAN 多次采样取均值适配器

    自动从 checkpoint 的 baseline_config 读取 num_layers, z_dim, n_samples。
    """

    def __init__(self, device: str = "cpu", num_layers: int = 2, z_dim: int = 16, n_samples: int = 5,
                 eval_seed: int = 20261112):
        super().__init__(device, eval_seed)
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.n_samples = n_samples
        self.model: Optional[ConditionalGANGenerator] = None

    def load(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})
        cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
        # 自动读取 baseline_config
        baseline_cfg = ckpt.get("baseline_config", {})
        self.num_layers = baseline_cfg.get("num_layers", self.num_layers)
        self.z_dim = baseline_cfg.get("z_dim", self.z_dim)
        self.n_samples = baseline_cfg.get("n_samples", self.n_samples)
        self.model = ConditionalGANGenerator(cfg, num_layers=self.num_layers, z_dim=self.z_dim).to(self.device)
        self.model.load_state_dict(ckpt["generator_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._batch_to_device(batch)
        with self._fixed_rng():
            B, N_qry, _ = batch["query"].shape
            preds = []
            for _ in range(self.n_samples):
                output = self.model(batch)
                preds.append(output["v_norm_pred"])
        return torch.stack(preds).mean(dim=0)

    def predict_with_std(self, batch: Dict[str, torch.Tensor]):
        """预测均值和标准差 (用于不确定性分析)"""
        batch = self._batch_to_device(batch)
        with self._fixed_rng():
            B, N_qry, _ = batch["query"].shape
            preds = []
            for _ in range(self.n_samples):
                output = self.model(batch)
                preds.append(output["v_norm_pred"])
        stacked = torch.stack(preds)
        return stacked.mean(dim=0), stacked.std(dim=0)


class DiffusionMeanAdapter(BaseModelAdapter):
    """Diffusion DDPM 多次采样取均值适配器

    自动从 checkpoint 的 baseline_config/scheduler_config 读取结构参数。
    """

    def __init__(
        self,
        device: str = "cpu",
        num_layers: int = 2,
        n_samples: int = 5,
        num_timesteps: int = 200,
        eval_seed: int = 20261112,
    ):
        super().__init__(device, eval_seed)
        self.num_layers = num_layers
        self.n_samples = n_samples
        self.num_timesteps = num_timesteps
        self.model: Optional[ConditionalDiffusionDenoiser] = None
        self.scheduler: Optional[DiffusionScheduler] = None

    def load(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})
        cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
        # 自动读取 baseline_config
        baseline_cfg = ckpt.get("baseline_config", {})
        self.num_layers = baseline_cfg.get("num_layers", self.num_layers)
        self.n_samples = baseline_cfg.get("n_samples", self.n_samples)
        # 加载去噪网络
        self.model = ConditionalDiffusionDenoiser(cfg, num_layers=self.num_layers).to(self.device)
        self.model.load_state_dict(ckpt["denoiser_state_dict"])
        self.model.eval()

        # 自动读取 scheduler_config
        sched_cfg = ckpt.get("scheduler_config", {})
        self.num_timesteps = sched_cfg.get("num_timesteps", self.num_timesteps)
        self.scheduler = DiffusionScheduler(
            num_timesteps=self.num_timesteps,
            beta_start=sched_cfg.get("beta_start", 1e-4),
            beta_end=sched_cfg.get("beta_end", 0.02),
            sampler=sched_cfg.get("type", "ddpm"),
        ).to(self.device)

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._batch_to_device(batch)
        with self._fixed_rng():
            B, N_qry = batch["target"].shape
            preds = []
            for _ in range(self.n_samples):
                v_pred = self.scheduler.sample(self.model, batch, (B, N_qry))
                preds.append(v_pred)
        return torch.stack(preds).mean(dim=0)

    def predict_with_std(self, batch: Dict[str, torch.Tensor]):
        """预测均值和标准差 (用于不确定性分析)"""
        batch = self._batch_to_device(batch)
        with self._fixed_rng():
            B, N_qry = batch["target"].shape
            preds = []
            for _ in range(self.n_samples):
                v_pred = self.scheduler.sample(self.model, batch, (B, N_qry))
                preds.append(v_pred)
        stacked = torch.stack(preds)
        return stacked.mean(dim=0), stacked.std(dim=0)


class TFTAdapter(BaseModelAdapter):
    """TFT Baseline 适配器"""

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.model: Optional[TFTBaseline] = None

    def load(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})
        if hasattr(cfg_dict, 'items'):
            cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
        else:
            cfg = cfg_dict  # already a ModelConfig instance
        self.model = TFTBaseline(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._batch_to_device(batch)
        output = self.model(batch)
        return output["v_norm_pred"]


class InformerAdapter(BaseModelAdapter):
    """Informer Baseline 适配器"""

    def __init__(self, device: str = "cpu", eval_seed: int = 20261112):
        super().__init__(device, eval_seed)
        self.model: Optional[InformerBaseline] = None

    def load(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})
        if hasattr(cfg_dict, 'items'):
            cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
        else:
            cfg = cfg_dict  # already a ModelConfig instance
        self.model = InformerBaseline(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._batch_to_device(batch)
        with self._fixed_rng():
            output = self.model(batch)
        return output["v_norm_pred"]


class MultiDimIDWAdapter(BaseModelAdapter):
    """4D-IDW 插值适配器（无需 checkpoint）"""

    requires_checkpoint: bool = False

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from evaluation.enhanced_traditional import MultiDimIDW
        self.idw = MultiDimIDW(power=2.0, w_x=1.0, w_d=0.5, w_b=0.5, w_g=0.25)

    def load(self, checkpoint_path: str):
        pass  # no checkpoint needed

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = batch["context"].cpu().numpy()
        query = batch["query"].cpu().numpy()
        ctx_mask = batch.get("context_mask")  # True=有效; padding 必须过滤, 否则零填充被当作观测
        B = context.shape[0]
        preds = []
        for b in range(B):
            if ctx_mask is not None:
                valid = ctx_mask[b].cpu().numpy().astype(bool)
            else:
                valid = np.ones(context.shape[1], dtype=bool)
            ctx_4d = context[b, valid, :4]
            ctx_val = context[b, valid, 4]
            qry_4d = query[b, :, :4]
            pred = self.idw.interpolate(ctx_4d, ctx_val, qry_4d)
            preds.append(torch.from_numpy(pred).float())
        return torch.stack(preds).to(self.device)


class OrdinaryKrigingAdapter(BaseModelAdapter):
    """Ordinary Kriging 插值适配器（无需 checkpoint）"""

    requires_checkpoint: bool = False

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        from evaluation.enhanced_traditional import OrdinaryKriging
        self.krig = OrdinaryKriging(use_4d=True, sill=1.0, range_param=0.3, nugget=0.01)

    def load(self, checkpoint_path: str):
        pass

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = batch["context"].cpu().numpy()
        query = batch["query"].cpu().numpy()
        ctx_mask = batch.get("context_mask")  # True=有效; padding 必须过滤, 否则零填充被当作观测
        B = context.shape[0]
        preds = []
        for b in range(B):
            if ctx_mask is not None:
                valid = ctx_mask[b].cpu().numpy().astype(bool)
            else:
                valid = np.ones(context.shape[1], dtype=bool)
            ctx_4d = context[b, valid, :4]
            ctx_val = context[b, valid, 4]
            qry_4d = query[b, :, :4]
            pred = self.krig.interpolate(ctx_4d, ctx_val, qry_4d)
            preds.append(torch.from_numpy(pred).float())
        return torch.stack(preds).to(self.device)


# 适配器注册表: model_type -> AdapterClass
ADAPTER_REGISTRY = {
    "hydro_qt": HydroQueryTransformerAdapter,
    "query_transformer": HydroQueryTransformerAdapter,
    "lstm": LSTMAdapter,
    "gan_z0": GANZ0Adapter,
    "gan_mean5": GANMeanAdapter,
    "diffusion_mean5": DiffusionMeanAdapter,
    "diffusion_ddpm_mean5": DiffusionMeanAdapter,
    "tft": TFTAdapter,
    "informer": InformerAdapter,
    "idw_4d": MultiDimIDWAdapter,
    "kriging": OrdinaryKrigingAdapter,
}

_INTERPOLATION_LOADED = False


def _ensure_interpolation_loaded():
    """延迟加载传统插值适配器, 避免循环导入"""
    global _INTERPOLATION_LOADED
    if _INTERPOLATION_LOADED:
        return
    from baseline.interpolation.adapter import (
        NearestXAdapter,
        LinearXAdapter,
        IDWXAdapter,
        IDWXDAdapter,
        CubicSplineAdapter,
        RBFXDAdapter,
    )
    ADAPTER_REGISTRY["nearest_x"] = NearestXAdapter
    ADAPTER_REGISTRY["linear_x"] = LinearXAdapter
    ADAPTER_REGISTRY["idw_x"] = IDWXAdapter
    ADAPTER_REGISTRY["idw_xd"] = IDWXDAdapter
    ADAPTER_REGISTRY["cubic_spline"] = CubicSplineAdapter
    ADAPTER_REGISTRY["rbf_xd"] = RBFXDAdapter
    _INTERPOLATION_LOADED = True


def create_adapter(model_type: str, device: str = "cpu", **kwargs) -> BaseModelAdapter:
    """创建模型适配器

    Args:
        model_type: 模型类型名称 (参见 ADAPTER_REGISTRY)
        device:     计算设备
        **kwargs:   适配器额外参数 (作为 fallback, 优先从 checkpoint 读取)
    """
    _ensure_interpolation_loaded()
    if model_type not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}, available: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[model_type](device=device, **kwargs)
