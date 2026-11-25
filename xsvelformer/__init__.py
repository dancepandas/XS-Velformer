"""
xs_velformer — XS-VelFormer 生产推理 SDK

把研究期 Hydro-Query Transformer 封装为 "原始物理量 → 流速预测 (m/s)" 的标准 pipeline。

典型用法:
    import xsvelformer as xsv

    pipe = xsv.XSVelFormerPipeline.from_artifact("artifacts/xsvelformer-v1")

    section = xsv.SectionData(
        distances=[0, 5, 10, 60, 120],           # m
        depths=[0.2, 1.5, 2.0, 1.8, 0.3],        # m
        velocities=[0.1, None, 1.2, None, 0.05],  # m/s, None=缺测
        water_stage=25.3,                         # m
        width=120.0,                              # m
    )
    res = pipe.impute_missing(section)
    # res.velocities (m/s), res.is_predicted

转换旧 checkpoint:
    python -m xsvelformer.convert_legacy checkpoints/full.pt artifacts/xsvelformer-v1
"""

from importlib import resources as _resources

from .schemas import PredictionResult, QuerySpec, SectionData, make_query_from_missing
from .normalizer import Normalizer, StatPair
from .model import XSVelFormerModel
from .pipeline import XSVelFormerPipeline
from .artifact import Manifest, load_artifact, write_artifact, ArtifactError
from .validation import ValidationError, validate_query, validate_section

__version__ = "1.0.0"


def default_artifact_path() -> str:
    """返回包内自带的默认产物包目录路径 (若随 wheel 一起分发时可用)。

    注: 当前 wheel 默认不含模型权重 (以便单独更新模型而不重打 wheel)。
    若产物包未打进包内, 此函数会抛 FileNotFoundError —— 此时请显式传入
    dirpath 使用 XSVelFormerPipeline.from_artifact(dirpath)。
    """
    try:
        # Python 3.9+ resources API
        return str(_resources.files("xsvelformer").joinpath("_artifacts/xsvelformer-v1"))
    except Exception as e:  # pragma: no cover
        raise FileNotFoundError(
            "包内未携带默认产物包。请用 XSVelFormerPipeline.from_artifact(dirpath) "
            "显式指定产物包目录 (含 model.pt/config.json/normalizer.json/manifest.json)。"
        ) from e

__all__ = [
    # 公共 API
    "XSVelFormerPipeline",
    "SectionData",
    "QuerySpec",
    "PredictionResult",
    "make_query_from_missing",
    # 进阶 (一般不需要直接用)
    "XSVelFormerModel",
    "Normalizer",
    "StatPair",
    "Manifest",
    "load_artifact",
    "write_artifact",
    # 异常
    "ValidationError",
    "ArtifactError",
    # 进阶校验
    "validate_section",
    "validate_query",
    # 工具
    "default_artifact_path",
    "__version__",
]
