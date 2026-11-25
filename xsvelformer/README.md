# XS-VelFormer

XS-VelFormer（原名 Hydro-Query Transformer）—— 河道断面流速插补 / 重构 / 加密的**生产推理 SDK**。

把研究期代码封装为「原始物理量 → 流速预测 (m/s)」的干净接口：归一化、特征工程（bank/grad）、padding、反归一化全部在 pipeline 内部完成，调用方只接触物理单位。

## 安装

```bash
pip install dist/xsvelformer-1.0.0-py3-none-any.whl
```

依赖：`numpy>=1.23`、`torch>=2.0`。

## 模型权重（需单独获取）

wheel **默认不含模型权重**（以便单独更新模型而不重打 wheel）。请从产物包目录加载：

```
artifacts/xsvelformer-v1/
├── model.pt           # state_dict
├── config.json        # 模型架构参数
├── normalizer.json    # v/Z/h_max/W 的 mean/std
└── manifest.json      # 版本 / 来源 / 指标 / IO schema
```

若需从训练期旧 checkpoint 生成产物包：

```bash
python -m xsvelformer.convert_legacy checkpoints/full.pt artifacts/xsvelformer-v1
```

## 用法

```python
import xsvelformer as xsv

pipe = xsv.XSVelFormerPipeline.from_artifact("artifacts/xsvelformer-v1")

section = xsv.SectionData(
    distances=[0, 5, 10, 60, 120],            # m，起点距（升序）
    depths=[0.2, 1.5, 2.0, 1.8, 0.3],          # m，水深
    velocities=[0.1, None, 1.2, None, 0.05],   # m/s，None=该点缺测
    water_stage=25.3,                          # m，水位
    width=120.0,                               # m，水面宽
)

# 1) 插补缺测点
res = pipe.impute_missing(section)
# res.velocities (m/s), res.is_predicted

# 2) 100 条测线重构整条断面
res2 = pipe.reconstruct(section, n_lines=100)

# 3) 加密到 200 条测线
res3 = pipe.densify(section, n_target=200)

# 4) 自定义 query 位置
res4 = pipe.predict(section, xsv.QuerySpec(distances=[20.0, 40.0]))

# 5) 批量推理（变长断面内部 padding，单次前向）
results = pipe.predict_many([sec1, sec2], [query1, query2])
```

## API 速览

| 名称 | 说明 |
|---|---|
| `XSVelFormerPipeline` | 推理编排器，`from_artifact(dirpath)` 构建 |
| `SectionData` | 原始断面输入（distances/depths/velocities/water_stage/width） |
| `QuerySpec` | 目标预测位置（distances/可选 depths） |
| `PredictionResult` | 输出（distances/depths/velocities/is_predicted/diagnostics） |
| `make_query_from_missing(section)` | 由缺测点自动生成 query |
| `validate_section / validate_query` | 输入校验 |
| `ValidationError / ArtifactError` | 异常 |

## 场景方法

- `impute_missing(section)` — 填充 `velocities` 为 `None` 的位置
- `reconstruct(section, n_lines)` — 等间距网格重构
- `densify(section, n_target)` — 在现有测线间加密到目标点数

## 物理后处理

- `clip_negative=True`（默认）：负预测流速裁剪到 0（物理兜底）