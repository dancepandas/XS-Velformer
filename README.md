# XS-VelFormer

**XS-VelFormer** is a morphology-aware cross-attention transformer for river
cross-section surface-velocity estimation. A single model handles three
production tasks that share the same inputs:

1. **Missing-line interpolation** — some measured verticals are missing;
2. **Standard cross-section reconstruction** — predict velocity at a fixed grid;
3. **Vertical densification** — predict velocity between sparse measured lines.

The model operates on **variable-length** cross sections (typically 10–100
measurement lines) and does not require resampling to a fixed grid.

## Model overview

| Component | Description |
|---|---|
| Context encoder | 8-D per line: `x_norm, d_norm, bank_norm, grad_norm, v_norm, Z_norm, h_max_norm, W_norm` |
| Query encoder | 7-D per target line: same geometry/global features, without velocity |
| Cross-attention | distance-biased attention with terms on `\|x\|, \|d\|, \|bank\|, \|grad\|` |
| Prediction head | MLP → standardized velocity, de-normalized with training statistics |

Training uses a masked-velocity objective plus physically motivated
regularizers (smoothness, depth–velocity correlation, near-bank behaviour,
zero-velocity retention, gradient smoothness). Evaluation reports normalized
masked MSE and physical RMSE / MAE / NSE / KGE / bias.

## Repository layout

```
.
├── config.py            # model / training / data configuration dataclasses
├── train.py             # training loop (StepLogger, TrainingHistory, Trainer)
├── main.py              # training CLI
├── predict.py           # unified inference interface (VelocityPredictor)
├── losses.py            # masked MSE + physical regularizers
├── models/              # encoders, morphology-aware cross-attention, full model
├── data/                # feature engineering, variable-length dataset, splits
├── evaluation/          # masking protocols, metrics, adapters, evaluation CLI
├── baseline/            # LSTM / GAN / Diffusion / Informer / TFT / PatchTST / interpolation
├── xsvelformer/         # packaged inference SDK (convert a checkpoint, run a pipeline)
├── examples/            # synthetic data generator, format description, quickstart
└── tests/               # unit tests that do not need proprietary data
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: install the inference SDK as a package
pip install -e .
```

Python ≥ 3.10 is required. A CUDA GPU is recommended for training; CPU is
sufficient for the quickstart.

## Quickstart (synthetic data, no proprietary data needed)

```bash
python examples/make_synthetic_sections.py --out examples/sections_synthetic.pkl
python examples/quickstart.py --data examples/sections_synthetic.pkl
```

`quickstart.py` builds a small model, runs a masked forward pass on a synthetic
cross section, and prints the error. It demonstrates the full input pipeline
(feature engineering → standardization → masking → model).

The input data format is documented in [`examples/data_format.md`](examples/data_format.md).

## Training

```bash
python main.py --data path/to/sections.pkl --epochs 60 --save checkpoints/model.pt
```

The paper uses the **station-and-time hold-out** split
(`--split station_and_time`): some stations are held out entirely, and for the
remaining stations the latest time blocks are held out. See
`data/preprocess.py::split_by_station_and_time`. The `stratified` mode is the
code default if no split is given.

## Evaluation

`evaluation/` contains the masking protocols (random / continuous / edge /
sparse-reconstruction / densification) and the metric implementations used in
the paper. A generic CLI is provided:

First generate a fixed test-case file (the masking protocol is stored with the
cases, so every model is scored on identical masks):

```bash
python -m evaluation.test_cases --data path/to/sections.pkl \
    --split station_and_time --output result/fixed_test_cases.pkl
```

Then evaluate a checkpoint:

```bash
python -m evaluation.evaluate \
    --model_type query_transformer \
    --checkpoint checkpoints/model.pt \
    --test_cases result/fixed_test_cases.pkl \
    --output result/eval.json
```

Baselines are implemented under `baseline/` (LSTM, Conditional GAN, DDPM
diffusion, Informer, TFT, PatchTST) and in `baseline/interpolation/`
(nearest, linear, IDW-x, IDW-xd, cubic spline, thin-plate RBF).

## Data source and availability

All field data used in this study — the river cross-section measurements used
for training, validation and testing, including the monitoring corpus and the
evaluation sets — were collected and provided by
**Wuhan Dashuiyun Technology Co., Ltd.** (武汉大水云科技有限公司), the
developer of the AiFlow visual flow-measurement platform. The data are
proprietary and **cannot be redistributed**; requests for data access should be
addressed to the company.

This repository therefore ships:

- the complete input/output schema and feature definitions;
- a synthetic section generator so every code path can be executed end-to-end;
- training, evaluation and baseline code.

Pre-trained checkpoints are not included. To package a checkpoint for
inference, use `python -m xsvelformer.convert_legacy` (see `xsvelformer/README.md`).

## Acknowledgements

We thank Wuhan Dashuiyun Technology Co., Ltd. (武汉大水云科技有限公司) for
collecting and providing the field measurements used in this work.

## Citation

If you use this code, please cite the corresponding paper (see `CITATION.cff`).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
