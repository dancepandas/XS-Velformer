# Data format

The training pipeline consumes a **pickle file containing a list of
cross-section dictionaries**. This mirrors the format produced by the
monitoring pipeline; the classes in `data/preprocess.py` read it directly.

## `sections.pkl`

```python
[
    {
        "x":          np.ndarray,  # [K] float, start-distance of each vertical (m)
        "h":          np.ndarray,  # [K] float, water depth at each vertical (m)
        "v":          np.ndarray,  # [K] float, surface velocity at each vertical (m/s)
        "Z":          float,       # water stage (m)
        "station_id": str,         # station identifier
        "time":       str,         # "YYYY-MM-DD HH:MM:SS" (required for time-based splits)
    },
    ...
]
```

* `K` may differ between sections (10–100 lines is typical). All arrays of one
  section must share the same length.
* `v` is the target. During training, a random subset of measured verticals is
  hidden; the model sees only the remaining lines plus the geometry of every
  line, and predicts velocity at the hidden ones.
* Sections with fewer than 4 lines are skipped by the split utilities.

## Per-line features

| Feature | Definition | Range |
|---|---|---|
| `x_norm` | `(x - x_min) / W` | [0, 1] |
| `d_norm` | `h / h_max` | [0, 1] |
| `bank_norm` | `2 * min(x_norm, 1 - x_norm)` | [0, 1], 0 at banks |
| `grad_norm` | local depth gradient, normalized by its maximum | [0, 1] |
| `v_norm` | `(v - mean_v) / std_v` (training statistics) | standardized |
| `Z_norm`, `h_max_norm`, `W_norm` | global standardization with training statistics | standardized |

Context lines carry 8 features (including `v_norm`); query lines carry 7
(no velocity). Standardization statistics are fitted **on the training split
only** and reused for validation/test to avoid leakage.

## Splits

`data/preprocess.py` implements three split modes:

| Mode | Held out |
|---|---|
| `station` | entire stations |
| `time` | the latest time block of every station |
| `station_and_time` (recommended) | union of the two: some stations entirely + the latest time blocks of the rest |

## Masking protocols

`evaluation/test_cases.py` generates the evaluation masks used in the paper:
random (query ratio 0.2/0.3/0.4), continuous blocks, edge blocks,
sparse-context reconstruction and densification.

## Metrics

* `masked_mse` — mean squared error in standardized velocity space (model selection);
* `rmse`, `mae`, `bias` — physical units (m/s); `nse`, `kge` — dimensionless.

The raw monitoring data are proprietary and are **not** distributed. Use
`examples/make_synthetic_sections.py` to create a compatible synthetic dataset.

## Data source

All field data used in this study were collected and provided by
**Wuhan Dashuiyun Technology Co., Ltd.** (武汉大水云科技有限公司). The data are
proprietary and are not redistributed with this repository.
