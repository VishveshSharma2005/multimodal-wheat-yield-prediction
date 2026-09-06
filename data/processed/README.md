# data/processed

Modelling-ready tables and model outputs.

## Committed

| File | Shape | What it is |
|---|---|---|
| `stage_features.csv` | 1485 x 55 | **The modelling table.** One row per (case, stage) = 165 cases x 9 stages. Every real feature, the `yield_kg_ha` target, `season_year_start` for the split, and the stage keys. |
| `targets.csv` | 165 | Per-case yield + quintile label. The `yield_quintile` column here is computed over `config.train_years` (2021-2024) and is **not** used by any model — `train_quintile_classifier.py` recomputes thresholds from the training split. |
| `image_features.csv` | 560 | Handcrafted features per auxiliary image |
| `image_stage_features.csv` | 5 | Per-stage image averages actually merged into the model |
| `validation_stagewise_predictions.csv` | 297 | **Raw** district-level 2023-24 predictions (33 districts x 9 stages). Cite these, not the stage aggregates in the figures. |
| `prediction_2025_stagewise_predictions.csv` | 297 | Raw 2024-25 **forecasts** |
| `mehsana_2025_stagewise_predictions.csv` | 9 | Mehsana 2024-25 **forecasts** |
| `lstm_predictions_2025.csv` | 33 | LSTM case-level 2024-25 **forecasts** |

## Gitignored (regenerable)

| File | Size | Regenerate with |
|---|---|---|
| `stage_features_with_split.csv` | 11 MB | `src/data/build_stage_table.py` |
| `stage_features_with_cnn.csv` | 11 MB | `src/data/build_stage_table.py` |
| `cnn_features.csv` | 618 MB | `src/deep_learning/extract_cnn_features.py` |
| `cnn_features/*.csv` (5) | 620 MB | same |
| `cnn_stage_features.csv` | 785 KB | `src/deep_learning/build_cnn_stage_features.py` |

`stage_features_with_split.csv` and `stage_features_with_cnn.csv` are written
from the same DataFrame and are 6,969 columns wide, of which **6,913 are the
all-NaN `cnn_*` columns**. `stage_features.csv` holds the identical 55 real
columns at 1/13th the size.

## Note on the 2024-25 target

All 297 rows with `season_year_start = 2024` have a **null** `yield_kg_ha`,
`actual_available = 0` and `evaluation_status = pending_actual_apy`, because no
official district-level 2024-25 APY exists. Until 2026-09-05 a fallback in
`src/data/ingest_apy_2025_ground_truth.py` populated 9 Mehsana rows with a
carried-forward 2023-24 value; that has been removed. See
[`../../docs/2025_yield_reference_methodology.md`](../../docs/2025_yield_reference_methodology.md).
