# Pipeline guide — the complete non-CNN pipeline

End-to-end walkthrough of the yield-prediction pipeline. **The CNN branch is not
covered here** — it is a separate wheat growth-stage image classifier that does
not feed the yield models (see `docs/models.md` §1.4).

Commands for each step are in [`RUN_GUIDE.md`](RUN_GUIDE.md).

---

## Overview

```
 ┌─ APY yield statistics (Gujarat Dept. of Agriculture)
 ├─ Weather (Open-Meteo / ERA5)                       ← data sources
 ├─ Sentinel-2 NDVI/NDRE/EVI (Google Earth Engine)
 └─ Auxiliary field images (Mendeley WheatPhenology)
                    │
                    ▼   ingestion
        data/interim/{apy_yield_all_districts, cases,
                      weather_daily, satellite_observations,
                      image_metadata}.csv
                    │
                    ▼   stage generation + feature engineering
        data/processed/stage_features.csv       (1485 × 55)
                    │
                    ▼   feature policy / leakage filter
        src/features/feature_policy.py          (blocks 18 columns)
                    │
                    ▼   split by season_year_start
        train 2020-22 (891) │ validation 2023 (297) │ forecast 2024 (297)
                    │
      ┌─────────────┼──────────────┬─────────────────┐
      ▼             ▼              ▼                 ▼
 stage-wise    combined      LSTM sequence    quintile Q1–Q5
 regression    regression    (9 timesteps)    classification
      │             │              │                 │
      └─────────────┴──────┬───────┴─────────────────┘
                           ▼   evaluation
       hold-out metrics · walk-forward CV · baselines · within-district skill
                           ▼   prediction
       validation / 2024-25 forecast / Mehsana forecast CSVs
                           ▼   visualization + reports
       reports/figures/*.png · reports/summaries/*.md
```

---

## Stage 1 — Data sources

| Source | What | Where it lands |
|---|---|---|
| Directorate of Agriculture, Gujarat — APY | District wheat yield, 2020-21…2023-24 | `data/raw/apy/*.xlsx`, `*.pdf` |
| Open-Meteo Archive (ERA5) | Daily rain, Tmin, Tmax, humidity, radiation | fetched by script |
| Sentinel-2 `COPERNICUS/S2_SR_HARMONIZED` via GEE | NDVI, NDRE, EVI, cloud % | `data/raw/satellite/*.csv` (153 files) |
| Mendeley *WheatPhenology* | 560 auxiliary field images | `data/raw/images/auxiliary_wheat_images/` |

Provenance and download instructions: [`../data/README.md`](../data/README.md).

> Weather is **Open-Meteo / ERA5**, not NASA POWER. There are **no soil features**.

## Stage 2 — Ingestion

| Step | Script | Input | Output |
|---|---|---|---|
| APY → district table | `src/data/extract_apy_all_districts_xlsx.py` | `data/raw/apy/*.xlsx` | `data/interim/apy_yield_all_districts.csv` (132 rows) |
| Build forecast cases | `src/data/create_prediction_2025_cases.py` | APY table | 2024-25 rows in `data/interim/cases.csv` |
| District coordinates | `src/data/build_coordinate_registry.py` | `data/raw/field_points/` | `data/interim/coordinate_registry.csv` |
| Weather download | `src/data/download_weather.py` | coordinate registry + Open-Meteo API | `data/interim/weather_daily.csv` (20,856 rows) |
| Satellite audit | `src/data/check_satellite_files.py` | `data/raw/satellite/` | `reports/satellite_file_check_report.md` |
| Satellite ingest | `src/data/ingest_satellite.py` | `data/raw/satellite/*.csv` | `data/interim/satellite_observations.csv` (6,779 rows) |
| Image inventory | `src/data/ingest_images.py` | `data/raw/images/` | `data/interim/image_metadata.csv` |
| 2024-25 ground truth | `src/data/ingest_apy_2025_ground_truth.py` | `data/raw/apy/` | `data/interim/apy_2025_ground_truth.csv` (**header-only** — none published) |

## Stage 3 — Preprocessing / image features

| Step | Script | Input | Output |
|---|---|---|---|
| Per-image handcrafted features | `src/features/extract_image_features.py` | auxiliary images | `data/processed/image_features.csv` (560 rows) |
| Per-stage image averages | `src/data/build_image_stage_features.py` | above | `data/processed/image_stage_features.csv` (5 rows) |

> These become **global per-stage constants** broadcast to every district and
> year, so they carry no district or season signal. Do not re-run
> `extract_image_features.py` unless the full 560-image set is restored — 335 are
> currently missing.

## Stage 4 — Stage generation + feature engineering

**Script:** `src/data/build_stage_table.py`
**Helpers:** `src/features/crop_stages.py`, `weather_features.py`, `vegetation_indices.py`

Splits each district-season into **9 days-after-sowing windows** defined in
`configs/config.yaml`, then aggregates every modality **only over data up to
`stage_end_date`** — the core leakage control.

| Input | Output |
|---|---|
| `cases.csv`, `weather_daily.csv`, `satellite_observations.csv`, `image_stage_features.csv` | `data/processed/stage_features.csv` (**1485 × 55** — the modelling table) |
| | `data/processed/stage_features_with_split.csv`, `stage_features_with_cnn.csv` (11 MB each, gitignored — same real columns plus 6,913 all-NaN `cnn_*`) |
| | `data/processed/targets.csv` (per-case yield + quintile label) |

Stages: `sowing` (0–10 DAS) → `early_vegetative` (10–20) → `tillering` (20–40) →
`stem_elongation` (40–60) → `booting_heading` (60–80) → `flowering` (80–95) →
`grain_filling_initial` (95–105) → `grain_filling` (105–115) →
`maturity_preharvest` (115–130).

**Verification:** `src/data/verify_training_data.py` → `reports/training_data_verification.md`;
`src/data/diagnose_features.py` → `reports/feature_diagnosis_report.md`.

## Stage 5 — Feature policy / leakage filtering

**Script:** `src/features/feature_policy.py` — the single source of truth,
imported by all four trainers so the rules cannot drift apart.

Blocks 18 columns:

| Category | Columns |
|---|---|
| Identifiers / split keys | `case_id`, `split`, `season_year_start/end`, `stage_start_date`, `stage_end_date` |
| **Target-derived** | `yield_kg_ha`, `yield_quintile`, `actual_yield_kg_ha`, `predicted_yield_kg_ha`, `absolute_error`, `percentage_error`, **`actual_available`**, **`evaluation_status`** |
| **Exact duplicates** | `days_after_sowing`, `cumulative_rain_mm`, `cumulative_rainfall`, `mean_tmin_c`, `mean_tmax_c`, `gdd_cumulative`, `cumulative_gdd`, `ndvi_growth_rate` |
| Free text / provenance | `source_file`, `source_sheet`, `notes`, `image_feature_source` |

Explicitly **protected** (never blocked): `stage`, `stage_name`, `stage_index`,
`stage_order`, `stage_start_das`, `stage_end_das`, `crop_duration_days`.

Columns >90 % missing are additionally dropped by each trainer.

**Output:** `reports/model_results/feature_audit.csv` (per-column decision + reason).
`scripts/smoke_test.py` fails if any model input ever violates this policy.

## Stage 6 — Train / validation / forecast split

Assigned by `season_year_start`, hard-coded identically in every trainer:

| Split | Seasons | Cases | Rows | Target |
|---|---|---|---|---|
| train | 2020-21, 2021-22, 2022-23 | 99 | 891 | yes |
| validation | 2023-24 | 33 | 297 | yes |
| forecast | 2024-25 | 33 | 297 | **no** (`pending_actual_apy`) |

## Stage 7 — Stage-wise regression

**Script:** `src/models/train_stagewise_regression.py`

Nine per-stage models. Five candidates fitted per stage — RandomForest,
GradientBoosting, HistGradientBoosting, LightGBM, XGBoost (all `random_state=42`)
— and the lowest 2023-24 RMSE is kept. Pipeline:
`ColumnTransformer(median-impute numeric | most-frequent + one-hot categorical)`
→ regressor. A pooled all-stage model is also fitted.

| Output | Where |
|---|---|
| 9 per-stage + 1 combined model | `models/regression/*.joblib` |
| Full 45-row metric grid | `reports/model_results/stagewise_regression_metrics.csv` |
| Pooled metrics | `reports/model_results/combined_regression_metrics.csv` |
| Winners + feature groups | `reports/model_results/stagewise_regression_summary.md` |
| Gini importances | `reports/model_results/feature_importance_*.csv` |
| Feature selection audit | `reports/model_results/ml_selected_features.csv` |

## Stage 8 — Temporal LSTM

**Script:** `src/models/train_lstm_sequence_model.py` (PyTorch)

One sequence per case: the 9 stages in DAS order → one final yield.
`LSTM(input_size=76, hidden=64, layers=2, dropout=0.2)` → `Linear(64,1)` on packed
sequences. Adam lr 1e-3, batch 16, MSE, early stopping patience 15.
Imputer, scaler, one-hot encoder and target normalisation are fitted on
**training rows only**. Seeded with 42 (Python/NumPy/torch/cuDNN/DataLoader) and
byte-reproducible.

| Output | Where |
|---|---|
| Checkpoint (+ `best_epoch`, `best_val_rmse`, `epochs_run`, `random_seed`) | `models/deep_learning/lstm_yield_model.pt` |
| Metrics | `reports/model_results/lstm_metrics.csv` |
| Per-epoch log | `reports/model_results/lstm_training_log.csv` |
| Feature/summary audit | `reports/model_results/lstm_summary.md`, `lstm_selected_features.csv` |
| Loss curve | `reports/figures/lstm_training_loss.png` |
| 2024-25 forecasts | `data/processed/lstm_predictions_2025.csv` |

## Stage 9 — Quintile classification

**Script:** `src/models/train_quintile_classifier.py`

Q1 (lowest) … Q5 (highest). Thresholds from `pd.qcut(train_y, 5)` on the
**training split only** — verified leakage-free. RandomForest vs
GradientBoosting per stage.

| Output | Where |
|---|---|
| 9 classifiers | `models/classification/*.joblib` |
| Metrics | `reports/model_results/quintile_classification_metrics.csv` |
| Summary | `reports/model_results/quintile_classification_summary.md` |
| Confusion matrices | `reports/model_results/confusion_matrix_<stage>.csv` |

## Stage 10 — Evaluation

Three **separate** evaluations. They answer different questions and are not
interchangeable.

| Evaluation | Script | Output |
|---|---|---|
| 2023-24 hold-out | (produced by the trainers above) | `stagewise_regression_metrics.csv`, `combined_regression_metrics.csv`, `lstm_metrics.csv`, `quintile_classification_metrics.csv` |
| **Walk-forward CV + baselines** | `src/models/run_temporal_validation.py` | `temporal_cv_metrics.csv`, `temporal_cv_fold_metrics.csv`, `baseline_comparison.csv`, `summaries/temporal_validation_summary.md`, `summaries/baseline_comparison.md`, `figures/temporal_cv_model_vs_baseline.png` |
| **Within-district (anomaly) skill** | `src/models/run_within_district_skill.py` | `within_district_skill.csv`, `summaries/within_district_skill.md` |

**Walk-forward protocol** — strictly forward-chaining, a fold never sees data at
or after its validation year (asserted by the smoke test):

```
fold 1: train 2020-21                  → validate 2021-22
fold 2: train 2020-21…2021-22          → validate 2022-23
fold 3: train 2020-21…2022-23          → validate 2023-24
```

2024-25 is never a fold — no published target.

**Baselines**, on the same folds: A = district historical mean · B = persistence
(last known year) · C = district identity only (no weather/satellite/image).

## Stage 11 — Prediction

**Script:** `src/models/predict_stagewise.py` — loads the saved `.joblib` models
and runs inference only. Never fits.

| Output | Rows | Type |
|---|---|---|
| `data/processed/validation_stagewise_predictions.csv` | 297 | validation (has actuals) |
| `data/processed/prediction_2025_stagewise_predictions.csv` | 297 | **forecast** (no actuals) |
| `data/processed/mehsana_2025_stagewise_predictions.csv` | 9 | **forecast** |
| `reports/model_results/prediction_2025_evaluation.md` | — | confirms metrics are N/A |

## Stage 12 — Visualization

**Script:** `src/visualization/plot_results.py`

| Output | From |
|---|---|
| `reports/figures/stagewise_rmse_trend.png` | validation predictions grouped by stage |
| `reports/figures/stagewise_mae_trend.png` | same |
| `reports/figures/mehsana_2025_stagewise_prediction.png` | Mehsana forecast (forecast line only) |
| `reports/model_results/stagewise_improvement_report.md` | per-stage validation metrics ordered by DAS |

## Stage 13 — Reports

| Report | Produced by |
|---|---|
| `reports/summaries/research_ready_results.md` | maintained by hand from the CSVs above — **the authoritative narrative** |
| `reports/summaries/final_model_summary.csv` | consolidated 38-row table |
| `reports/summaries/forecast_2024_25.md` | forecast status and limitations |
| `reports/model_results/baseline_ml_report.md` | `run_baseline_ml.py` |
| `reports/README.md` | index of everything above |

---

## Leakage controls — all verified

| Control | Where |
|---|---|
| Stage aggregates use only data ≤ `stage_end_date` | `build_stage_table.py` |
| Forecast season never enters training | `_compute_split()` in every trainer |
| Target-derived columns blocked from all model inputs | `src/features/feature_policy.py` |
| Quintile thresholds from training rows only | `train_quintile_classifier.py` |
| LSTM imputer/scaler/encoder/target-norm fitted on training rows only | `train_lstm_sequence_model.py` |
| Walk-forward folds never see the future | `run_temporal_validation.py`, asserted in the smoke test |
| Within-district centring uses training-season means only | `run_within_district_skill.py` |
| 2024-25 target must stay null | `ingest_apy_2025_ground_truth.py`, asserted across 8 artefacts |

## What this pipeline does **not** include

* **CNN features.** The CNN is a separate image classifier; its features never
  reach the yield models.
* **Soil data.** Not ingested, not modelled.
* **Drone / farm logs.** Empty templates only.
