# Run guide — Windows PowerShell

Exact commands, grouped by what they actually do. **No CNN commands appear here**
— the CNN models are locked and are not part of the yield pipeline.

Legend: **[REQUIRED]** for a working setup · **[OPTIONAL]** extra analysis or a
full rebuild.

> Anything under **TRAINING** overwrites model artefacts and authoritative
> metrics. Everything under **EVALUATION**, **INFERENCE** and **REPORTS** is safe
> to run repeatedly.

---

## 1. Environment setup **[REQUIRED]**

The `.joblib` artefacts are **scikit-learn-version locked** — sklearn pickles are
not portable across feature releases in either direction. Use Python 3.10 and the
pinned versions, or plan to retrain.

```powershell
cd D:\BIG_Works_Done\Crop_yeild_predictions

# Recommended: conda (matches the reference environment exactly)
conda create -n wheat python=3.10 -y
conda activate wheat
```

Or a venv, if you already have Python 3.10:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `Activate.ps1` is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2. Dependencies **[REQUIRED]**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Reference environment (produced every result in `reports/`):
Python 3.10.16 · numpy 1.24.3 · pandas 2.2.3 · **scikit-learn 1.6.1** ·
scipy 1.15.3 · joblib 1.4.2 · matplotlib 3.10.0 · torch 2.6.0+cpu ·
xgboost 3.2.0 · lightgbm 4.6.0

> **xgboost and lightgbm must both be installed.** The trainers treat them as
> optional imports — if either is missing they are silently skipped and the
> five-model comparison quietly becomes a three-model comparison.

## 3. Smoke test **[REQUIRED]** — run this first

Read-only. Trains nothing, downloads nothing, overwrites nothing.

```powershell
python scripts\smoke_test.py
```

Expected: **31 PASS, 3 SKIP, 0 FAIL**. Any FAIL means the environment or the data
is wrong — fix that before running anything else.

## 4. Data preparation **[OPTIONAL]**

**Not needed for normal use.** `data/processed/stage_features.csv` ships with the
repository and is all the modelling needs. Only rebuild if you have restored the
raw inputs described in [`../data/README.md`](../data/README.md).

```powershell
python src\data\extract_apy_all_districts_xlsx.py   # APY xlsx -> district table
python src\data\create_prediction_2025_cases.py     # 2024-25 case rows
python src\data\build_coordinate_registry.py        # district lat/lon
python src\data\download_weather.py                 # NETWORK: Open-Meteo archive
python src\data\check_satellite_files.py            # audit GEE exports
python src\data\ingest_satellite.py                 # -> satellite_observations.csv
python src\data\build_image_stage_features.py       # per-stage image averages
python src\data\build_stage_table.py                # -> stage_features*.csv
python src\data\verify_training_data.py             # sanity report
```

> **Do not run** `src\features\extract_image_features.py` — 335 of the 560
> referenced auxiliary images are missing from this copy, and re-running it would
> silently change the per-stage averages the saved models were fitted on.

---

## TRAINING — overwrites models and metrics

Only run these if you intend to replace the current artefacts. Results will not
be bit-identical to the shipped ones unless your environment matches exactly.

```powershell
# [OPTIONAL] stage-wise + pooled regression  (~1 min)
#   -> models/regression/*.joblib (10)
#   -> reports/model_results/stagewise_regression_metrics.csv
#   -> reports/model_results/combined_regression_metrics.csv
python src\models\train_stagewise_regression.py

# [OPTIONAL] Q1-Q5 quintile classifiers  (~20 s)
#   -> models/classification/*.joblib (9)
#   -> reports/model_results/quintile_classification_metrics.csv
python src\models\train_quintile_classifier.py

# [OPTIONAL] temporal LSTM  (~30 s, seed 42, deterministic)
#   -> models/deep_learning/lstm_yield_model.pt
#   -> reports/model_results/lstm_metrics.csv, lstm_training_log.csv
#   -> reports/figures/lstm_training_loss.png
#   -> data/processed/lstm_predictions_2025.csv
python src\models\train_lstm_sequence_model.py

# [OPTIONAL] convenience wrapper: regression + quintile + baseline_ml_report.md
python run_baseline_ml.py
```

> `run_training.py` and `run_training_with_cnn.py` also exist. They rebuild the
> entire data tree and retrain everything (10–30 min, network required, and the
> CNN variant touches the CNN pipeline). **Not needed** for normal use.

## EVALUATION — evaluates existing data, writes only new report files

Neither of these touches `models/`, and neither overwrites the hold-out metrics.

```powershell
# [REQUIRED for the paper] walk-forward temporal CV + the three naive baselines  (~15 s)
#   -> reports/model_results/temporal_cv_metrics.csv
#   -> reports/model_results/temporal_cv_fold_metrics.csv
#   -> reports/model_results/baseline_comparison.csv
#   -> reports/summaries/temporal_validation_summary.md
#   -> reports/summaries/baseline_comparison.md
#   -> reports/figures/temporal_cv_model_vs_baseline.png
python src\models\run_temporal_validation.py

# [REQUIRED for the paper] within-district (anomaly) skill  (~5 s)
#   -> reports/model_results/within_district_skill.csv
#   -> reports/summaries/within_district_skill.md
python src\models\run_within_district_skill.py
```

> `run_within_district_skill.py` reads
> `data/processed/validation_stagewise_predictions.csv`, so run **INFERENCE**
> first if you have just retrained.

## INFERENCE — predictions only, never fits

```powershell
# [REQUIRED] loads the saved .joblib models and predicts  (~30 s)
#   -> data/processed/validation_stagewise_predictions.csv   (297, has actuals)
#   -> data/processed/prediction_2025_stagewise_predictions.csv (297, forecast)
#   -> data/processed/mehsana_2025_stagewise_predictions.csv (9, forecast)
#   -> reports/model_results/prediction_2025_evaluation.md
python src\models\predict_stagewise.py
```

## REPORTS — figures and derived reports

```powershell
# [REQUIRED] stage-wise trend figures + improvement report  (~10 s)
#   -> reports/figures/stagewise_rmse_trend.png
#   -> reports/figures/stagewise_mae_trend.png
#   -> reports/figures/mehsana_2025_stagewise_prediction.png
#   -> reports/model_results/stagewise_improvement_report.md
python src\visualization\plot_results.py

# [OPTIONAL] data-quality inspections (write reports only)
python src\data\verify_training_data.py
python src\data\diagnose_features.py
python src\data\check_satellite_files.py
```

---

## Recommended order from a fresh clone

```powershell
cd D:\BIG_Works_Done\Crop_yeild_predictions
conda create -n wheat python=3.10 -y ; conda activate wheat
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts\smoke_test.py                        # verify        [REQUIRED]
python src\models\predict_stagewise.py              # inference     [REQUIRED]
python src\models\run_temporal_validation.py        # evaluation    [REQUIRED]
python src\models\run_within_district_skill.py      # evaluation    [REQUIRED]
python src\visualization\plot_results.py            # reports       [REQUIRED]
python scripts\smoke_test.py                        # verify again
```

Nothing above retrains a model. Total runtime ≈ 1 minute.

## Full reproduction from scratch (retrains everything except the CNN)

```powershell
python src\models\train_stagewise_regression.py
python src\models\train_quintile_classifier.py
python src\models\train_lstm_sequence_model.py
python src\models\predict_stagewise.py
python src\models\run_temporal_validation.py
python src\models\run_within_district_skill.py
python src\visualization\plot_results.py
python scripts\smoke_test.py
```

## Reading the results

```powershell
notepad reports\summaries\research_ready_results.md   # start here
notepad reports\README.md                             # index of every result file
```

## Determinism

Seeds are 42 throughout — regression, quintile classifiers and the LSTM (Python,
NumPy, torch, cuDNN flags, and the DataLoader generator). Splits are hard-coded
by season, not sampled. Two consecutive LSTM runs produce byte-identical
artefacts. Cross-version library differences can still shift the last digits.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'` | scikit-learn newer than 1.6.x | `pip install "scikit-learn>=1.6,<1.7"` or retrain |
| `ModuleNotFoundError: numpy._core.multiarray` | artefacts pickled under NumPy 2, running NumPy 1 | match the pinned versions or retrain |
| Only 3 regressors in the metrics grid | xgboost / lightgbm not installed | `pip install xgboost lightgbm` and retrain |
| Smoke test: `no leaked features in any model` FAILS | a trainer bypassed `src/features/feature_policy.py` | do not ignore this — it means target leakage |
| Smoke test: `2024-25 ground-truth integrity` FAILS | something wrote a 2024-25 target | see `docs/2025_yield_reference_methodology.md` |
