# Verified Pipeline Architecture

Everything below was traced through the code and confirmed against the artefacts
on disk. Where the conceptual design and the implementation differ, the
implementation is what is documented.

---

## 1. The pipeline that actually runs

```
data/raw/apy/*.xlsx                      data/raw/satellite/*.csv (153 GEE exports)
        |                                          |
extract_apy_all_districts_xlsx.py          check_satellite_files.py
        |                                   ingest_satellite.py
        v                                          |
data/interim/apy_yield_all_districts.csv           v
   (33 districts x 4 years)              data/interim/satellite_observations.csv
        |                                    (NDVI / NDRE / EVI / cloud_pct)
        |                                          |
        +--> data/interim/cases.csv <--------------+
        |    (165 cases = 33 districts x 5 seasons)
        |            ^                             |
        |            |                             |
   Open-Meteo archive API                          |
   download_weather.py                             |
        |            |                             |
        v            |                             |
data/interim/weather_daily.csv                     |
   (20,856 daily rows)                             |
        |                                          |
        +----------------+-------------------------+
                         |
              src/data/build_stage_table.py
              (9 DAS windows per case; every
               aggregate uses data <= stage_end_date)
                         |
                         v
        data/processed/stage_features.csv
        1485 rows x 55 cols  = 165 cases x 9 stages
                         |
      +------------------+-------------------+
      |                                      |
 train_stagewise_regression.py        train_lstm_sequence_model.py
 train_quintile_classifier.py         (9-step sequence per case)
      |                                      |
      v                                      v
 models/regression/*.joblib (10)      models/deep_learning/lstm_yield_model.pt
 models/classification/*.joblib (9)          |
      |                                      v
      v                             data/processed/lstm_predictions_2025.csv
 predict_stagewise.py
      |
      v
 validation_stagewise_predictions.csv     (297 rows, 2023-24, scored)
 prediction_2025_stagewise_predictions.csv (297 rows, 2024-25, FORECAST)
 mehsana_2025_stagewise_predictions.csv    (9 rows, FORECAST)
      |
      v
 plot_results.py -> reports/figures/*.png
```

## 2. The CNN branch — separate, not fused

```
data/raw/images/wheat_stage_dataset/  (5,211 field images, 5 classes)
        |
   train_{densenet121,mobilenetv2,inceptionv3,vgg16,xception}.py
        |
        v
   models/cnn/*_best.keras  (5 models, 271 MB)
        |
   evaluate_cnn_models.py -> reports/model_results/cnn_evaluation_metrics.csv
        |
   extract_cnn_features.py
        |
        v
   data/processed/cnn_features.csv  (5,211 rows x 6,912-dim concatenated vector)
        |
   build_cnn_stage_features.py  -> mean-pools to 5 rows, keyed on the IMAGE
        |                           class name (1_Tillering, 2_Jointing, 3_BH,
        v                           4_Flowering, 5_Filling)
   data/processed/cnn_stage_features.csv
        |
   build_stage_table.py::_merge_cnn_stage_features()
        |
        |    merge on "stage" ...  but the stage table uses agronomic names
        |    (sowing, tillering, ... maturity_preharvest).
        |    NO KEY MATCHES.
        v
   6,913 cnn_* columns, 100% NaN, in every row
        |
        v
   dropped by the ">90% missing" filter in both the regression and
   the LSTM feature selectors.
```

**The CNN contributes nothing to yield prediction in the current repository.**
It is a standalone 5-class wheat growth-stage image classifier. See
`docs/models.md` section 1 for the evidence and the diagnosis.

## 3. Leakage controls that were verified

| Control | Where | Verified |
|---|---|---|
| Stage aggregates use only data up to `stage_end_date` | `build_stage_table.py::aggregate_weather` / `aggregate_satellite` | Yes — date-window filter on every aggregate |
| 2024-25 rows never enter training | `_compute_split()` in both trainers; `df["split"] == "train"` filter | Yes — `n_train = 99` per stage = 33 districts x 3 years |
| Quintile thresholds from training years only | `train_quintile_classifier.py:187` — `pd.qcut` on `split == "train"` | Yes |
| LSTM imputer / scaler / one-hot fitted on train only | `train_lstm_sequence_model.py` — `fit` on `train_stack` and `train_mask` | Yes |
| LSTM target normalisation from train only | `y_mean`, `y_std` from `y_train` | Yes |
| CNN train/val/test images disjoint | `data/interim/cnn_dataset_split.csv` | Yes — 0 image paths in more than one split |
| 2025-26 satellite files excluded from the 2024-25 season | `ingest_satellite.py` season guard | Yes |

## 4. Known methodological caveats (documented, not silently changed)

0. **Shared feature policy (added 2026-09-06).** All four trainers now import
   `src/features/feature_policy.py` instead of each keeping a private exclusion
   list. It blocks identifiers, split keys, free text, **target-derived columns**
   (`actual_available`, `evaluation_status`, ...) and 8 exact-duplicate aliases.
   `scripts/smoke_test.py` fails if any model input violates it.

1. **Model selection uses the validation set.** For each stage,
   `train_stagewise_regression.py` fits all five candidate regressors and keeps
   the one with the lowest 2023-24 RMSE, then reports that same RMSE. The
   reported metrics are therefore selection-optimistic. With only 33 validation
   cases per stage, a nested or leave-one-year-out scheme would be more honest.
   The same applies to `train_quintile_classifier.py`.

2. **`configs/config.yaml: train_years` uses a different year convention from
   the split logic.** The config says `[2021, 2022, 2023, 2024]` but the split
   functions hard-code `season_year_start in {2020, 2021, 2022} -> train`.
   `build_stage_table.py:450` matches `train_years` against `season_year` (which
   equals `season_year_start`), so the quintile thresholds written into
   `data/processed/targets.csv` are computed over 2021-2024, i.e. they include
   the 2023-24 validation year. **That column is not consumed downstream** —
   `train_quintile_classifier.py` recomputes thresholds correctly from the train
   split — so no reported result is affected, but the config key is misleading
   and should be reconciled.

3. **Two identical modelling tables.** `build_stage_table.py` writes the same
   DataFrame to both `stage_features_with_cnn.csv` and
   `stage_features_with_split.csv`. The trainers prefer `_with_cnn`.
   `ingest_apy_2025_ground_truth.py::_write_stage_updates` previously rewrote
   only `_with_split`, letting the two drift apart; as of 2026-09-05 it refreshes
   `stage_features.csv`, `_with_split` and `_with_cnn` together. Both wide files
   are 11 MB and
   ~99% empty CNN columns; `data/processed/stage_features.csv` (858 KB, 55
   columns) holds every real feature and is the file tracked in Git.

4. **The 2024-25 forecast is satellite-blind outside Mehsana.** Only Mehsana has
   Sentinel-2 exports for the 2024-25 season, so 288 of the 297 forecast rows
   have NDVI / NDRE / EVI imputed to the training median. See
   `docs/dataset.md` section 5.

5. **The 2024-25 target column is empty by design.** District-level 2024-25 APY
   has not been published, so `yield_kg_ha` is null and `evaluation_status` is
   `pending_actual_apy` for all 297 forecast rows, and no 2024-25 metric is
   computed. A carry-forward fallback that had populated 9 of those rows with a
   2023-24 value was removed on 2026-09-05. See
   `docs/2025_yield_reference_methodology.md`.

## 5. Entry points

| Script | What it does | Cost |
|---|---|---|
| `scripts/smoke_test.py` | Read-only wiring check | seconds |
| `src/models/predict_stagewise.py` | Inference with saved models | seconds |
| `src/visualization/plot_results.py` | Regenerate figures / stage report | seconds |
| `run_baseline_ml.py` | **Retrains** regression + quintile classifiers | ~1-3 min |
| `run_training.py` | Full data refresh + retrain everything (needs network) | ~10-30 min |
| `run_training_with_cnn.py` | CNN feature merge + retrain everything | ~10-30 min |
| `src/deep_learning/train_*.py` | **Retrains a CNN** — do not run | ~4-63 min each on GPU |

See `docs/reproducibility.md` for the lightweight / expensive split.
