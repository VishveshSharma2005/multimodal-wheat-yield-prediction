# Reproducibility

---

## 1. Environment

### Reference environment

Every result under `reports/` was produced by this exact environment on
2026-09-06:

| Package | Version |
|---|---|
| Python | 3.10.16 |
| numpy | 1.24.3 |
| pandas | 2.2.3 |
| scikit-learn | **1.6.1** |
| scipy | 1.15.3 |
| joblib | 1.4.2 |
| matplotlib | 3.10.0 |
| torch | 2.6.0+cpu |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |

* **The `.joblib` artefacts are scikit-learn-version locked.** sklearn pickles are
  not portable across feature releases in either direction. Under 1.8 they fail
  with `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`;
  pickles made under 1.8 fail under 1.6 with
  `ModuleNotFoundError: No module named 'numpy._core.multiarray'`. Install the
  pinned versions, or retrain with `python run_baseline_ml.py`.
* **All five regressors must be installed together.** xgboost and lightgbm are
  optional at import time — if they are missing the trainers silently skip them
  and the 5-model comparison quietly becomes a 3-model comparison.
* The saved `.keras` CNN models need **Keras 3 (TensorFlow >= 2.16)**. The
  reference environment has TensorFlow 2.15 / Keras 2, so the smoke test reports
  `CNN model loads (TensorFlow)` as SKIP there. The weights are intact; only
  deserialisation needs the newer Keras.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# only if you need the CNN experiment:
pip install -r requirements_colab_cnn.txt
```

If `Activate.ps1` is blocked:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Conda

```bash
conda create -n wheat python=3.11 -y
conda activate wheat
pip install -r requirements.txt
```

### Verify

```bash
python scripts/smoke_test.py
```

Read-only. Nothing is trained, downloaded or overwritten. See section 5 for the
expected output.

---

## 2. What ships with the repository and what does not

| Needed for | Ships in Git? | If not, see |
|---|---|---|
| `scripts/smoke_test.py` | yes | — |
| Inference with saved regression / classifier / LSTM models | yes | — |
| `data/processed/stage_features.csv` (1485 x 55, the modelling table) | yes | — |
| Prediction + validation output CSVs | yes | — |
| All metrics CSVs and figures | yes | — |
| Re-running `predict_stagewise.py` | yes | — |
| Retraining regression / quintile / LSTM | **partly** | needs `stage_features_with_split.csv`, regenerate with `build_stage_table.py` |
| Rebuilding the stage table | **no** | needs raw weather + satellite, see `data/README.md` |
| CNN inference / feature extraction | **no** | needs `models/cnn/*.keras` (271 MB) and the image dataset |
| Retraining a CNN | **no** | needs the 5,211-image dataset + GPU |

---

## 3. LIGHTWEIGHT commands — safe to run

Seconds to a couple of minutes. No training, no downloads.

```bash
# read-only wiring / schema / artefact check
python scripts/smoke_test.py

# inference with the saved per-stage models -> rewrites the 3 prediction CSVs
python src/models/predict_stagewise.py

# regenerate stage-wise figures and the improvement report from existing predictions
python src/visualization/plot_results.py

# data-quality inspections (write reports only)
python src/data/verify_training_data.py
python src/data/diagnose_features.py
python src/data/check_satellite_files.py

# dry-run archive scan (add --apply to actually move files)
python src/utils/cleanup_project.py
```

### Robustness experiments (minutes, no artefact overwritten)

These fit models internally but write only to NEW report files. They never touch
`models/` and never overwrite the authoritative hold-out metrics.

```bash
# walk-forward temporal CV + the three naive baselines
python src/models/run_temporal_validation.py

# within-district (anomaly) skill of the saved validation predictions
python src/models/run_within_district_skill.py
```

> `predict_stagewise.py` overwrites the three prediction CSVs and
> `reports/model_results/prediction_2025_evaluation.md`. It runs inference with
> the saved models only - no model is refitted.

---

## 4. EXPENSIVE commands — will overwrite authoritative results

Do not run these casually. Each **overwrites** metrics files and model artefacts,
and results will not be bit-identical (LSTM and library-version drift).

| Command | Cost | Overwrites |
|---|---|---|
| `python run_baseline_ml.py` | ~1-3 min CPU | all regression + quintile models and metrics |
| `python src/models/train_stagewise_regression.py` | ~1 min CPU | 10 regression `.joblib` + stagewise/combined metrics |
| `python src/models/train_quintile_classifier.py` | ~20 s CPU | 9 classifier `.joblib` + quintile metrics |
| `python src/models/train_lstm_sequence_model.py` | ~30 s CPU | `lstm_yield_model.pt`, `lstm_metrics.csv`, `lstm_training_log.csv`, `lstm_summary.md`, `lstm_predictions_2025.csv`, `lstm_training_loss.png` (deterministic under seed 42) |
| `python src/models/run_cnn_ablation_study.py` | ~1 min CPU | `cnn_ablation_study.csv` |
| `python run_training.py` | ~10-30 min + **network** | the whole interim/processed tree and every model |
| `python run_training_with_cnn.py` | ~10-30 min | stage table + every model |
| `python src/deep_learning/extract_cnn_features.py` | ~20-40 min GPU | 618 MB `cnn_features.csv` + 5 per-model feature files |
| `python src/deep_learning/train_<model>.py` | 4-63 min GPU **each** | **a trained CNN — do not run** |
| `python src/deep_learning/run_cnn_pipeline.py` | ~2.5 h GPU | **all five CNNs — do not run** |

**The five CNN models are locked.** They are trained, evaluated, and treated as
read-only artefacts. Nothing in the lightweight path touches them.

`run_training.py` additionally calls the Open-Meteo API
(`src/data/download_weather.py`) and expects the raw APY workbooks and Sentinel-2
exports to be present locally.

---

## 5. Expected smoke-test output

With `requirements.txt` installed (no TensorFlow):

```
-- environment --
[PASS] python version - 3.11.x
[PASS] import pandas / numpy / yaml / sklearn / joblib / matplotlib

-- configuration --
[PASS] load configs/config.yaml - 9 stages defined
[PASS] configured paths resolve - 8/12 configured dirs exist

-- source tree --
[PASS] all sources compile - 51 files compile cleanly
[PASS] src.utils importable

-- datasets --
[PASS] stage_features.csv schema - 1485 rows, 33 districts x 9 stages x 5 years
[PASS] 2024-25 ground-truth integrity - correctly empty across 8 artefacts
[PASS] no leaked features in any model - regression=35, lstm=76
[PASS] LSTM checkpoint contract - epoch 28/43, val_rmse 113.38, seed 42
[PASS] temporal CV / baselines / within-district - 3 folds, 5 models vs 3 baselines
[PASS] no 2025 metrics without ground truth - MAPE/RMSE/MAE/R2 all N/A
[PASS] prediction output row counts - 297 / 297 / 9

-- model artefacts (load only) --
[PASS] regression pipelines load - 10 regression pipelines load
[PASS] regression inference on 5 rows
[PASS] quintile classifiers load - 9 quintile classifiers load
[PASS] LSTM checkpoint loads - input_size=84, max_len=9
[SKIP] CNN .keras files present        (271 MB, not in Git)
[SKIP] CNN model loads (TensorFlow)

-- outputs --
[PASS] output directories
```

`configured paths resolve` reports 8/12 because `config.yaml` declares
`weather_dir`, `drone_dir`, `farm_logs_dir` and `soil_dir`, which this project
never populated. That is expected.

`2024-25 ground-truth integrity` is a permanent regression check: it fails if any
`season_year_start == 2024` row ever carries a non-null yield again. It caught a
carry-forward fallback that had written a 2023-24 value into 9 Mehsana 2024-25
rows; that fallback was removed on 2026-09-05 and the check now passes. See
`docs/2025_yield_reference_methodology.md`.

---

## 6. Rebuilding from raw data

Only if you have restored the raw inputs listed in `data/README.md`:

```bash
python src/data/extract_apy_all_districts_xlsx.py   # APY xlsx -> district table
python src/data/create_prediction_2025_cases.py     # 2024-25 case rows
python src/data/build_coordinate_registry.py        # district lat/lon
python src/data/download_weather.py                 # NETWORK: Open-Meteo archive
python src/data/check_satellite_files.py            # audit GEE exports
python src/data/ingest_satellite.py                 # -> satellite_observations.csv
python src/features/extract_image_features.py       # handcrafted image features
python src/data/build_image_stage_features.py       # per-stage image averages
python src/data/build_stage_table.py                # -> stage_features*.csv
python src/data/verify_training_data.py             # sanity report
```

Then retrain with `run_baseline_ml.py` and
`src/models/train_lstm_sequence_model.py`.

**Do not run `src/features/extract_image_features.py` without restoring the full
Mendeley auxiliary set.** 335 of the 560 referenced images are missing from this
copy; re-running it would silently change the per-stage image averages that the
saved models were fitted on.

`src/data/ingest_apy_2025_ground_truth.py` is safe to run: its carry-forward
fallback was removed on 2026-09-05. With no official district-level 2024-25 APY
present it writes an empty ground-truth table and leaves the 2024-25 target null
(`pending_actual_apy`) in `cases.csv` and all three stage tables.
---

## 7. Determinism

* Regression, quintile classifiers, CNNs: `random_state` / `tf.random.set_seed`
  = 42. Reproducible up to library-version differences.
* **Fixed data splits.** Train = `season_year_start` in {2020, 2021, 2022};
  validation = 2023; forecast = 2024. Hard-coded identically in every trainer, not
  randomly sampled, so there is no split seed to set.
* **Temporal CV folds are deterministic** (forward-chaining by year, no sampling).
* **LSTM: seeded with 42 since 2026-09-05** — Python, NumPy, PyTorch, the CUDA
  RNGs, cuDNN determinism flags and the training DataLoader's shuffle generator.
  Verified: two consecutive runs produce byte-identical `lstm_yield_model.pt`,
  `lstm_metrics.csv`, `lstm_training_log.csv` and `lstm_predictions_2025.csv`.
  Earlier unseeded runs recorded RMSE 202.51 and 166.69; the current seeded run
  records 148.81.
* Model *selection* in the regression and quintile trainers depends on validation
  RMSE / F1, so a re-run that changes any metric can also change which model is
  saved.
