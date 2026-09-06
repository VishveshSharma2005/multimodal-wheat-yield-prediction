# Project Audit and Cleanup Report

**Date:** 2026-09-05 (audit + cleanup, then Fix #1 applied)
**Repository:** `D:\BIG_Works_Done\Crop_yeild_predictions`
**Scope:** full audit, cleanup, documentation, local-runnability and GitHub preparation.

**Guardrails observed:** no model of any kind was retrained. All 25 model
artefacts (5 CNN, 10 regression, 9 classification, 1 LSTM) are byte-identical to
their pre-cleanup state, verified by MD5. No validation or training metric was
recomputed or edited.

Two scientific defects were found and **both have now been fixed** at the
owner's instruction: **Fix #1** (2024-25 ground-truth contamination, section 26A)
and **Fix #2** (LSTM best-checkpoint bug, section 26C).

Model-artefact accounting: 24 of 25 artefacts (5 CNN, 10 regression, 9
classification) are byte-identical to their pre-cleanup state, verified by MD5.
The single exception is `models/deep_learning/lstm_yield_model.pt`, deliberately
retrained under Fix #2. **No CNN, regression or classification model was
retrained.**

---

## 1. Repository status

**Ready** — engineering-clean, GitHub-ready, leakage-free, and independently
validated. Three research-integrity defects were found and fixed (sections 26A,
26C, 26D). The headline scientific finding is negative and is reported as such:
no model beats a naive district-mean baseline under temporal validation.

* Was **not** a Git repository. `git init` has been run; nothing has been committed or pushed.
* 196 files / **~14 MB** staged for the first commit. Largest tracked file 1.1 MB.
* ~4.1 GB of raw and derived data correctly excluded.
* No secrets, tokens, credentials or `.env` files anywhere in the tree.
* `scripts/smoke_test.py`: **31 PASS, 3 SKIP, 0 FAIL**.

## 2. Repository structure (after cleanup)

```
configs/         config.yaml
data/            raw/ (mostly ignored) | interim/ | processed/   + 4 READMEs
docs/            pipeline, dataset, models, results, reproducibility,
                 2025_yield_reference_methodology                 [NEW]
models/          cnn/ (5 .keras, ignored) | regression/ (10) |
                 classification/ (9) | deep_learning/ (1)
r_and_d/         R&D.docx, flow_diagrams.docx, update.docx, docs/
reports/         model_results/ (58) | figures/ (4) | summaries/ (1) + README  [NEW]
scripts/         smoke_test.py                                     [NEW]
src/             data/ (20) | features/ (6) | models/ (6) |
                 deep_learning/ (11) | visualization/ (1) | utils/ (4)
archive/         local-only: scratch notebooks, superseded root reports  [NEW, ignored]
run_training.py  run_training_with_cnn.py  run_baseline_ml.py
requirements.txt requirements_colab_cnn.txt  .gitignore  .gitattributes
```

## 3. Architecture as previously described

Agricultural data -> preprocessing -> temporal/stage alignment -> **CNN feature
extraction -> multi-modal feature fusion** -> stage-wise regression -> temporal
LSTM -> final yield -> Q1-Q5 classification.

## 4. Final verified architecture

```
APY yield + Open-Meteo weather + Sentinel-2 indices + crop calendar
        -> stage-wise feature table (1485 x 55, 9 DAS windows per case)
        -> per-stage GradientBoosting regression  -> final yield
        -> LSTM over the 9-stage sequence         -> final yield
        -> per-stage quintile classifier          -> Q1..Q5

Field images -> 5 CNN classifiers -> growth-stage class   [SEPARATE, NOT FUSED]
```

The **CNN -> fusion -> regression** link **does not exist**. See section 9.

## 5. Dataset years

Training 2020-21, 2021-22, 2022-23 (891 rows). Validation 2023-24 (297 rows).
Forecast 2024-25 (297 rows, target unpublished). Verified from
`season_year_start` in `data/processed/stage_features.csv`.

## 6. District count

**33**, verified. 33 x 5 seasons = 165 cases; 165 x 9 stages = 1485 rows, with no
duplicate `(district, year, stage)` keys.

## 7. Growth stages

All 9 expected stages present and correctly ordered; `stage_end_das` is
monotonically increasing within every one of the 165 cases. Crop duration
125-126 days.

## 8. Feature modalities

| Modality | Columns | Reaches a model? |
|---|---|---|
| Weather (Open-Meteo ERA5) | 12 | **Yes** |
| Satellite (Sentinel-2 NDVI/NDRE/EVI) | 11-12 | **Yes** |
| Location + crop calendar | ~10 | **Yes** |
| Handcrafted image features | 6 | Yes, but constant within each per-stage model -> zero information |
| CNN deep features | 6,913 | **No — 100% NaN, dropped** |
| Soil | 0 | Not implemented at all |

Weather is **Open-Meteo**, not NASA POWER. There are **no soil features**.

## 9. CNN models preserved

All five preserved **unchanged, unretrained, undeleted**:

| Path | Size | Test accuracy | Epochs run |
|---|---|---|---|
| `models/cnn/xception_best.keras` | 84.1 MB | 0.9923 | 29 |
| `models/cnn/inceptionv3_best.keras` | 88.4 MB | 0.9616 | 21 |
| `models/cnn/vgg16_best.keras` | 59.0 MB | 0.9731 | 21 |
| `models/cnn/densenet121_best.keras` | 29.7 MB | 0.9847 | 20 |
| `models/cnn/mobilenetv2_best.keras` | 9.7 MB | 0.9015 | 14 |

Created by `src/deep_learning/train_<name>.py`. Architecture: ImageNet backbone
(frozen) -> `GlobalAveragePooling2D("cnn_gap")` -> `Dropout(0.2)` ->
`Dense(5, softmax)`. Task: 5-class wheat growth-stage image classification.
Configured for up to 100 epochs; early stopping (patience 5,
`restore_best_weights=True`) halted them at 14-29 epochs.

**Downstream usage: none.** `build_cnn_stage_features.py` writes
`cnn_stage_features.csv` keyed on the *image* class names (`1_Tillering`,
`2_Jointing`, `3_BH`, `4_Flowering`, `5_Filling`), while
`build_stage_table.py::_merge_cnn_stage_features()` merges it into the stage
table on `stage`, which uses agronomic names. No key matches, so all 6,913
`cnn_*` columns are NaN and are dropped by the ">90% missing" filter.

Independent confirmations already in the repository:

* `stage_features_with_cnn.csv`: 0 of 1485 rows carry any CNN value
* `ml_selected_features.csv`: 62,217 `cnn_*` rows dropped "missing > 90%", 0 selected
* `lstm_summary.md`: every `cnn_feat_*` listed "missing 100.00%"
* `cnn_ablation_study.csv`: "+Handcrafted" and "+CNN" rows byte-identical (both feature_count 31)
* `cnn_improvement_report.md`: "Feature count increase: 0, RMSE improvement: 0.0"
* every prediction row: `cnn_available_flag = 0`

## 10. Regression models

`RandomForest`, `GradientBoosting`, `HistGradientBoosting`, `LightGBM`, `XGBoost`
— all five present and all five ran successfully (45 `status = ok` rows).

**GradientBoostingRegressor is the best model on all 9 stages** and on the pooled
model, per `reports/model_results/stagewise_regression_metrics.csv`. **This
result is preserved unchanged.** Mean validation RMSE: GB 212.3, RF 272.4,
XGB 275.6, LightGBM 593.2, HistGB 623.8.

Artefacts: 10 `models/regression/*.joblib`, all verified loadable and able to run
inference.

## 11. LSTM status

**Implemented, trained, evaluated, preserved.** PyTorch, 2x64 hidden, dropout
0.2, 84 input features, 9 timesteps, 99/33/33 sequences. Imputer, scaler,
one-hot encoder and target normalisation all fitted on training rows only — no
leakage. Validation: **MAE 119.63, RMSE 148.81, MAPE 3.98%, R2 0.9618**, restored
from the best epoch (40 of 55 run), seed 42, reproducible.
Supersedes the pre-fix RMSE 166.69 — see section 26C.

**Defect FIXED (section 26C):** `best_state = model.state_dict()` stored tensor
references, so the checkpoint held the final epoch rather than the best. Now
deep-copied, and the run is seeded. Current result below.

## 12. Quintile classification status

**Implemented, trained, evaluated, preserved.** Q1-Q5, one classifier per stage,
`RandomForestClassifier` vs `GradientBoostingClassifier`.

Thresholds are computed with `pd.qcut` on the **training split only**
(`train_quintile_classifier.py:187`) — **verified, no leakage**.

Validation accuracy 0.4545-0.8485; macro F1 0.4390-0.8461. 9 confusion matrices
present. 9 `models/classification/*.joblib` verified loadable.

## 13. Training setup

99 rows/stage (33 districts x 3 seasons); 891 rows for the pooled and LSTM
models. `random_state = 42` everywhere except the LSTM.

## 14. Validation setup

2023-24, 33 rows/stage, 297 total. All reported metrics are validation metrics —
**no training predictions are presented as validation anywhere**. Raw
district-level predictions are preserved in
`data/processed/validation_stagewise_predictions.csv` (297 rows); the figures
plot a stage-wise aggregate (297 grouped by stage), documented in
`docs/results.md` section 8.

**Caveat:** the best model per stage is selected on the same validation set it is
then scored on, so the headline metrics are selection-optimistic.

## 15. 2024-25 prediction setup

297 forecast rows (33 x 9) + a 9-row Mehsana view + 33 LSTM case forecasts.
Only Mehsana has 2024-25 Sentinel-2 exports, so 288 of the 297 rows have
median-imputed vegetation indices.

## 16. 2024-25 APY ground-truth status

**No official district-level 2024-25 Gujarat wheat APY exists.** The only 2024-25
figure is a **state-level** Final Advance Estimate of 3280.69 kg/ha.

**Contamination found and FIXED (section 26A).** A fallback branch in
`src/data/ingest_apy_2025_ground_truth.py` (`if not frames:`) carried the most
recent district value forward when the 2024-25 filter returned nothing. It had
written Mehsana's **2023-24** yield (2796.06 kg/ha) into the 9 Mehsana 2024-25
rows with `actual_available = 1`, `evaluation_status = "available"`, producing
the invalid "Mehsana 2025 MAPE 11.93 / RMSE 348.85 / MAE 333.67 / R2 0.0" in
`reports/model_results/prediction_2025_evaluation.md`.

The fallback has been removed and the affected outputs regenerated. All 297
2024-25 rows now carry a null target and `pending_actual_apy`, and that report
prints `N/A` for every metric.

The value is doubly wrong: `apy_wheat_trace.csv` (manual PDF transcription) gives
Mehsana 2023-24 = 2796.06, while the machine-parsed XLSX table gives 3255.21, and
the trace file's area figures (389.00 vs 671.46 thousand ha) indicate a row
misalignment during transcription.

**Actions taken:** fallback removed from the ingestion script; contaminated
target cleared across `cases.csv` and all three stage tables; prediction outputs,
the 2025 evaluation report and the Mehsana figure regenerated by **inference
only**; full record in `docs/2025_yield_reference_methodology.md`; permanent
regression check in `scripts/smoke_test.py`, which now passes.

## 17. Proxy / reference methodology

**No proxy was created**, and none should be created to make the results look
complete. If a comparison anchor is needed, the defensible options — in order —
are: wait for the official district release; quote the Gujarat **state** 2024-25
benchmark of 3280.69 kg/ha against the state aggregate of the forecasts; or
report the Mehsana 2020-21..2023-24 observed band (3169.33-3389.73 kg/ha) as a
plausibility check. Any such value must be named `state_benchmark_yield_kg_ha`
or `reference_2025_yield_kg_ha` — never `actual_yield_kg_ha`.

## 18. Authoritative result files

```
reports/model_results/stagewise_regression_metrics.csv     <- headline regression table
reports/model_results/combined_regression_metrics.csv
reports/model_results/stagewise_regression_summary.md
reports/model_results/quintile_classification_metrics.csv
reports/model_results/confusion_matrix_<stage>.csv          (9)
reports/model_results/lstm_metrics.csv
reports/model_results/lstm_training_log.csv
reports/model_results/cnn_evaluation_metrics.csv            <- held-out CNN test set
reports/model_results/cnn_model_comparison.csv
reports/model_results/cnn_ablation_study.csv
reports/model_results/feature_importance_*.csv              (10)
reports/model_results/stagewise_improvement_report.md
reports/summaries/final_model_summary.csv                   <- NEW consolidated table
data/processed/validation_stagewise_predictions.csv         (297 raw)
data/processed/prediction_2025_stagewise_predictions.csv    (297 forecast)
data/processed/mehsana_2025_stagewise_predictions.csv       (9 forecast)
data/processed/lstm_predictions_2025.csv                    (33 forecast)
reports/figures/*.png                                       (4)
```

`reports/README.md` classifies **every** file under `reports/` as AUTHORITATIVE /
CONTAMINATED / SUPERSEDED / DIAGNOSTIC / LARGE.

## 19. Files created

| Path | Purpose |
|---|---|
| `.gitignore` | excludes ~4.1 GB of data, 271 MB of CNN weights, caches, secrets |
| `.gitattributes` | LF normalisation + binary markers |
| `CLEANUP_REPORT.md` | this file |
| `docs/pipeline.md` | verified data/model flow, leakage controls, caveats |
| `docs/dataset.md` | years, districts, stages, every feature modality, missingness |
| `docs/models.md` | CNN / regression / LSTM / quintile, artefacts, defects |
| `docs/results.md` | all authoritative numbers + ablation + safe headline claim |
| `docs/reproducibility.md` | environment, lightweight vs expensive commands, rebuild |
| `docs/2025_yield_reference_methodology.md` | **the 2024-25 ground-truth analysis** |
| `data/README.md` | what is excluded, why, sources, expected paths |
| `reports/README.md` | status of every report file |
| `reports/summaries/final_model_summary.csv` | 26-row consolidated model table |
| `scripts/smoke_test.py` | read-only verification (never trains) |

## 20. Files modified

| Path | Change |
|---|---|
| `README.md` | rewritten from scratch, verified against the code |
| `requirements.txt` | removed unused `seaborn`, `opencv-python`, `torchvision`; added `openpyxl`, `pillow`, `tabulate`; pinned `numpy>=2.0` and `scikit-learn>=1.8` (required to unpickle the saved models); moved TensorFlow out |
| `requirements_colab_cnn.txt` | reduced to `tensorflow>=2.16` (Keras 3 needed for the `.keras` files) |
| `run_training.py`, `run_baseline_ml.py` | `.venv` detection now also finds `bin/python`, so it works on Linux/macOS as well as Windows |
| `data/raw/README.md`, `data/interim/README.md`, `data/processed/README.md` | replaced generic placeholders with a real inventory |
| `reports/model_results/prediction_2025_evaluation.md` | regenerated by `predict_stagewise.py`; now reports `pending_actual_apy` and `N/A` for every 2025 metric |
| `reports/model_results/baseline_ml_report.md` | regenerated from the current metrics CSVs via `run_baseline_ml._write_report()` — **no model retrained**; removes a stale snapshot that claimed XGBoost won every stage with RMSE ~100 |

No file under `src/`, `models/`, `configs/`, `data/interim/` or
`data/processed/` had its scientific content changed.

## 21. Files moved

| From | To | Reason |
|---|---|---|
| `CropYeild_main.ipynb` | `archive/notebooks/` | 0 bytes, empty |
| `Testing.ipynb`, `Untitled.ipynb` | `archive/notebooks/` | GPU/CUDA scratch cells, hardcoded `/home/student/...` |
| `src/deep_learning/RDC_3.ipynb`, `phd.ipynb` | `archive/notebooks/` | Colab/Linux driver notebooks; `run_cnn_pipeline.py` is the real entry point |
| `CLEAN_PROJECT_REPORT.md`, `PROJECT_STRUCTURE_AFTER_CLEANUP.md` | `archive/stale_root_reports/` | describe an `archive_unused/` folder and a `D:\Crop_yeild_predictions` root that do not exist here |
| `reports/data_quality_report.md`, `reports/image_dataset_notes.md` | `archive/report_templates/` | unfilled `[__]` templates; superseded by `docs/dataset.md` |
| `flow_diagrams.docx`, `update.docx` | `r_and_d/` | tidy the repository root |
| `.venv/` | `.venv_broken_linux_leftover/` | broken venv from `/home/student/...`, Linux layout, contains only pip |

`archive/` and `.venv_broken_linux_leftover/` are gitignored.

## 22. Files deleted

Only exact duplicates and caches — each verified byte-identical to a live file
before removal:

* 6 `src/**/__pycache__/` directories
* 4 `.ipynb_checkpoints/` directories (13 files; the 7 `.py` and 5 result files
  were `diff`-verified identical to their live counterparts)
* `anaconda_projects/db/project_filebrowser.db` (IDE state)

**No dataset, model artefact, metrics file or research output was deleted.**

## 23. Large datasets excluded from GitHub

| Item | Size | Why |
|---|---|---|
| `data/raw/images/` (5,437 files) | 2.9 GB | image corpora do not belong in Git |
| `data/processed/cnn_features/` (5 CSVs) | 620 MB | regenerable deep-feature dumps |
| `data/processed/cnn_features.csv` | 618 MB | regenerable |
| `data/raw/apy/*.xlsx` / `*.pdf` | 19 MB | government publications; the extracted table is committed instead |
| `data/processed/stage_features_with_{cnn,split}.csv` | 11 MB each | 6,913 of 6,969 columns are empty; `stage_features.csv` (858 KB) has all real columns |
| `data/interim/weather_daily*.csv` | 5.5 MB | regenerable from the Open-Meteo API |
| `data/interim/satellite_observations.csv` | 1.7 MB | regenerable from the GEE exports |
| `data/raw/satellite/` (153 CSVs) | 1.2 MB | regenerable; small enough to commit if preferred |
| `reports/model_results/ml_selected_features.csv` | 3.9 MB | 62k-row audit trail |
| `reports/feature_diagnosis_report.md` | 507 KB | 6,969-column missingness dump |
| `.venv_broken_linux_leftover/` | 126 MB | dead virtualenv |

## 24. Large models excluded from GitHub

`models/cnn/*.keras` — **271 MB total**. Preserved on disk, excluded from Git.
Recommended distribution: **Zenodo** (gives a citable DOI) or a GitHub Release;
Git LFS is possible but 271 MB per clone will exhaust GitHub's 1 GB/month free
LFS bandwidth quickly. Expected local path: `models/cnn/<name>_best.keras`.

All small artefacts **are** committed: 10 regression `.joblib` (1.3 MB), 9
classification `.joblib` (9.5 MB), 1 LSTM `.pt` (294 KB).

## 25. Local smoke-test result

`python scripts/smoke_test.py` — **23 PASS, 6 SKIP, 0 FAIL**.

Verified:

* Python version and all required imports
* `configs/config.yaml` loads; 8 of 12 configured directories exist (the 4 absent
  ones — `weather`, `drone`, `farm_logs`, `soil` — were never populated)
* **51 source files compile cleanly**; `src.utils` imports
* `stage_features.csv`: 1485 rows, 33 districts, 9 correct stages, no duplicate
  keys, monotonic DAS in all 165 cases
* prediction outputs: 297 / 297 / 9 rows
* **all 10 regression pipelines load**, and inference on 5 real rows returns
  2836.3 kg/ha mean
* **all 9 quintile classifiers load**
* **LSTM checkpoint loads** (`input_size=84`, `max_len=9`) — verified in a
  torch-enabled environment
* all 5 CNN `.keras` files present (271 MB)
* output directories creatable

`2024-25 ground-truth integrity` now **passes** — the 2024-25 target is correctly
empty. It remains in place as a permanent regression guard.

Environment note: the saved `.joblib` files were pickled under scikit-learn 1.8.0
/ NumPy 2.x and **cannot** be loaded by scikit-learn 1.6 (confirmed:
`ModuleNotFoundError: No module named 'numpy._core.multiarray'`). The `.keras`
files need **Keras 3 / TensorFlow >= 2.16** (confirmed: Keras 2 raises
`Could not deserialize class 'Functional'`). Both pins are now in the
requirements files.

## 26A. Fix #1 applied — 2024-25 ground-truth contamination

Applied at the owner's instruction. Code changes, all in
`src/data/ingest_apy_2025_ground_truth.py`:

1. Deleted the `if not frames:` carry-forward fallback. With no district-level
   prediction-season row the function now returns an empty frame and logs a
   warning. **No proxy or estimated actual is created.**
2. `_write_cases_updates()` builds its lookup from prediction-season rows only
   (`season_year_start in PREDICTION_SEASON_START_YEARS`) and assigns
   unconditionally, so a stale value from an earlier run is cleared rather than
   retained.
3. `_write_stage_updates()` now refreshes all three stage tables
   (`stage_features.csv`, `_with_split`, `_with_cnn`) instead of only
   `_with_split`, which the models do not read.
4. Removed the "re-use the previous output" branch in `main()`, which
   re-injected the contaminated file on every run.
5. `_is_district_level()` now rejects aggregate district names
   (`ALL_DISTRICTS`, `STATE`, `TOTAL`, ...) and no longer has its `scope` column
   overwritten, so the state-level 2024-25 Final Advance Estimate can never be
   admitted as a district target.

Commands run — **inference and reporting only, no fitting**:

```
python src/data/ingest_apy_2025_ground_truth.py   # data prep, no model touched
python src/models/predict_stagewise.py            # loads saved .joblib, predicts
python src/visualization/plot_results.py          # replots from existing predictions
python scripts/smoke_test.py                      # read-only verification
```

Verification:

| Check | Result |
|---|---|
| `cases.csv` 2024-25 rows with a target | **0 / 33** |
| All three stage tables, 2024-25 rows with a target | **0 / 297** |
| 2024-25 `evaluation_status` | `pending_actual_apy` (297 / 297) |
| Training + validation targets | **1188 / 1188 intact**, unchanged |
| `predicted_yield_kg_ha` (all three prediction files) | **bit-identical**, max abs diff 0.0 |
| `model_name` per row | unchanged |
| Validation metrics (`stagewise_improvement_report.md`) | **byte-identical** |
| Model artefacts | **25 / 25 byte-identical (MD5)** |
| 2025 MAPE / RMSE / MAE / R2 | now `N/A` |
| Smoke test | 23 PASS, 6 SKIP, **0 FAIL** |

## 26B. Remaining issues

1. **No CNN-regression fusion.** Repairing the merge key alone would not help —
   the design broadcasts one mean vector per stage to all districts and years,
   which is constant within each per-stage model. Real fusion needs per-case
   imagery that this project does not have.
2. **Model selection on the validation set** inflates the reported regression and
   quintile metrics. Nested or leave-one-year-out validation would be more honest.
3. **`config.train_years` uses a different year convention** from the hard-coded
   split logic, so the `yield_quintile` column in `targets.csv` is computed over
   2021-2024 (including the validation year). **That column is not consumed by
   any model**, so no result is affected, but the config key is misleading.
4. **335 of 560 auxiliary images are missing** from this copy. Re-running
   `extract_image_features.py` would change the features the saved models were
   fitted on.
5. **Handcrafted image features carry no information** inside a per-stage model
   (they are global stage constants). The ablation shows they make the pooled
   model slightly worse.
6. **2024-25 satellite gap** — only Mehsana has 2024-25 Sentinel-2 data, so 288
   of 297 forecast rows have imputed vegetation indices.
7. **Duplicated modelling tables.** `stage_features_with_cnn.csv` and
    `stage_features_with_split.csv` are written identically. The divergent-update
    half of this problem is fixed (the ingestion script now refreshes all three
    tables together), but the duplication itself remains.
8. **`area_ha` is 100% empty**; `lat`/`lon` exist for only 14 of 33 districts.
9. **Provenance paths.** Generated result artefacts still contain
    `/home/student/...` and `D:\Crop_yeild_predictions\...` strings in `Dataset:`
    lines and `model_path` columns. These are provenance records, not executable
    paths — no active code depends on an absolute path. Left as-is so no result
    file is edited.
10. **`.venv_broken_linux_leftover/`** — 126 MB of dead virtualenv from
    `/home/student/...`. Gitignored; safe to delete manually.

## 26C. Fix #2 applied — LSTM best-checkpoint bug

Applied at the owner's instruction. Changes, all in
`src/models/train_lstm_sequence_model.py`:

1. **`best_state = copy.deepcopy(model.state_dict())`** — the actual fix.
   `state_dict()` returns tensors that alias the live parameters, so the snapshot
   was mutated in place by every later epoch and the "best" checkpoint silently
   became the final epoch. Applied at both assignment sites.
2. **`_set_seeds(torch)`** seeds Python, NumPy, PyTorch and the CUDA RNGs, and
   sets `cudnn.deterministic = True` / `benchmark = False`. Seed 42, matching the
   rest of the project.
3. **Seeded `torch.Generator`** passed to the training `DataLoader` so the
   shuffle order is reproducible.
4. **`best_epoch` tracked** and logged on restore; `best_epoch`, `best_val_rmse`,
   `epochs_run` and `random_seed` are now stored in the checkpoint for audit.

Architecture, input features, and the train/validation split were **not** touched:
`LSTM(input_size=84, hidden_size=64, num_layers=2, dropout=0.2) -> Linear(64, 1)`,
84 features, 9 timesteps, 99 train / 33 validation / 33 forecast sequences —
all identical to the pre-fix run.

Command run — **the LSTM only**:

```
python src/models/train_lstm_sequence_model.py
```

### Proof the checkpoint is now correct

| | Epoch | val_rmse |
|---|---|---|
| Best epoch (restored) | **40** | **148.8119** |
| Last epoch (what the bug used to save) | 55 | 166.8315 |
| Reported in `lstm_metrics.csv` | — | **148.8119** |

The reported metric matches the **best** epoch and not the last. Before the fix
the reverse held exactly: reported 166.6919 == last epoch 34, while best epoch 19
scored 113.9027.

### Old vs corrected result

| Run | Checkpoint kept | MAE | RMSE | MAPE % | R2 |
|---|---|---|---|---|---|
| Pre-fix (unseeded) | final epoch 34 — **bug** | 120.59 | **166.69** | 4.03 | 0.9521 |
| **Corrected (seed 42)** | **best epoch 40** | **119.63** | **148.81** | **3.98** | **0.9618** |

The corrected run is also freshly seeded, so the two rows come from different
training trajectories — a before/after of the pipeline, not two evaluations of
the same weights. **Do not cite 166.69.**

### Reproducibility

Two consecutive runs produced **byte-identical** `lstm_yield_model.pt`,
`lstm_metrics.csv`, `lstm_training_log.csv` and `lstm_predictions_2025.csv`
(MD5-verified). The LSTM was previously the only unseeded model in the project.

### Files changed by Fix #2

| File | Change |
|---|---|
| `src/models/train_lstm_sequence_model.py` | deepcopy fix + seeding + checkpoint metadata |
| `models/deep_learning/lstm_yield_model.pt` | retrained (LSTM only), now epoch 40 |
| `reports/model_results/lstm_metrics.csv` | RMSE 166.69 -> 148.81 |
| `reports/model_results/lstm_training_log.csv` | 34 -> 55 epochs |
| `reports/model_results/lstm_summary.md` | regenerated (gitignored) |
| `reports/model_results/lstm_selected_features.csv` | regenerated (gitignored), still 84 features |
| `reports/figures/lstm_training_loss.png` | regenerated, 55 epochs |
| `data/processed/lstm_predictions_2025.csv` | regenerated; 2024-25 actual stays null (Fix #1 holds) |
| `reports/summaries/final_model_summary.csv` | LSTM row updated |

Verified untouched: all 5 CNN `.keras`, all 10 regression `.joblib`, all 9
classification `.joblib` — byte-identical by MD5.

## 26D. Research-readiness pass — 2026-09-06

Phases: leakage fix -> retrain affected models -> robust evaluation -> feature
audit -> tests -> docs. **CNN weights untouched throughout (5/5 MD5-identical).**

### Leakage / feature-policy fix

Created `src/features/feature_policy.py` as the single source of truth; all four
trainers (`train_stagewise_regression`, `train_quintile_classifier`,
`train_lstm_sequence_model`, `run_cnn_ablation_study`) now import it instead of
each carrying a private, already-diverged copy.

| Removed | Count | Why |
|---|---|---|
| `actual_available` | 1 | Target-availability indicator: constant 1 in train/validation, 0 in every 2024-25 forecast row. Train/serve distribution shift. |
| `evaluation_status` | 1 | Same class of bookkeeping column. |
| Exact-duplicate aliases | 8 | `days_after_sowing`, `cumulative_rain_mm`, `cumulative_rainfall`, `mean_tmin_c`, `mean_tmax_c`, `gdd_cumulative`, `cumulative_gdd`, `ndvi_growth_rate` — verified elementwise identical to a retained column. |

**Also fixed a pre-existing bug:** a bare `"text"` token in the free-text guard
matched `"texture"` and had been silently dropping
`image_texture_proxy_stage_avg` — a legitimate numeric image feature — from every
model since the project began. Matching is now on `_`-separated word boundaries.

Full per-column decisions: `reports/model_results/feature_audit.csv`.

### Models retrained (CNN excluded)

| Model | Original (leaky) | Corrected | Note |
|---|---|---|---|
| Stage-wise regression | mean RMSE 212.26 | **209.68** | GradientBoosting still wins all 9 stages |
| Combined regression | RMSE 129.39, R2 0.9712 | **132.94, R2 0.9695** | Slightly worse — kept as-is |
| LSTM | RMSE 148.81 | **113.38, R2 0.9778** | Best epoch 28/43, seed 42, byte-reproducible |
| Quintile (best stage acc) | 0.8485 | **0.8182** | Slightly worse — kept as-is |

Originals preserved as `reports/model_results/*_original.csv` / `*_original.md`;
original artefacts archived locally to `archive/models_pre_leakage_fix/`.

### New robust evaluation — the headline finding

**Walk-forward temporal CV** (3 forward-chaining folds; a fold never sees data at
or after its validation year — asserted by the smoke test) and **three naive
baselines** on the same folds:

| Kind | Name | RMSE (mean ± sd) | R2 |
|---|---|---|---|
| **baseline** | **A: district historical mean** | **155.25 ± 56.39** | **0.9567** |
| baseline | B: persistence | 158.96 ± 46.66 | 0.9558 |
| model | LightGBMRegressor | 169.50 ± 33.02 | 0.9510 |
| baseline | C: district identity only | 189.66 ± 41.72 | 0.9385 |
| model | RandomForestRegressor | 216.70 ± 90.72 | 0.9149 |
| model | GradientBoostingRegressor | 222.11 ± 98.38 | 0.9095 |
| model | XGBoostRegressor | 267.97 ± 175.01 | 0.8522 |
| model | HistGradientBoostingRegressor | 503.19 ± 343.15 | 0.4641 |

**No model beats the district-historical-mean baseline.** The 2023-24 hold-out
winner (GradientBoosting) is only 3rd under temporal CV, behind LightGBM —
direct evidence of model-selection optimism.

**Within-district skill:** removing each district's training-season mean gives
anomaly **R2 = −4.43** (negative at all 9 stages, −2.87 to −7.27), versus −0.0007
for predicting no anomaly. The model explains none of the season-to-season
variation once district climatology is removed.

Conclusion, reported as found: **the project's high raw-yield R2 is carried
largely by district identity, not by the environmental features.** The defensible
claim is district-level yield reconstruction, not in-season forecasting skill.

### Environment pinned

All results now come from ONE environment (Python 3.10.16, numpy 1.24.3,
scikit-learn 1.6.1, torch 2.6.0, xgboost 3.2.0, lightgbm 4.6.0) — the only one
available here with all five regressors plus torch. `requirements.txt` was
re-pinned to match, because scikit-learn pickles are not portable across feature
releases in either direction (verified empirically both ways).

### Tests

`scripts/smoke_test.py`: **31 PASS, 3 SKIP, 0 FAIL**. New checks: feature-policy
importable, duplicate-alias table still valid, no leaked features in any model
input, LSTM checkpoint contract (keys, architecture 64x2, feature count, best
epoch == log minimum), CNN weights intact, temporal-CV fold ordering has no
future leak.

## 27. Exact local run commands

```powershell
cd D:\BIG_Works_Done\Crop_yeild_predictions

# 1. environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. verify (read-only, trains nothing; expect 0 failures)
python scripts\smoke_test.py

# 3. inference with the saved models
python src\models\predict_stagewise.py

# 4. regenerate figures and the stage-wise report
python src\visualization\plot_results.py
```

Linux / macOS: replace step 1 with
`python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

**Do not run** `run_training.py`, `run_training_with_cnn.py`,
`run_baseline_ml.py`, `src/deep_learning/train_*.py` or `run_cnn_pipeline.py`
unless you intend to overwrite the authoritative results. Costs are listed in
`docs/reproducibility.md` section 4.
