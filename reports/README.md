# Results index

Everything under `reports/`, classified so you can tell at a glance what to cite.

> **Start here:** [`summaries/research_ready_results.md`](summaries/research_ready_results.md)
>
> **Headline finding:** under forward-chaining temporal validation **no model beats
> a naive district-historical-mean baseline**, and within-district anomaly skill is
> **R² = −4.43**. The high raw-yield R² is carried largely by district identity.
> Read that before quoting any single-season metric.

**Legend** — `PAPER` safe to cite · `CONTEXT` supporting/diagnostic · `SUPERSEDED` do not cite

---

## 1. FINAL RESULTS

| File | Contains | Split | Type | Paper? |
|---|---|---|---|---|
| [`summaries/research_ready_results.md`](summaries/research_ready_results.md) | The one authoritative narrative: primary model, LSTM, quintile, CNN, temporal CV, baselines, forecast — kept as separate result families, not one leaderboard | all | narrative | **PAPER** |
| [`summaries/final_model_summary.csv`](summaries/final_model_summary.csv) | 38 rows: every model × task with split sizes, MAE/RMSE/MAPE/R², status and artefact path | all | consolidated table | **PAPER** |
| [`model_results/temporal_cv_metrics.csv`](model_results/temporal_cv_metrics.csv) | 5 regressors + 3 baselines, mean ± sd over 3 forward-chaining folds | walk-forward CV (2021-22, 2022-23, 2023-24) | **cross-validation** | **PAPER** |
| [`model_results/temporal_cv_fold_metrics.csv`](model_results/temporal_cv_fold_metrics.csv) | Per-fold detail, 24 rows, with the train years used | walk-forward CV | cross-validation | **PAPER** |
| [`model_results/baseline_comparison.csv`](model_results/baseline_comparison.csv) | Same aggregate + RMSE relative to the best model | walk-forward CV | cross-validation | **PAPER** |
| [`model_results/within_district_skill.csv`](model_results/within_district_skill.csv) | Anomaly-space skill after removing training-season district means | 2023-24 validation | validation | **PAPER** |
| [`model_results/feature_audit.csv`](model_results/feature_audit.csv) | Per-column decision: retained / excluded and why (55 rows) | n/a | audit | **PAPER** (appendix) |
| [`model_results/stagewise_regression_metrics.csv`](model_results/stagewise_regression_metrics.csv) | 45 rows = 9 stages × 5 regressors | 2023-24 hold-out | validation | **PAPER** — with the selection-optimism caveat |
| [`model_results/combined_regression_metrics.csv`](model_results/combined_regression_metrics.csv) | 5 rows, pooled all-stage model | 2023-24 hold-out | validation | **PAPER** — same caveat |
| [`model_results/lstm_metrics.csv`](model_results/lstm_metrics.csv) | MAE 92.29 · RMSE 113.38 · MAPE 3.12 % · R² 0.9778, best epoch 28/43, seed 42 | 2023-24 hold-out | validation | **PAPER** — not covered by temporal CV |
| [`model_results/quintile_classification_metrics.csv`](model_results/quintile_classification_metrics.csv) | 18 rows = 9 stages × 2 classifiers | 2023-24 hold-out | validation | **PAPER** |
| [`model_results/cnn_evaluation_metrics.csv`](model_results/cnn_evaluation_metrics.csv) | 5 CNNs, Xception best at 99.23 % | **held-out test** (782 images) | test | **PAPER** — as an *independent phenology* result |

### Supporting narratives (all PAPER-safe)

| File | Contains |
|---|---|
| [`summaries/temporal_validation_summary.md`](summaries/temporal_validation_summary.md) | Walk-forward protocol, fold table, per-fold results, statistical caution |
| [`summaries/baseline_comparison.md`](summaries/baseline_comparison.md) | Baseline definitions (A district mean, B persistence, C district-identity-only) and interpretation |
| [`summaries/within_district_skill.md`](summaries/within_district_skill.md) | What anomaly skill measures, why the centring is leakage-free, limits |
| [`summaries/forecast_2024_25.md`](summaries/forecast_2024_25.md) | Forecast-only status, the 288/297 imputation limitation, exact wording to use |

---

## 2. PREDICTIONS

All under `data/processed/`, not `reports/`.

| File | Rows | Split | Type | Paper? |
|---|---|---|---|---|
| `data/processed/validation_stagewise_predictions.csv` | 297 | 2023-24 | **validation** — has actuals and errors | **PAPER**. These are the *raw district-level* predictions; the figures show a stage-wise aggregate of them |
| `data/processed/prediction_2025_stagewise_predictions.csv` | 297 | 2024-25 | **forecast** — no actuals, no errors | **PAPER** as forecast only |
| `data/processed/mehsana_2025_stagewise_predictions.csv` | 9 | 2024-25 Mehsana | **forecast** | **PAPER** as forecast only |
| `data/processed/lstm_predictions_2025.csv` | 33 | 2024-25 | **forecast** | **PAPER** as forecast only |

> Every 2024-25 row has a null target and `evaluation_status = pending_actual_apy`.
> **No 2024-25 accuracy metric exists or may be computed.** 288 of the 297 rows
> also use median-imputed vegetation indices.

---

## 3. FIGURES

| File | Shows | Paper? |
|---|---|---|
| [`figures/temporal_cv_model_vs_baseline.png`](figures/temporal_cv_model_vs_baseline.png) | RMSE ± sd, models (blue) vs naive baselines (red), walk-forward CV | **PAPER — the key figure.** Carries the headline finding |
| [`figures/stagewise_rmse_trend.png`](figures/stagewise_rmse_trend.png) | Validation RMSE vs stage-end DAS (297 predictions grouped by stage) | **PAPER** |
| [`figures/stagewise_mae_trend.png`](figures/stagewise_mae_trend.png) | Same aggregation for MAE | **PAPER** |
| [`figures/mehsana_2025_stagewise_prediction.png`](figures/mehsana_2025_stagewise_prediction.png) | Mehsana 2024-25 forecast trajectory — forecast line only, no "actual" line exists | **PAPER** as forecast only |
| [`figures/lstm_training_loss.png`](figures/lstm_training_loss.png) | LSTM train/validation loss over 43 epochs | CONTEXT (appendix) |

---

## 4. CONTEXT — supporting model outputs

| File | Contains |
|---|---|
| `model_results/stagewise_regression_summary.md` | Best model per stage + feature groups used ("cnn" is absent from every stage) |
| `model_results/combined_regression_summary.md` | Pooled model winner and split sizes |
| `model_results/quintile_classification_summary.md` | Best classifier per stage; confirms thresholds are training-only |
| `model_results/confusion_matrix_<stage>.csv` (9) | 5×5 Q1–Q5 matrices, 33 validation cases each |
| `model_results/feature_importance_<stage>.csv` (9) + `_combined.csv` | Gini importances of the saved models |
| `model_results/stagewise_improvement_report.md` | Validation metrics ordered by DAS; notes the trend does not improve monotonically |
| `model_results/baseline_ml_report.md` | Markdown view of the stagewise + combined + quintile CSVs, regenerated from current metrics. *Name is historical — "baseline ML models", unrelated to the naive baselines in `baseline_comparison.csv`* |
| `model_results/lstm_training_log.csv` | Per-epoch loss, 43 epochs. Best epoch 28 (val_rmse 113.38) **is** the epoch saved |
| `model_results/prediction_2025_evaluation.md` | Confirms 0 rows with actual APY, all metrics N/A |
| `model_results/lstm_summary.md`, `lstm_selected_features.csv`, `ml_selected_features.csv` | Large per-feature audit trails (gitignored) |

## 5. CONTEXT — CNN (independent phenology experiment)

| File | Contains |
|---|---|
| `model_results/cnn_model_comparison.csv` | Test metrics + training time per backbone |
| `model_results/<model>_meta.json` (5) | Config + best validation accuracy. `"epochs": 100` is the configured *maximum*; early stopping ran 14–29 |
| `model_results/<model>_history.csv` (5) | Per-epoch history; row count = epochs actually run |
| `model_results/<model>_confusion_matrix.csv` (5) | 5×5 stage-classification matrices |
| `model_results/<model>_training_summary.json` (5) | Contains `/home/student/...` paths — training-machine provenance |
| `model_results/cnn_ablation_study.csv` | Modality ablation. The "+Handcrafted" and "+CNN" rows are **identical** (both 24 features) — CNN adds zero features |
| `model_results/cnn_improvement_report.md` | States "Feature count increase: 0, RMSE improvement: 0.0" |
| `model_results/cnn_pipeline_validation_report.md` | CNN pipeline wiring check |
| `model_results/corrupted_images.csv` | Unreadable images excluded from the CNN split |

> **The CNN is not fused into the yield models.** Cite it as a standalone wheat
> growth-stage classification result. See `docs/models.md` §1.4.

## 6. CONTEXT — data pipeline diagnostics

`apy_ingestion_report.md` · `coordinate_registry_report.md` · `data_sources_report.md` ·
`weather_coverage_report.md` · `satellite_coverage_summary.md` ·
`satellite_file_check_report.md` · `satellite_ingestion_report.md` ·
`prediction_2025_satellite_audit.md` (confirms 2024-25 Sentinel-2 is Mehsana-only) ·
`gee_export_tasks.md` · `training_data_verification.md` ·
`feature_diagnosis_report.md` (507 KB, gitignored)

These document data provenance and coverage. They are not results.

---

## 7. SUPERSEDED — do not cite

| Location | What | Tracked? |
|---|---|---|
| [`model_results/superseded/`](model_results/superseded/) | 7 pre-leakage-fix metric files, kept **in the repo** as auditable provenance for the corrections. See its README | yes |
| `archive/superseded_reports/` | `final_model_report.md`, `final_research_summary.md`, `final_evaluation_2025.md`, `final_colab_readiness_report.md`, `apy_ingestion_summary.md` — earlier narratives whose metrics no longer match any current model | no |
| `archive/models_pre_leakage_fix/` | The 20 model artefacts from the pre-fix run | no |

---

## 8. Which split is which

| Term | Seasons | Has actuals? | Use for |
|---|---|---|---|
| **train** | 2020-21, 2021-22, 2022-23 (891 rows) | yes | fitting only |
| **validation / hold-out** | 2023-24 (297 rows) | yes | the single-season metrics |
| **walk-forward CV** | 2021-22, 2022-23, 2023-24 as 3 folds | yes | generalisation estimate |
| **forecast** | 2024-25 (297 rows) | **no** | forecasts only — never an accuracy metric |

Nothing in this repository reports `model.predict(X_train)` as validation performance.
