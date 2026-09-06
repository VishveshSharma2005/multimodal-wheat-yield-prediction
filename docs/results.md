# Results

Every number here was read from a file in this repository. Nothing was
recomputed, improved, or estimated.

**Authoritative narrative: [`reports/summaries/research_ready_results.md`](../reports/summaries/research_ready_results.md)**
**Consolidated table: [`reports/summaries/final_model_summary.csv`](../reports/summaries/final_model_summary.csv)**

> **Read the headline finding first.** Under forward-chaining temporal validation
> no model beats a district-historical-mean baseline, and within-district anomaly
> R² is −4.43. The single-season numbers below are selection-optimistic and are
> carried largely by district identity. See
> [`temporal_validation_summary.md`](../reports/summaries/temporal_validation_summary.md),
> [`baseline_comparison.md`](../reports/summaries/baseline_comparison.md) and
> [`within_district_skill.md`](../reports/summaries/within_district_skill.md).

---

## 1. Authoritative result files

| File | Contents | Produced by |
|---|---|---|
| `reports/model_results/stagewise_regression_metrics.csv` | 45 rows = 9 stages x 5 regressors, validation metrics | `train_stagewise_regression.py` |
| `reports/model_results/combined_regression_metrics.csv` | 5 rows, pooled all-stage model | `train_stagewise_regression.py::_train_combined` |
| `reports/model_results/stagewise_regression_summary.md` | Best model + feature groups per stage | `train_stagewise_regression.py` |
| `reports/model_results/quintile_classification_metrics.csv` | 18 rows = 9 stages x 2 classifiers | `train_quintile_classifier.py` |
| `reports/model_results/confusion_matrix_<stage>.csv` (9) | Q1-Q5 confusion matrices | `train_quintile_classifier.py` |
| `reports/model_results/lstm_metrics.csv` | LSTM validation metrics | `train_lstm_sequence_model.py` |
| `reports/model_results/lstm_training_log.csv` | Per-epoch train/val loss (55 epochs) | `train_lstm_sequence_model.py` |
| `reports/model_results/cnn_evaluation_metrics.csv` | CNN held-out **test** metrics | `evaluate_cnn_models.py` |
| `reports/model_results/cnn_model_comparison.csv` | CNN test metrics + training time | `compare_cnn_models.py` |
| `reports/model_results/<model>_history.csv` / `_meta.json` (5 each) | CNN per-epoch history and config | `train_<model>.py` |
| `reports/model_results/cnn_ablation_study.csv` | Modality ablation | `run_cnn_ablation_study.py` |
| `reports/model_results/feature_importance_*.csv` (10) | Gini importances per stage | `train_stagewise_regression.py` |
| `reports/model_results/stagewise_improvement_report.md` | Validation metrics ordered by DAS | `plot_results.py` |
| `data/processed/validation_stagewise_predictions.csv` | **297 raw district-level 2023-24 predictions** | `predict_stagewise.py` |
| `data/processed/prediction_2025_stagewise_predictions.csv` | 297 raw 2024-25 **forecasts** | `predict_stagewise.py` |
| `data/processed/mehsana_2025_stagewise_predictions.csv` | 9 Mehsana 2024-25 **forecasts** | `predict_stagewise.py` |
| `data/processed/lstm_predictions_2025.csv` | 33 case-level LSTM 2024-25 **forecasts** | `train_lstm_sequence_model.py` |
| `reports/figures/*.png` (4) | RMSE / MAE trends, Mehsana forecast, LSTM loss | `plot_results.py`, `train_lstm_sequence_model.py` |

Full status of every file under `reports/`, including which are superseded, is in
[`reports/README.md`](../reports/README.md).

---

## 2. Stage-wise regression — validation season 2023-24

33 districts per stage. **GradientBoostingRegressor won all 9 stages** on RMSE.

| Stage | DAS | MAE | RMSE | MAPE % | R2 |
|---|---|---|---|---|---|
| sowing | 10 | 159.17 | 222.40 | 5.60 | 0.9148 |
| early_vegetative | 20 | 158.54 | 198.87 | 5.29 | 0.9319 |
| tillering | 40 | 163.19 | 204.81 | 5.76 | 0.9277 |
| stem_elongation | 60 | 169.40 | 203.02 | 5.87 | 0.9290 |
| **booting_heading** | 80 | 156.69 | **183.11** | 5.36 | **0.9422** |
| flowering | 95 | 177.43 | 226.60 | 6.03 | 0.9115 |
| grain_filling_initial | 105 | 180.74 | **254.42** | 6.12 | 0.8885 |
| grain_filling | 115 | **133.25** | 186.26 | **4.29** | 0.9402 |
| maturity_preharvest | 130 | 180.91 | 230.85 | 6.01 | 0.9082 |

Best stage by RMSE: `booting_heading` (183.11). Worst: `grain_filling_initial`
(254.42). There is **no monotone improvement** as the season progresses —
`plot_results.py` labels the trend "worsens" from stage 1 to stage 9. This is
consistent with the feature audit: the satellite indices that would sharpen late
stages are the same aggregates recomputed over a longer window, and the image
features are stage-constants.

### 2.1 Model comparison (mean validation RMSE over the 9 stages)

| Model | Mean RMSE | Stages won |
|---|---|---|
| **GradientBoostingRegressor** | **212.3** | **9 / 9** |
| XGBoostRegressor | 275.6 | 0 |
| RandomForestRegressor | 272.4 | 0 |
| LightGBMRegressor | 593.2 | 0 |
| HistGradientBoostingRegressor | 623.8 | 0 |

HistGradientBoosting and LightGBM perform poorly here because both need more
data per leaf than 99 training rows supply; their defaults effectively predict
near the global mean (R2 0.22-0.50).

### 2.2 Combined (pooled all-stage) model — 891 train / 297 validation

| Model | MAE | RMSE | MAPE % | R2 |
|---|---|---|---|---|
| **GradientBoostingRegressor** | 101.69 | **129.39** | 3.29 | **0.9712** |
| LightGBMRegressor | **96.26** | 133.52 | **3.17** | 0.9693 |
| RandomForestRegressor | 100.58 | 143.95 | 3.30 | 0.9643 |
| XGBoostRegressor | 100.60 | 141.10 | 3.36 | 0.9657 |
| HistGradientBoostingRegressor | 103.43 | 142.48 | 3.43 | 0.9650 |

The pooled model looks better than the per-stage models mainly because it sees
9x more rows and can use `stage` as a feature. It is **not** a like-for-like
comparison with the stage-wise numbers.

---

## 3. Temporal LSTM — validation season 2023-24

99 train / 33 validation sequences, 9 timesteps, 84 features. Seed 42.

| MAE | MSE | RMSE | MAPE % | R2 |
|---|---|---|---|---|
| 119.63 | 22,144.97 | **148.81** | 3.98 | 0.9618 |

Restored from the **best** epoch (40 of 55 run), verified: the reported RMSE
equals epoch 40's `val_rmse` (148.8119) and not the last epoch's (166.8315).

### 3.1 Superseded LSTM result

| Run | Checkpoint kept | MAE | RMSE | MAPE % | R2 |
|---|---|---|---|---|---|
| Pre-fix (unseeded) | final epoch 34 — **bug** | 120.59 | **166.69** | 4.03 | 0.9521 |
| **Corrected (seed 42)** | **best epoch 40** | **119.63** | **148.81** | **3.98** | **0.9618** |

The 166.69 figure came from a checkpoint that silently kept the *last* epoch
instead of the best one (`state_dict()` stored by reference rather than
deep-copied). **Do not cite 166.69.** Because the corrected run is also freshly
seeded, the two rows come from different training trajectories — this is a
before/after of the pipeline, not two evaluations of the same weights. Details:
`docs/models.md` section 3.1.

---

## 4. Quintile classification (Q1-Q5) — validation season 2023-24

33 validation cases per stage, spread over 5 classes.

| Stage | Best model | Accuracy | Macro F1 |
|---|---|---|---|
| sowing | GradientBoosting | 0.6970 | 0.7008 |
| **early_vegetative** | GradientBoosting | **0.8485** | **0.8461** |
| tillering | GradientBoosting | 0.7879 | 0.7836 |
| stem_elongation | RandomForest | 0.7273 | 0.7231 |
| booting_heading | GradientBoosting | 0.6970 | 0.6564 |
| flowering | RandomForest | 0.7576 | 0.7289 |
| grain_filling_initial | RandomForest | 0.6364 | 0.6143 |
| grain_filling | GradientBoosting | 0.7273 | 0.7224 |
| maturity_preharvest | RandomForest | 0.6970 | 0.6753 |

Worst single result in the grid: RandomForest at `sowing`, accuracy 0.4545.

With ~6-7 validation cases per class, these are indicative only.

---

## 5. CNN wheat growth-stage classification — held-out test set (782 images)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **xception** | **0.9923** | 0.9924 | 0.9923 | **0.9923** |
| densenet121 | 0.9847 | 0.9848 | 0.9847 | 0.9846 |
| vgg16 | 0.9731 | 0.9731 | 0.9731 | 0.9731 |
| inceptionv3 | 0.9616 | 0.9659 | 0.9616 | 0.9621 |
| mobilenetv2 | 0.9015 | 0.9245 | 0.9015 | 0.8987 |

**This is a separate image-classification experiment.** It is not part of the
yield model. See `docs/models.md` section 1.4.

---

## 6. Modality ablation

`reports/model_results/cnn_ablation_study.csv` — single pooled
`HistGradientBoostingRegressor`, 891 train / 297 validation.

| Feature set | Features | MAE | RMSE | MAPE % | R2 |
|---|---|---|---|---|---|
| Weather only | 12 | 623.11 | 780.28 | 19.06 | -0.049 |
| Weather + Satellite | 23 | **481.79** | **651.52** | **13.97** | **0.269** |
| + Handcrafted images | 31 | 483.80 | 653.33 | 14.04 | 0.265 |
| + CNN features | **31** | 483.80 | 653.33 | 14.04 | 0.265 |

Read this carefully:

* Satellite indices are the single largest gain (RMSE 780 -> 652).
* Handcrafted image features make it marginally **worse** — they are stage
  constants (`docs/dataset.md` 5.3).
* The last two rows are **identical**, including the feature count. Adding CNN
  features added **zero** features because they are all NaN.
* These absolute numbers are much worse than section 2 because this study uses a
  single pooled HistGradientBoosting model, which was the weakest candidate in
  the per-stage comparison. Use it for *relative* modality comparison only.

---

## 7. 2024-25 outputs — forecasts, not evaluations

| Output | Rows |
|---|---|
| `prediction_2025_stagewise_predictions.csv` | 297 (33 districts x 9 stages) |
| `mehsana_2025_stagewise_predictions.csv` | 9 (Mehsana, 1 per stage) |
| `lstm_predictions_2025.csv` | 33 (1 per district) |

Mehsana 2024-25 stage-wise forecast (GradientBoostingRegressor):

| Stage | DAS | Forecast yield (kg/ha) |
|---|---|---|
| sowing | 10 | 3186.4 |
| early_vegetative | 20 | 3080.4 |
| tillering | 40 | 3001.7 |
| stem_elongation | 60 | 3019.7 |
| booting_heading | 80 | 3067.0 |
| flowering | 95 | 3095.4 |
| grain_filling_initial | 105 | 3330.2 |
| grain_filling | 115 | 3140.2 |
| maturity_preharvest | 130 | 3246.6 |

For context (not as an error metric): Mehsana observed yields were 3389.73
(2020-21), 3292.80 (2021-22), 3169.33 (2022-23), 3255.21 (2023-24) kg/ha, and the
Gujarat **state** 2024-25 Final Advance Estimate is 3280.69 kg/ha. The forecasts
sit inside the historical band.

**No accuracy metric exists for 2024-25.** District-level 2024-25 Gujarat wheat
APY has not been published, so `reports/model_results/prediction_2025_evaluation.md`
reports MAPE / RMSE / MAE / R2 as `N/A` with `pending_actual_apy` for all 297
rows. An earlier version printed `MAPE = 11.93`, computed against a
carried-forward 2023-24 value; that fallback was removed on 2026-09-05 and the
file regenerated by inference only. Full record:
[`docs/2025_yield_reference_methodology.md`](2025_yield_reference_methodology.md).
Also note that 288 of the 297 forecast rows have no 2024-25 satellite data
(Mehsana is the only district with 2024-25 Sentinel-2 exports), so only the 9
Mehsana rows are genuinely multi-modal forecasts.

---

## 8. Validation predictions: raw vs aggregated

`data/processed/validation_stagewise_predictions.csv` holds all **297 raw
district-level predictions** and is the file to cite. The figures
`reports/figures/stagewise_rmse_trend.png` and `stagewise_mae_trend.png` plot a
**stage-wise aggregate**: `plot_results.py` groups the 297 rows by stage
(33 districts each) and computes one MAE/RMSE/MAPE/R2 per stage. That aggregation
is documented here so the two are not confused. The raw predictions are never
replaced by the aggregates.

---

## 9. Headline claim that is safe to make

> On the held-out 2023-24 season (33 Gujarat districts, 297 stage-level
> observations), stage-wise GradientBoosting regressors predict final wheat yield
> with MAPE between 4.29% and 6.12% (RMSE 183-254 kg/ha, R2 0.89-0.94), using
> weather and Sentinel-2 vegetation indices aggregated up to the end of each
> growth stage. A separate Xception classifier identifies wheat growth stage from
> field images with 99.23% accuracy on a held-out test set; its features are not
> currently fused into the yield model.
