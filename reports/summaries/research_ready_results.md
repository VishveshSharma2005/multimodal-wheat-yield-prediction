# Research results — authoritative summary

**Generated 2026-09-06.** Every number here was read from a file in this
repository. Result families that answer different questions are kept separate;
they are **not** combined into one leaderboard.

---

## Headline finding — read this first

> **Under forward-chaining temporal validation, none of the five regressors beats
> a naive district-historical-mean baseline, and the stage-wise model has
> negative within-district skill (anomaly R² = −4.43).**
>
> The high raw-yield R² reported for this project (0.92–0.97) is carried almost
> entirely by **district identity**, not by the weather / satellite / calendar
> features. Gujarat districts have persistent, systematically different wheat
> yields, and every district appears in both the training and the validation
> split, so a model that learns only "which district is this" scores very well on
> raw yield.
>
> This does not invalidate the pipeline, the data engineering, or the CNN
> experiment. It does mean the correct claim is about **district-level yield
> reconstruction**, not about in-season environmental forecasting skill.

---

## A. Primary yield model — stage-wise regression

Nine per-stage models. Five candidates fitted per stage; the lowest 2023-24
validation RMSE is kept. **GradientBoostingRegressor wins all 9 stages.**

2023-24 hold-out, corrected feature set (33 districts per stage):

| Stage | DAS | MAE | RMSE | MAPE % | R² |
|---|---|---|---|---|---|
| sowing | 10 | 159.74 | 220.50 | 5.59 | 0.9162 |
| early_vegetative | 20 | 154.99 | 195.50 | 5.07 | 0.9341 |
| tillering | 40 | 169.54 | 212.96 | 5.93 | 0.9219 |
| stem_elongation | 60 | 164.50 | 200.69 | 5.70 | 0.9306 |
| **booting_heading** | 80 | 161.15 | **187.98** | 5.49 | **0.9391** |
| flowering | 95 | 149.82 | 200.12 | 4.98 | 0.9310 |
| grain_filling_initial | 105 | 190.01 | **260.30** | 6.43 | 0.8833 |
| grain_filling | 115 | **128.29** | 178.13 | **4.07** | 0.9453 |
| maturity_preharvest | 130 | 181.49 | 230.91 | 6.03 | 0.9081 |

Mean best-per-stage RMSE **209.68**. Pooled all-stage model:
**GradientBoostingRegressor, MAE 104.47, RMSE 132.94, MAPE 3.38 %, R² 0.9695**
(891 train / 297 validation).

Source: `reports/model_results/stagewise_regression_metrics.csv`,
`combined_regression_metrics.csv`.

**Caveat that applies to all of the above:** the winning model is selected on the
same 2023-24 season it is then scored on. These numbers are selection-optimistic.
Section E is the honest generalisation estimate.

## B. Temporal sequence model — LSTM

`LSTM(input_size=76, hidden=64, layers=2, dropout=0.2) → Linear(64,1)` over the
9-stage sequence. 99 train / 33 validation sequences. Imputer, scaler, one-hot
encoder and target normalisation fitted on training rows only. Seed 42,
byte-reproducible.

**MAE 92.29, RMSE 113.38, MAPE 3.12 %, R² 0.9778**, restored from the best
epoch (28 of 43 run).

Source: `reports/model_results/lstm_metrics.csv`.

The LSTM is the strongest single model on the 2023-24 hold-out. Section E shows
this does not necessarily generalise — the LSTM is not included in the temporal
CV (see "Not covered" below).

## C. Yield classification — Q1–Q5

One classifier per stage, RandomForest vs GradientBoosting, thresholds from
`pd.qcut` on the **training split only** (verified leakage-free).

Across all 18 fitted (stage, classifier) combinations validation accuracy spans
**0.4242 – 0.8182**. Taking the selected classifier per stage, accuracy is
**0.6364 – 0.8182**; best `early_vegetative` (0.8182, macro-F1 0.8175), worst
`grain_filling_initial` (0.6364).
Confusion matrices in `reports/model_results/confusion_matrix_<stage>.csv`.

With 33 validation cases across 5 classes (~6–7 per class) these are indicative
only. Section E's baseline logic applies here too: a classifier that knows the
district can place most districts in the right quintile without any agronomy.

## D. Independent phenology model — CNN

**Not part of the yield pipeline.** Five ImageNet-pretrained backbones
(frozen base → `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(5, softmax)`)
classify wheat growth stage from field images.

Held-out **test** split (782 images, disjoint from train/validation):

| Model | Accuracy | F1 |
|---|---|---|
| **xception** | **0.9923** | **0.9923** |
| densenet121 | 0.9847 | 0.9846 |
| vgg16 | 0.9731 | 0.9731 |
| inceptionv3 | 0.9616 | 0.9621 |
| mobilenetv2 | 0.9015 | 0.8987 |

The CNN feature vectors are **not** fused into the yield models. The image-class
vocabulary (`1_Tillering`, `2_Jointing`, `3_BH`, `4_Flowering`, `5_Filling`) does
not match the agronomic stage vocabulary used by the yield table, so the merge
produced 6,913 all-NaN columns that were dropped before fitting. Even with the
key repaired, the design would broadcast one mean vector per stage to all
districts and years, carrying no district- or season-specific information.

**Report the CNN as a standalone wheat growth-stage classification result.**
Do not describe this project as CNN→yield multi-modal fusion.

## E. Temporal robustness — walk-forward validation and baselines

Strictly forward-chaining. For a held-out season the model sees only earlier
seasons. 2024-25 is never a fold (no published target).

| Fold | Train | Validate |
|---|---|---|
| 1 | 2020-21 | 2021-22 |
| 2 | 2020-21 … 2021-22 | 2022-23 |
| 3 | 2020-21 … 2022-23 | 2023-24 |

Mean ± sd over the three folds, pooled across stages:

| Kind | Name | MAE | RMSE | MAPE % | R² |
|---|---|---|---|---|---|
| **baseline** | **Baseline A: district historical mean** | **113.61 ± 38.92** | **155.25 ± 56.39** | **3.62** | **0.9567 ± 0.026** |
| baseline | Baseline B: persistence (last known year) | 111.96 ± 32.84 | 158.96 ± 46.66 | 3.63 | 0.9558 ± 0.022 |
| model | LightGBMRegressor | 124.51 ± 25.99 | 169.50 ± 33.02 | 4.00 | 0.9510 ± 0.018 |
| baseline | Baseline C: district identity only | 158.74 ± 33.79 | 189.66 ± 41.72 | 5.02 | 0.9385 ± 0.024 |
| model | RandomForestRegressor | 156.70 ± 69.17 | 216.70 ± 90.72 | 4.98 | 0.9149 ± 0.065 |
| model | GradientBoostingRegressor | 173.37 ± 70.89 | 222.11 ± 98.38 | 5.46 | 0.9095 ± 0.071 |
| model | XGBoostRegressor | 190.28 ± 117.69 | 267.97 ± 175.01 | 6.05 | 0.8522 ± 0.171 |
| model | HistGradientBoostingRegressor | 342.99 ± 237.30 | 503.19 ± 343.15 | 9.94 | 0.4641 ± 0.519 |

Three things follow:

1. **No model beats Baseline A.** The best model (LightGBM, RMSE 169.50) is worse
   than predicting each district's historical mean (155.25). In every individual
   fold a baseline is at or near the top.
2. **The 2023-24 winner is not the walk-forward winner.** GradientBoosting wins
   the single-season hold-out but ranks 3rd of 5 under temporal CV, behind
   LightGBM. This is direct evidence of model-selection optimism in Section A.
3. **The gap between Baseline C (district identity only, RMSE 189.66) and the
   full models (169.50–503.19)** is the only part attributable to environmental
   features — and it is small, inconsistent, and swamped by fold-to-fold spread.

Sources: `reports/model_results/temporal_cv_metrics.csv`,
`temporal_cv_fold_metrics.csv`, `baseline_comparison.csv`;
`reports/summaries/temporal_validation_summary.md`,
`baseline_comparison.md`; figure `reports/figures/temporal_cv_model_vs_baseline.png`.

**Statistical caution:** only 3 folds, and they are nested (later folds contain
earlier training data). A standard deviation over 3 values is a weak estimate of
spread. **No claim of statistical significance is made or supported.**

## F. Within-district (anomaly) skill

Removing each district's **training-season** mean from both observed and
predicted 2023-24 yield isolates the season-specific signal. The centring uses
training seasons only, so it is leakage-free.

| Space | Predictor | n | MAE | RMSE | R² |
|---|---|---|---|---|---|
| raw yield | stage-wise model | 297 | 162.17 | 210.99 | 0.9233 |
| raw yield | district historical mean | 297 | **70.29** | **90.54** | **0.9859** |
| anomaly | stage-wise model | 297 | 162.17 | 210.99 | **−4.4342** |
| anomaly | zero anomaly (= district mean) | 297 | 70.29 | 90.54 | −0.0007 |

Observed anomaly standard deviation: 90.51 kg/ha.

**Anomaly R² is −4.43.** Once district climatology is removed, the model explains
none of the remaining season-to-season variation — it is substantially worse than
predicting no anomaly at all. Negative at every one of the 9 stages
(−2.87 to −7.27).

This is a predictive-skill decomposition, **not** a causal claim. It does not say
weather and vegetation indices are irrelevant to wheat yield; it says this model
on this data does not extract usable season-level signal from them.

Sources: `reports/model_results/within_district_skill.csv`,
`reports/summaries/within_district_skill.md`.

## G. 2024-25 forecast — forecast only, no accuracy

297 stage-level forecasts (33 districts × 9 stages) + a 9-row Mehsana view + 33
case-level LSTM forecasts.

**District-level 2024-25 Gujarat wheat APY has not been published.** Therefore:

* `yield_kg_ha` is null and `evaluation_status = pending_actual_apy` for all 297 rows
* **no MAPE, RMSE, MAE or R² exists or may be computed for 2024-25**
* the Gujarat **state-level** Final Advance Estimate (3280.69 kg/ha) is a
  benchmark only and must never be assigned to a district

**288 of the 297 forecast rows have median-imputed NDVI / NDRE / EVI** — only
Mehsana has 2024-25 Sentinel-2 exports. Those 288 rows are driven almost entirely
by weather, location and calendar features.

Details: `reports/summaries/forecast_2024_25.md`,
`docs/2025_yield_reference_methodology.md`.

---

## Not covered by the robustness analysis

Stated so the gaps are not mistaken for results:

* **The LSTM and the quintile classifiers were not run through temporal CV.**
  Their reported numbers are single-season hold-out only, and the Section E
  caveat applies to them by analogy but has not been measured.
* **No significance testing.** 3 folds and 33 validation cases do not support it.
* **No leave-one-district-out evaluation.** Would test spatial rather than
  temporal generalisation.
* **CNN was not re-evaluated.** Its weights are locked and unchanged.

## What changed on 2026-09-05/06

| Fix | Effect |
|---|---|
| 2024-25 ground-truth contamination removed | An invalid "Mehsana 2025 MAPE 11.93" was withdrawn; 2024-25 is forecast-only |
| LSTM best-checkpoint bug (`copy.deepcopy`) + seeding | Checkpoint now holds the best epoch, run is byte-reproducible |
| `actual_available` removed from all feature sets | Removed a target-availability feature that was constant in train/validation and flipped at forecast time |
| 8 exact-duplicate feature aliases removed | De-duplicated feature space; feature importances no longer split across aliases |
| `"text"` substring bug fixed | `image_texture_proxy_stage_avg` was being silently dropped; it is now available to the models |
| Temporal CV, baselines, within-district skill added | Produced the headline finding above |

Pre-fix results are preserved as `reports/model_results/*_original.csv` and are
**superseded** — do not cite them.
