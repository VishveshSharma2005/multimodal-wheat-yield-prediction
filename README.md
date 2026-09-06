# Stage-wise Multi-Modal Wheat Yield Prediction

Stage-wise forecasting of final wheat (Rabi) yield for **33 districts of Gujarat,
India**, with **Mehsana** as the focus district, from weather, Sentinel-2
vegetation indices and crop-calendar features aggregated over **9 growth stages**.

> Every claim in this README was verified against the code and artefacts in this
> repository. Where the original design and the implementation diverge, the
> implementation is what is described.

> ### Headline finding — read before citing any metric
>
> **Under forward-chaining temporal validation, none of the five regressors beats
> a naive "predict each district's historical mean" baseline, and the stage-wise
> model has negative within-district skill (anomaly R² = −4.43).**
>
> Gujarat districts have persistent, systematically different wheat yields, and
> every district appears in both the training and the validation split. The high
> raw-yield R² (0.92–0.97) is therefore carried largely by **district identity**,
> not by the weather / satellite / calendar features.
>
> The correct claim for this work is **district-level yield reconstruction**, not
> in-season environmental forecasting skill. Full analysis:
> [`reports/summaries/research_ready_results.md`](reports/summaries/research_ready_results.md).


---

## Quick Start

```text
Environment  →  Data/features  →  Train models  →  Evaluate  →  Predict  →  Visualize  →  Reports
   conda        ships in repo      [optional]      temporal CV   inference    figures     read results
                                                   + baselines
```

```powershell
cd D:\BIG_Works_Done\Crop_yeild_predictions
conda create -n wheat python=3.10 -y ; conda activate wheat
pip install -r requirements.txt

python scripts\smoke_test.py                   # verify (expect 0 FAIL)
python src\models\predict_stagewise.py         # inference
python src\models\run_temporal_validation.py   # temporal CV + baselines
python src\models\run_within_district_skill.py # within-district skill
python src\visualization\plot_results.py       # figures
```

About one minute total. **Nothing above retrains a model.**
The modelling table (`data/processed/stage_features.csv`) ships with the repo, so
no data preparation is needed. Full command reference:
[`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md).

### Where to look

| You want | Go to |
|---|---|
| **The results, and what is safe to cite** | [`reports/summaries/research_ready_results.md`](reports/summaries/research_ready_results.md) |
| An index of every result file | [`reports/README.md`](reports/README.md) |
| How the pipeline works end to end | [`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md) |
| How to run each step | [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md) |
| Where every file lives | [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md) |
| What to publish on GitHub | [`docs/GITHUB_GUIDE.md`](docs/GITHUB_GUIDE.md) |
| Key figure (models vs baselines) | [`reports/figures/temporal_cv_model_vs_baseline.png`](reports/figures/temporal_cv_model_vs_baseline.png) |
| Why there is no 2024-25 accuracy | [`docs/2025_yield_reference_methodology.md`](docs/2025_yield_reference_methodology.md) |
| Every correction that was applied | [`CLEANUP_REPORT.md`](CLEANUP_REPORT.md) |

---

## Research Objective

At the end of each of the 9 wheat growth stages, predict the **final harvest
yield** (kg/ha) for a district-season, using only information available up to
that point in the season. The goal is a progressively refined in-season forecast
rather than a single end-of-season estimate.

The unit of analysis is a **case** = district x season. The target is repeated
across the 9 stages of a case; the models never predict a "per-stage yield".

## Problem Background

District yield statistics arrive months after harvest. An in-season forecast that
improves as the crop develops is useful for procurement, storage and advisory
planning. Wheat in Gujarat is a Rabi crop sown around late November and harvested
around late March (125-126 day duration), which makes a stage-indexed forecasting
frame natural.

## Dataset

| | |
|---|---|
| Cases | 165 (33 districts x 5 seasons) |
| Rows | 1,485 (165 cases x 9 stages) |
| Real feature columns | 33 (+ keys, target, split) |
| Crop / season | Wheat / Rabi |
| Modelling table | [`data/processed/stage_features.csv`](data/processed/stage_features.csv) |

Details: [`docs/dataset.md`](docs/dataset.md).

## Data Sources

| Modality | Source | Status |
|---|---|---|
| Yield (target) | Area-Production-Yield, Directorate of Agriculture, Government of Gujarat — <https://dag.gujarat.gov.in/Home/AreaProductionAndYield> | **used** |
| Weather | **Open-Meteo Archive** (ERA5) — <https://archive-api.open-meteo.com/v1/archive> | **used** |
| Satellite | Sentinel-2 `COPERNICUS/S2_SR_HARMONIZED` via Google Earth Engine | **used** |
| Field images (CNN) | Wheat growth-stage image set, 5,211 images, 5 classes | **used — separate experiment** |
| Auxiliary images | Mendeley *WheatPhenology: A Multi-Stage Field Image Dataset* | **used — handcrafted features only** |
| Soil (SoilGrids / ISRIC) | — | **not used** |
| Drone / farm logs | — | **not used** (empty templates only) |

Weather is Open-Meteo, **not** NASA POWER — correcting an earlier project
write-up. There are **no soil features** in the modelling table.

## Temporal Coverage

| Split | Agri years | Cases | Rows | Target |
|---|---|---|---|---|
| Training | 2020-21, 2021-22, 2022-23 | 99 | 891 | district APY yield |
| Validation | 2023-24 | 33 | 297 | district APY yield |
| Forecast | 2024-25 | 33 | 297 | **not published** — `pending_actual_apy` |

99 training and 33 validation observations per stage.

## District Coverage

All 33 Gujarat districts: Ahmedabad, Amreli, Anand, Arvalli, Banaskantha,
Bharuch, Bhavnagar, Botad, Chhota Udepur, Dahod, Dangs, Devbhoomi Dwarka,
Gandhinagar, Gir Somnath, Jamnagar, Junagadh, Kachchh, Kheda, Mahisagar,
**Mehsana**, Morbi, Narmada, Navsari, Panchmahal, Patan, Porbandar, Rajkot,
Sabarkantha, Surat, Surendranagar, Tapi, Vadodara, Valsad.

## Wheat Growth Stages

Days-after-sowing windows from [`configs/config.yaml`](configs/config.yaml):

| # | Stage | DAS | | # | Stage | DAS |
|---|---|---|---|---|---|---|
| 1 | `sowing` | 0-10 | | 6 | `flowering` | 80-95 |
| 2 | `early_vegetative` | 10-20 | | 7 | `grain_filling_initial` | 95-105 |
| 3 | `tillering` | 20-40 | | 8 | `grain_filling` | 105-115 |
| 4 | `stem_elongation` | 40-60 | | 9 | `maturity_preharvest` | 115-130 |
| 5 | `booting_heading` | 60-80 | | | | |

## Feature Modalities

| Modality | Features | Reaches the model? |
|---|---|---|
| Weather (rain, Tmin/Tmax, heat-stress days, GDD) | 12 | **Yes** |
| Satellite (NDVI / NDRE / EVI last, mean, max, slope) | 11-12 | **Yes** |
| Location + crop calendar (district, lat/lon, stage, DAS) | ~10 | **Yes** |
| Handcrafted image features | 6 | Yes, but they are **stage constants** — zero information inside a per-stage model |
| CNN deep features | 6,913 columns | **No — 100% NaN, dropped before fitting** |
| Soil | 0 | n/a |

## Architecture

```
APY yield ---+
Weather -----+---> stage-wise feature table (1485 x 55) ---+--> per-stage regression --> yield
Sentinel-2 --+          9 DAS windows per case             |    (9 GradientBoosting models)
Images ------+          leakage-guarded aggregates         +--> LSTM (9-step sequence) --> yield
                                                           +--> quintile classifier --> Q1..Q5

Field images --> 5 CNNs --> growth-stage class     [separate experiment, not fused]
```

Verified data flow, leakage controls and caveats: [`docs/pipeline.md`](docs/pipeline.md).

## CNN Experiment

Five ImageNet-pretrained backbones (frozen base -> `GlobalAveragePooling2D` ->
`Dropout(0.2)` -> `Dense(5, softmax)`) classify wheat growth stage from field
images. Trained on 3,647 images, validated on 782, evaluated on a **disjoint
held-out test set of 782**.

| Model | Test accuracy | Test F1 | Epochs run |
|---|---|---|---|
| **xception** | **0.9923** | **0.9923** | 29 |
| densenet121 | 0.9847 | 0.9846 | 20 |
| vgg16 | 0.9731 | 0.9731 | 21 |
| inceptionv3 | 0.9616 | 0.9621 | 21 |
| mobilenetv2 | 0.9015 | 0.8987 | 14 |

Configured for up to 100 epochs; early stopping (patience 5,
`restore_best_weights=True`) halted training at 14-29 epochs.

> **The CNN features are not fused into the yield model.**
> `build_stage_table.py` merges `cnn_stage_features.csv` on the `stage` column,
> but that table is keyed on the *image* class names (`1_Tillering`, `2_Jointing`,
> `3_BH`, `4_Flowering`, `5_Filling`) while the stage table uses the agronomic
> names (`sowing` ... `maturity_preharvest`). No key matches, so all 6,913
> `cnn_*` columns are NaN and are dropped by the ">90% missing" filter.
>
> Confirmed independently by `ml_selected_features.csv` (0 CNN features selected),
> `lstm_summary.md` (every `cnn_feat_*` "missing 100.00%"), the identical
> "+Handcrafted" and "+CNN" rows of `cnn_ablation_study.csv`, and
> `cnn_available_flag = 0` in every prediction row.
>
> **Describe the CNN as a standalone growth-stage classification experiment.**
> Do not claim CNN-to-regression multi-modal fusion. Details:
> [`docs/models.md`](docs/models.md) section 1.4.

## Multi-Modal Regression

One model per stage; five candidates fitted, lowest validation RMSE kept:
`RandomForestRegressor`, `GradientBoostingRegressor`,
`HistGradientBoostingRegressor`, `LGBMRegressor`, `XGBRegressor`.

**GradientBoostingRegressor won all 9 stages** on the 2023-24 hold-out. A pooled
all-stage model is also fitted (`best_combined_model.joblib`), where
GradientBoosting also wins.

*Caveat:* the winner is chosen on the validation season and then scored on that
same season, so these metrics are selection-optimistic. Under walk-forward
temporal CV (see [Validation Methodology](#validation-methodology)) the ranking
changes — **LightGBM** becomes the best model and GradientBoosting drops to 3rd.

## Validation Methodology

Three separate evaluations are reported, and they are **not** interchangeable:

| Evaluation | What it answers | Where |
|---|---|---|
| **2023-24 hold-out** | Fit quality on one held-out season | `stagewise_regression_metrics.csv` |
| **Walk-forward temporal CV** | Does it generalise across seasons? | `temporal_cv_metrics.csv` |
| **Within-district anomaly skill** | Any signal beyond district identity? | `within_district_skill.csv` |

### Walk-forward temporal cross-validation

Strictly forward-chaining — a held-out season is predicted using only earlier
seasons. 2024-25 is never a fold (no published target).

| Fold | Train | Validate |
|---|---|---|
| 1 | 2020-21 | 2021-22 |
| 2 | 2020-21 … 2021-22 | 2022-23 |
| 3 | 2020-21 … 2022-23 | 2023-24 |

Mean ± sd RMSE (kg/ha) over the 3 folds:

| Kind | Name | RMSE | R² |
|---|---|---|---|
| **baseline** | **A: district historical mean** | **155.25 ± 56.39** | **0.9567** |
| baseline | B: persistence (last known year) | 158.96 ± 46.66 | 0.9558 |
| model | LightGBMRegressor | 169.50 ± 33.02 | 0.9510 |
| baseline | C: district identity only | 189.66 ± 41.72 | 0.9385 |
| model | RandomForestRegressor | 216.70 ± 90.72 | 0.9149 |
| model | GradientBoostingRegressor | 222.11 ± 98.38 | 0.9095 |
| model | XGBoostRegressor | 267.97 ± 175.01 | 0.8522 |
| model | HistGradientBoostingRegressor | 503.19 ± 343.15 | 0.4641 |

**No model beats Baseline A.** Only 3 nested folds — no significance is claimed.

## Baselines

Evaluated on the same folds, each seeing only seasons before the held-out year:

* **A — district historical mean:** predict each district's mean over training seasons.
* **B — persistence:** predict the district's most recent known yield.
* **C — district identity only:** GradientBoosting on district + stage, no weather,
  satellite or image features.

The gap between Baseline C and the full models is the only part of performance
attributable to environmental features. It is small and inconsistent.

## Within-District Skill

Removing each district's **training-season** mean from both observed and predicted
2023-24 yield isolates the season-specific signal (leakage-free — the centring
never sees validation data):

| Space | Predictor | MAE | RMSE | R² |
|---|---|---|---|---|
| raw yield | stage-wise model | 162.17 | 210.99 | 0.9233 |
| raw yield | district historical mean | **70.29** | **90.54** | **0.9859** |
| anomaly | stage-wise model | 162.17 | 210.99 | **−4.4342** |
| anomaly | zero anomaly (= district mean) | 70.29 | 90.54 | −0.0007 |

**Anomaly R² = −4.43**, negative at all 9 stages. Once district climatology is
removed the model explains none of the remaining season-to-season variation.
This is a predictive-skill decomposition, not a causal claim.

## LSTM Temporal Modeling

PyTorch `LSTM(input_size=84, hidden_size=64, num_layers=2, dropout=0.2)` over the
9-stage sequence -> `Linear(64, 1)`. 99 train / 33 validation sequences. Imputer,
scaler, one-hot encoder and target normalisation all fitted on training rows only.
Seeded with 42 and verified reproducible.

Validation: **MAE 92.29, RMSE 113.38, MAPE 3.12%, R2 0.9778** — restored from
the best epoch (28 of 43 run). The LSTM was **not** run through temporal CV, so
the caveats above apply to it by analogy but have not been measured.

> **Superseded result: an earlier run reported RMSE 166.69 / MAE 120.59 /
> MAPE 4.03% / R2 0.9521. Do not cite it.** That checkpoint kept the *final*
> epoch instead of the best one, because `best_state = model.state_dict()` stored
> tensors by reference and later epochs mutated them. Fixed on 2026-09-05 with
> `copy.deepcopy(...)`; the reported RMSE now matches the best epoch (148.8119)
> and not the last (166.8315). See [`docs/models.md`](docs/models.md) section 3.1.

## Quintile Classification

Q1 (lowest) - Q5 (highest) yield class per stage. Thresholds from
`pd.qcut(train_y, 5)` on the **training split only** — verified, no leakage.
`RandomForestClassifier` vs `GradientBoostingClassifier`, best per stage kept.

Validation accuracy **0.6364 - 0.8182** for the selected classifier per stage
(0.4242 - 0.8182 across all 18 fitted combinations); best `early_vegetative`
(0.8182). With ~6-7 validation cases per class these are indicative only, and the
district-identity caveat above applies here too.

## Evaluation Metrics

MAE, MSE, RMSE, MAPE (zero-safe denominator), R2 for regression; accuracy, macro
F1 and 5x5 confusion matrices for classification. All reported on the **2023-24
validation season**, never on training predictions.

## Results

Stage-wise regression, **2023-24 hold-out**, GradientBoostingRegressor
(corrected feature set):

| Stage | DAS | MAE | RMSE | MAPE % | R2 |
|---|---|---|---|---|---|
| sowing | 10 | 159.74 | 220.50 | 5.59 | 0.9162 |
| early_vegetative | 20 | 154.99 | 195.50 | 5.07 | 0.9341 |
| tillering | 40 | 169.54 | 212.96 | 5.93 | 0.9219 |
| stem_elongation | 60 | 164.50 | 200.69 | 5.70 | 0.9306 |
| **booting_heading** | 80 | 161.15 | **187.98** | 5.49 | **0.9391** |
| flowering | 95 | 149.82 | 200.12 | **4.98** | 0.9310 |
| grain_filling_initial | 105 | 190.01 | 260.30 | 6.43 | 0.8833 |
| grain_filling | 115 | **128.29** | 178.13 | 4.07 | 0.9453 |
| maturity_preharvest | 130 | 181.49 | 230.91 | 6.03 | 0.9081 |

Pooled all-stage model: MAE 104.47, RMSE 132.94, MAPE 3.38%, R2 0.9695.
LSTM: MAE 92.29, RMSE 113.38, MAPE 3.12%, R2 0.9778.

Accuracy does **not** improve monotonically across the season — the best stage is
`booting_heading` (DAS 80), not `maturity_preharvest`.

> These are single-season hold-out numbers with the model selected on that same
> season. Read them together with the temporal CV and within-district results
> above, which show the model does not beat a district-mean baseline.

Full tables, the modality ablation and the authoritative file list:
[`docs/results.md`](docs/results.md) and
[`reports/summaries/final_model_summary.csv`](reports/summaries/final_model_summary.csv).

## 2024-25 Forecast

297 stage-level forecasts (33 districts x 9 stages), plus a 9-row Mehsana view
and 33 case-level LSTM forecasts. Mehsana ranges 3001.7 - 3330.2 kg/ha across the
9 stages, inside the 2020-21..2023-24 observed band of 3169.33 - 3389.73 kg/ha.

> Only Mehsana has 2024-25 Sentinel-2 exports. For the other 32 districts the
> 2024-25 vegetation indices are missing and median-imputed, so 288 of the 297
> forecast rows are driven mainly by weather and calendar features.

## 2024-25 Ground-Truth Limitation

**No official district-level 2024-25 Gujarat wheat APY exists in this
repository.** The only 2024-25 figure is a **state-level** Final Advance Estimate
of 3280.69 kg/ha. The 2024-25 outputs are therefore **forecasts with no
computable accuracy metric**.

All 297 forecast rows carry `evaluation_status = pending_actual_apy` and a null
target, and no 2024-25 MAPE / RMSE / MAE / R2 is computed anywhere.

> **Previously fixed contamination (2026-09-05).** A carry-forward fallback in
> `src/data/ingest_apy_2025_ground_truth.py` had written Mehsana's **2023-24**
> yield (2796.06 kg/ha) into the 9 Mehsana 2024-25 rows as a real actual,
> producing an invalid "Mehsana MAPE 11.93". The fallback has been removed, the
> target cleared, and the affected outputs regenerated by inference only — no
> model was retrained and every forecast value is bit-identical. **If you have an
> older copy of `prediction_2025_evaluation.md` or a slide quoting a 2025 MAPE,
> discard it.** Full record:
> [`docs/2025_yield_reference_methodology.md`](docs/2025_yield_reference_methodology.md).

## Installation

Requires **Python 3.10-3.12**. The saved `.joblib` pipelines need
**scikit-learn >= 1.8 and NumPy >= 2.0**; older versions cannot unpickle them.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Conda**

```bash
conda create -n wheat python=3.11 -y && conda activate wheat
pip install -r requirements.txt
```

TensorFlow is deliberately excluded from `requirements.txt`. Install
`requirements_colab_cnn.txt` only if you need the CNN branch (needs Keras 3 /
TensorFlow >= 2.16).

## Local Execution

```bash
# 1. read-only verification (seconds, trains nothing)
python scripts/smoke_test.py

# 2. inference with the saved per-stage models -> rewrites the 3 prediction CSVs
python src/models/predict_stagewise.py

# 3. regenerate figures + the stage-wise report from existing predictions
python src/visualization/plot_results.py
```

`scripts/smoke_test.py` is expected to report **no failures**. Its
`2024-25 ground-truth integrity` check is a permanent guard against the
carry-forward defect described above.

Expensive commands that overwrite authoritative results (`run_training.py`,
`run_baseline_ml.py`, `run_training_with_cnn.py`, any
`src/deep_learning/train_*.py`) are listed with their costs in
[`docs/reproducibility.md`](docs/reproducibility.md) section 4. **The five CNN
models are locked; nothing in the normal path retrains them.**

## Repository Structure

```
.
├── configs/config.yaml            stage windows, paths, weather API, label map
├── data/
│   ├── raw/                       source exports (mostly gitignored) - see data/README.md
│   ├── interim/                   cleaned tables
│   └── processed/                 stage_features.csv + prediction outputs
├── docs/                          pipeline, dataset, models, results, reproducibility,
│                                  2025_yield_reference_methodology
├── models/
│   ├── cnn/                       5 .keras classifiers (271 MB, gitignored)
│   ├── regression/                9 per-stage + 1 combined .joblib
│   ├── classification/            9 per-stage quintile .joblib
│   └── deep_learning/             lstm_yield_model.pt
├── reports/
│   ├── model_results/             all metrics CSVs and model reports
│   ├── figures/                   4 PNGs
│   ├── summaries/                 final_model_summary.csv
│   └── README.md                  status of every report file
├── r_and_d/                       research write-ups and stage definitions
├── scripts/smoke_test.py          read-only verification
├── src/
│   ├── data/                      ingestion + stage-table construction (20 scripts)
│   ├── features/                  crop stages, vegetation indices, weather, image features
│   ├── models/                    regression, quintile, LSTM, prediction, ablation
│   ├── deep_learning/             CNN training / evaluation / feature extraction
│   ├── visualization/             plot_results.py
│   └── utils/                     config, paths, logging, cleanup
├── run_training.py                full pipeline (EXPENSIVE, needs network)
├── run_training_with_cnn.py       CNN-merge pipeline (EXPENSIVE)
├── run_baseline_ml.py             retrain regression + quintile (EXPENSIVE)
├── requirements.txt
└── requirements_colab_cnn.txt
```

## Large Dataset Handling

~4.1 GB of raw and derived data stays out of Git: the 5,211-image CNN dataset
(2.9 GB), the Mendeley auxiliary images, the APY workbooks (19 MB), the
Sentinel-2 exports, the 618 MB `cnn_features.csv` and the two 11 MB
`stage_features_with_*` tables (6,913 of whose 6,969 columns are empty).

What ships instead: `data/processed/stage_features.csv` — 858 KB, the identical
55 real feature columns — plus every prediction and metrics file. Total tracked
size **~14 MB**.

Sources, expected paths and download instructions:
[`data/README.md`](data/README.md).

## Model Artifacts

| Artefact | Size | In Git |
|---|---|---|
| `models/regression/*.joblib` (10) | ~1.3 MB | yes |
| `models/classification/*.joblib` (9) | ~9.5 MB | yes |
| `models/deep_learning/lstm_yield_model.pt` | 294 KB | yes |
| `models/cnn/*.keras` (5) | **271 MB** | **no** |

The CNN weights are preserved on disk but excluded from Git. Distribute them via
Git LFS, a GitHub Release, or Zenodo (recommended — also gives a citable DOI) and
place them at `models/cnn/<name>_best.keras`. See
[`docs/models.md`](docs/models.md) section 5.

## Reproducibility

See [`docs/reproducibility.md`](docs/reproducibility.md) for the environment,
the lightweight-vs-expensive command split, expected smoke-test output, and the
full rebuild-from-raw sequence.

Determinism: regression, quintile classifiers, CNNs and (since 2026-09-05) the
LSTM all seed with 42. Two consecutive LSTM runs now produce byte-identical
artefacts.

## Limitations

1. **No CNN-to-regression fusion.** The merge key mismatch means the CNN
   contributes nothing to yield prediction. Even repaired, the current design
   would broadcast one mean vector per stage to all districts and years, carrying
   no district or year signal. Real fusion needs per-case imagery, which this
   project does not have.
2. **No model beats a naive baseline under temporal CV**, and within-district
   anomaly skill is negative (R² = −4.43). The headline R² reflects district
   identity more than environmental signal. This is the most important limitation.
3. **No 2024-25 ground truth.** District-level 2024-25 APY has not been
   published, so no 2024-25 accuracy metric is computable. The 2024-25 outputs
   are forecasts only, and 288 of 297 rows use imputed vegetation indices.
3. **Small validation set.** 33 cases per stage; 99 training cases per stage.
   Confidence intervals are wide and the per-stage differences in the results
   table are not individually significant.
4. **Model selection on the validation set** inflates the reported metrics.
   Nested or leave-one-year-out validation would be more honest.
5. **Handcrafted image features carry no information** inside a per-stage model —
   they are global per-stage constants from an external image set with no
   district or date linkage.
6. **Satellite coverage gap in the forecast season.** Only Mehsana has 2024-25
   Sentinel-2 data; 288 of 297 forecast rows have imputed vegetation indices.
7. **No soil features**, despite soil appearing in earlier project descriptions.
8. **`area_ha` is 100% empty**; `lat`/`lon` exist for only 14 of 33 districts.
10. **335 of 560 auxiliary images are missing** from this working copy, so
    re-running the handcrafted image feature extraction would not reproduce the
    features the saved models were trained on.
11. **District-level aggregation.** A district-mean yield hides within-district
    variation; the "field images" are external and not from these districts.

## Citation

```bibtex
@misc{stagewise_wheat_yield_gujarat,
  title  = {Stage-wise Multi-Modal Wheat Yield Prediction for Gujarat, India},
  note   = {Stage-wise regression, LSTM temporal modelling and CNN growth-stage
            classification over 33 Gujarat districts, 2020-21 to 2024-25},
  year   = {2026}
}
```

Please also cite the underlying data sources: the Directorate of Agriculture,
Government of Gujarat (Area-Production-Yield statistics); Open-Meteo (ERA5
archive); Copernicus Sentinel-2 via Google Earth Engine; and the Mendeley
*WheatPhenology* dataset.
