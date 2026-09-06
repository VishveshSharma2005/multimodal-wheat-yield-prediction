# Project map

Where everything lives and why.

```text
Crop_yeild_predictions/
├── README.md                     project overview + quick start
├── CLEANUP_REPORT.md             full audit trail of every correction applied
├── requirements.txt              pinned core dependencies (sklearn-version locked)
├── requirements_colab_cnn.txt    TensorFlow, only for the CNN experiment
├── .gitignore / .gitattributes   exclusions + line-ending normalisation
│
├── configs/
│   └── config.yaml               9 stage DAS windows, paths, weather API, label map
│
├── scripts/
│   └── smoke_test.py             read-only integrity check (31 checks, 0 FAIL)
│
├── src/
│   ├── data/                     (20) ingestion + stage-table construction
│   ├── features/                 (7)  crop stages, weather, vegetation indices,
│   │                                  image features, FEATURE POLICY
│   ├── models/                   (8)  regression, quintile, LSTM, prediction,
│   │                                  temporal CV, within-district skill, ablation
│   ├── deep_learning/            (11) CNN training/eval/features  [SEPARATE BRANCH]
│   ├── visualization/            (1)  plot_results.py
│   └── utils/                    (4)  config, paths, logging, cleanup
│
├── data/
│   ├── raw/                      source exports — mostly gitignored
│   ├── interim/                  cleaned tables — partly tracked
│   └── processed/                modelling table + predictions — partly tracked
│
├── models/
│   ├── regression/               (10) per-stage + combined .joblib      [tracked]
│   ├── classification/           (9)  quintile .joblib                  [tracked]
│   ├── deep_learning/            (1)  lstm_yield_model.pt               [tracked]
│   └── cnn/                      (5)  .keras, 271 MB              [NOT tracked]
│
├── reports/
│   ├── README.md                 THE RESULTS INDEX — what to cite
│   ├── summaries/                (6) authoritative narratives + consolidated CSV
│   ├── model_results/            (63) metrics, importances, confusion matrices
│   │   └── superseded/           (8) pre-leakage-fix results, provenance only
│   ├── figures/                  (5) final figures
│   └── *.md                      data-pipeline diagnostics
│
├── docs/
│   ├── PIPELINE_GUIDE.md         end-to-end non-CNN pipeline
│   ├── RUN_GUIDE.md              exact PowerShell commands
│   ├── PROJECT_MAP.md            this file
│   ├── GITHUB_GUIDE.md           what to commit / exclude
│   ├── dataset.md                years, districts, stages, modalities, missingness
│   ├── models.md                 CNN, regression, LSTM, quintile in detail
│   ├── results.md                results narrative + authoritative file list
│   ├── pipeline.md               verified data flow + leakage controls + caveats
│   ├── reproducibility.md        environment, seeds, cheap vs expensive commands
│   └── 2025_yield_reference_methodology.md   the 2024-25 ground-truth record
│
├── r_and_d/                      research write-ups, stage definitions, .docx
├── run_baseline_ml.py            retrain regression + quintile  [EXPENSIVE]
├── run_training.py               full rebuild + retrain         [EXPENSIVE]
├── run_training_with_cnn.py      CNN-merge variant              [EXPENSIVE]
└── archive/                      local-only: superseded reports, scratch notebooks,
                                  pre-fix model artefacts        [NOT tracked]
```

---

## Directories in detail

### `configs/`
`config.yaml` defines the 9 growth stages as days-after-sowing windows, all
project paths, the Open-Meteo weather settings, and the image-class → stage label
map. Paths are resolved project-relative by `src/utils/paths.py`, so there are no
machine-specific absolute paths in the active pipeline.

### `scripts/`
`smoke_test.py` — the safety net. Verifies the environment, config, that all 54
source files compile, dataset schema (1485 rows / 33 districts / 9 stages /
monotonic DAS), **that the 2024-25 target is empty across 8 artefacts**, **that no
target-derived or duplicate feature reaches any model**, the LSTM checkpoint
contract, that the 5 CNN weight files are intact, and that no temporal-CV fold
sees the future. Trains nothing.

### `src/data/` — ingestion and stage-table construction
The important ones:

| File | Role |
|---|---|
| `extract_apy_all_districts_xlsx.py` | APY workbooks → the authoritative district yield table |
| `download_weather.py` | Open-Meteo / ERA5 daily weather |
| `ingest_satellite.py` | Sentinel-2 GEE exports → observations table |
| **`build_stage_table.py`** | **The core**: 9 DAS windows per case, all leakage-guarded aggregates |
| `ingest_apy_2025_ground_truth.py` | 2024-25 target — correctly yields nothing; the carry-forward fallback was removed |
| `verify_training_data.py`, `diagnose_features.py` | Data-quality reports |

### `src/features/`
`crop_stages.py` (DAS windows) · `weather_features.py` (GDD, heat stress) ·
`vegetation_indices.py` (NDVI/NDRE/EVI slopes) · `extract_image_features.py` ·
`image_stage_features.py` · **`feature_policy.py`**.

**`feature_policy.py` is the single source of truth for which columns may be
model inputs.** All four trainers import it, so the exclusion rules cannot drift
apart (they previously did). It blocks identifiers, split keys, free text,
target-derived columns (`actual_available`, `evaluation_status`, …) and 8 verified
exact-duplicate aliases, while explicitly protecting the legitimate stage/DAS
indices.

### `src/models/`

| File | Role |
|---|---|
| `train_stagewise_regression.py` | 9 per-stage models + a pooled model, 5 candidates each |
| `train_quintile_classifier.py` | Q1–Q5 per stage, training-only thresholds |
| `train_lstm_sequence_model.py` | 9-step sequence model, seeded, best-checkpoint restore |
| `predict_stagewise.py` | Inference only — validation + 2024-25 forecasts |
| **`run_temporal_validation.py`** | **Walk-forward CV + the 3 naive baselines** |
| **`run_within_district_skill.py`** | **Anomaly-space skill after removing district climatology** |
| `run_cnn_ablation_study.py`, `generate_cnn_improvement_report.py` | Modality ablation |

### `src/deep_learning/` — separate branch
CNN training, evaluation and feature extraction. **The CNN is a wheat
growth-stage image classifier and its features do not reach the yield models.**
The five `.keras` files are locked, never retrained.

### `data/`

| Path | Contents | Tracked |
|---|---|---|
| `raw/apy/` | APY workbooks (19 MB) + small provenance CSVs | only the CSVs + README |
| `raw/satellite/` | 153 Sentinel-2 GEE exports | no |
| `raw/images/` | 5,211 CNN images + 560 auxiliary images (2.9 GB) | no |
| `interim/` | cleaned tables — `apy_yield_all_districts.csv` is the authoritative target | small ones yes |
| **`processed/stage_features.csv`** | **The modelling table, 1485 × 55** | **yes (858 KB)** |
| `processed/*_predictions*.csv` | validation + forecast outputs | yes |
| `processed/stage_features_with_{cnn,split}.csv` | same real columns + 6,913 all-NaN `cnn_*` (11 MB each) | no |
| `processed/cnn_features*` | 1.2 GB of deep features | no |

Full provenance and download instructions: [`../data/README.md`](../data/README.md).

### `models/`

| Path | Size | Tracked | Note |
|---|---|---|---|
| `regression/` (10) | ~1.3 MB | yes | GradientBoosting won all 9 stages |
| `classification/` (9) | ~9.5 MB | yes | quintile classifiers |
| `deep_learning/` (1) | ~295 KB | yes | LSTM, epoch 28/43, seed 42 |
| `cnn/` (5) | **271 MB** | **no** | locked, byte-identical, distribute via Zenodo/Release |

### `reports/`
Start at [`reports/README.md`](../reports/README.md) — it labels every file
`PAPER` / `CONTEXT` / `SUPERSEDED` and states which split each result uses.

### `docs/`
This directory. `PIPELINE_GUIDE.md` for how it works, `RUN_GUIDE.md` for how to
run it, `GITHUB_GUIDE.md` for what to publish, and the four subject docs
(`dataset`, `models`, `results`, `reproducibility`) plus the 2024-25 ground-truth
record.

### `archive/` — local only, gitignored
`superseded_reports/` (earlier narratives whose metrics no longer match any
model) · `models_pre_leakage_fix/` (the 20 pre-fix artefacts) · `notebooks/`
(scratch notebooks with hard-coded paths) · `report_templates/` · 
`stale_root_reports/`. Nothing here is needed to run or reproduce the project;
it is kept for auditability.

---

## Where to start reading

| Question | File |
|---|---|
| What does this project do? | `README.md` |
| What are the results, and what can I cite? | `reports/summaries/research_ready_results.md` |
| Where is result X? | `reports/README.md` |
| How does the pipeline work? | `docs/PIPELINE_GUIDE.md` |
| How do I run it? | `docs/RUN_GUIDE.md` |
| Can I trust the numbers? | `CLEANUP_REPORT.md`, `docs/pipeline.md` |
| Why is there no 2024-25 accuracy? | `docs/2025_yield_reference_methodology.md` |
