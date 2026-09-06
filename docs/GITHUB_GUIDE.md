# GitHub guide

What belongs in the public repository, what must stay out, and how to obtain the
excluded artefacts.

**Current state:** 218 files, ~16 MB staged, largest tracked file 1.1 MB, no
secrets, no remote configured, nothing committed.

---

## COMMIT TO GITHUB

### Source code — all of it
```
src/data/          (20)  ingestion + stage-table construction
src/features/      (7)   feature engineering + feature_policy.py
src/models/        (8)   regression, quintile, LSTM, prediction, temporal CV,
                         within-district skill, ablation
src/deep_learning/ (11)  CNN scripts (code only — weights excluded)
src/visualization/ (1)   plot_results.py
src/utils/         (4)   config, paths, logging, cleanup
scripts/smoke_test.py
run_baseline_ml.py  run_training.py  run_training_with_cnn.py
```

### Configuration and environment
```
configs/config.yaml
requirements.txt              pinned; the .joblib files are sklearn-version locked
requirements_colab_cnn.txt    TensorFlow, CNN branch only
.gitignore  .gitattributes
```

### Documentation
```
README.md
CLEANUP_REPORT.md             the audit trail — worth publishing, it is the
                              evidence that the corrections were applied
docs/                         all 10 files
data/README.md  data/raw/README.md  data/interim/README.md  data/processed/README.md
reports/README.md
r_and_d/docs/                 stage definitions, data dictionary
```

### Final results — small and essential
```
reports/summaries/            (6) research_ready_results.md, final_model_summary.csv,
                                  temporal_validation_summary.md, baseline_comparison.md,
                                  within_district_skill.md, forecast_2024_25.md
reports/model_results/        (63) metrics CSVs, feature importances, confusion
                                   matrices, CNN evaluation, feature audit
reports/model_results/superseded/  (8) pre-leakage-fix provenance — see below
reports/figures/              (5) all final figures
reports/*.md                  data-pipeline diagnostics
```

> `reports/model_results/superseded/` is intentionally **committed**. It is 8 small
> files that let a reviewer verify the before/after of the leakage fix. Its README
> states clearly that nothing in it may be cited.

### Small data required for reproducibility
```
data/processed/stage_features.csv                       858 KB — THE modelling table
data/processed/targets.csv
data/processed/image_features.csv, image_stage_features.csv
data/processed/validation_stagewise_predictions.csv     297 rows, validation
data/processed/prediction_2025_stagewise_predictions.csv 297 rows, forecast
data/processed/mehsana_2025_stagewise_predictions.csv   9 rows, forecast
data/processed/lstm_predictions_2025.csv                33 rows, forecast
data/interim/apy_yield_all_districts.csv                the authoritative target table
data/interim/{apy_yield, apy_2025_ground_truth, cases,
              coordinate_registry, image_metadata}.csv
data/raw/apy/*.csv + README_APY_CSV.md                  provenance CSVs
data/raw/field_points/field_coordinates.csv
data/raw/templates/*.csv                                empty schema stubs
```

`stage_features.csv` carries the identical 55 real feature columns as the 11 MB
`stage_features_with_*.csv` files, at 1/13th the size. It is enough to re-model
from without any raw data.

### Model artefacts — the small ones
```
models/regression/*.joblib        (10)  ~1.3 MB total
models/classification/*.joblib    (9)   ~9.5 MB total
models/deep_learning/lstm_yield_model.pt (1) ~295 KB
```

These make the repository usable for inference straight after cloning.

---

## DO NOT COMMIT

| Item | Size | Why |
|---|---|---|
| `data/raw/images/` | **2.9 GB** | 5,437 image files; GitHub is the wrong host for an image corpus |
| `data/processed/cnn_features.csv` | **618 MB** | regenerable deep-feature dump |
| `data/processed/cnn_features/*.csv` (5) | **620 MB** | same |
| **`models/cnn/*.keras` (5)** | **271 MB** | large binary weights — see distribution options below |
| `data/raw/apy/*.xlsx`, `*.pdf` | 19 MB | government publications; the extracted table is committed instead |
| `data/processed/stage_features_with_{cnn,split}.csv` | 11 MB each | 6,913 of 6,969 columns are empty |
| `data/interim/weather_daily*.csv` | 5.5 MB | regenerable from the Open-Meteo API |
| `data/interim/satellite_observations.csv` | 1.7 MB | regenerable from the GEE exports |
| `data/raw/satellite/` (153 CSVs) | 1.2 MB | regenerable; small enough to commit if you prefer |
| `reports/model_results/ml_selected_features.csv` | 3.9 MB | 62k-row audit trail |
| `reports/model_results/lstm_summary.md`, `lstm_selected_features.csv` | 540 KB | per-feature audit trails |
| `reports/feature_diagnosis_report.md` | 507 KB | 6,969-column missingness dump |
| `archive/` | 137 MB | superseded reports, scratch notebooks, pre-fix artefacts |
| `.venv/`, `.venv_broken_linux_leftover/` | 126 MB | virtual environments |
| `__pycache__/`, `.ipynb_checkpoints/`, `anaconda_projects/` | — | caches and IDE state |
| `.env`, `*.pem`, `*.key`, `credentials*.json` | — | secrets (none exist — the tree is clean) |

All of the above are already covered by `.gitignore`. Verify with:

```powershell
git check-ignore -v data\raw\images models\cnn archive
git status --ignored --porcelain | Select-String '^!!'
```

---

## How to obtain the excluded artefacts

### CNN weights (271 MB) — required only for the CNN branch

Not needed for the yield pipeline, the smoke test, inference, or any result in
`reports/summaries/`. Choose one distribution route:

1. **Zenodo (recommended).** Upload the five `.keras` files, cite the DOI in the
   README. A DOI is also citable in a thesis, which solves two problems at once.
2. **GitHub Release asset.** Attach them to a tagged release. Not counted against
   repository size, but no DOI.
3. **Git LFS.** `git lfs track "models/cnn/*.keras"`. Works, but GitHub's free tier
   gives 1 GB storage / 1 GB bandwidth per month — 271 MB per clone exhausts that
   in three clones.
4. **Retrain.** `python src/deep_learning/run_cnn_pipeline.py` — needs the
   5,211-image dataset and a GPU, ~2.5 h. Results will not be bit-identical.

Expected local path either way: `models/cnn/<name>_best.keras`.
Loading requires **Keras 3 / TensorFlow ≥ 2.16**.

### Raw datasets

| Dataset | Source |
|---|---|
| APY workbooks | <https://dag.gujarat.gov.in/Home/AreaProductionAndYield> |
| Weather | Open-Meteo Archive API (free, no key) — `python src\data\download_weather.py` |
| Sentinel-2 | Google Earth Engine — `python src\data\export_sentinel2_gee.py`, then download the Drive exports |
| CNN image set | 5,211 images in 5 class folders under `data/raw/images/wheat_stage_dataset/` |
| Auxiliary images | Mendeley Data — *WheatPhenology: A Multi-Stage Field Image Dataset* |

Full instructions, expected directory layout and licence notes:
[`../data/README.md`](../data/README.md).

### Large derived tables

All regenerable — see the "Data preparation" section of
[`RUN_GUIDE.md`](RUN_GUIDE.md).

---

## Pre-publication checklist

```powershell
python scripts\smoke_test.py                      # expect 0 FAIL
git status
git ls-files | Measure-Object -Line               # expect ~218
git ls-files | ForEach-Object { Get-Item $_ } | Sort-Object Length -Descending |
    Select-Object -First 5 Length, Name           # expect nothing above ~1.2 MB
git status --ignored --porcelain | Select-String '^!!'
```

Confirm before publishing:

- [ ] No file over ~2 MB tracked
- [ ] `models/cnn/`, `data/raw/images/`, `data/processed/cnn_features/` all ignored
- [ ] `archive/` and `.venv*` ignored
- [ ] No secrets (verified: none in the tree)
- [ ] `README.md` headline finding is present and accurate
- [ ] `reports/README.md` reflects the current file set
- [ ] CNN described as an independent phenology classifier, never as fused
- [ ] No 2024-25 accuracy metric anywhere

## Not done here — deliberately

No commit was made, no remote was added, nothing was pushed. The repository is
staged and ready; publishing is your decision.

Suggested first commit message:

```
Stage-wise multi-modal wheat yield prediction: audited pipeline, leakage fixes,
temporal validation and baselines
```
