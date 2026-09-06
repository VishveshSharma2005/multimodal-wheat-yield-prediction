# Data

The repository ships the small, derived tables needed to inspect and reproduce
the modelling. It does **not** ship the bulk raw inputs (~4.1 GB) or the large
intermediate CSVs. This file lists every excluded item, why it is excluded, where
it came from, and where to put it locally.

```
data/
├── raw/          original, unmodified source exports        (mostly NOT in Git)
├── interim/      cleaned, standardised tables               (partly in Git)
└── processed/    modelling-ready tables and outputs         (partly in Git)
```

Local footprint when fully populated: **~4.1 GB** (`raw` 3.0 GB, `processed`
1.2 GB, `interim` 7.7 MB). Tracked in Git: **~1.3 MB**.

---

## 1. What IS in Git

| Path | Size | Why kept |
|---|---|---|
| `processed/stage_features.csv` | 858 KB | **The modelling table.** 1485 rows x 55 columns — every real feature, target, split key and stage key. This is the file to inspect or re-model from. |
| `processed/targets.csv` | 6 KB | Per-case yield + quintile label |
| `processed/image_features.csv` | 186 KB | Per-image handcrafted features (560 rows) |
| `processed/image_stage_features.csv` | 1 KB | Per-stage image averages actually merged into the model |
| `processed/validation_stagewise_predictions.csv` | 98 KB | 297 raw 2023-24 predictions |
| `processed/prediction_2025_stagewise_predictions.csv` | 89 KB | 297 raw 2024-25 forecasts |
| `processed/mehsana_2025_stagewise_predictions.csv` | 3 KB | 9 Mehsana forecasts |
| `processed/lstm_predictions_2025.csv` | 2 KB | 33 LSTM forecasts |
| `interim/apy_yield_all_districts.csv` | 12 KB | **Authoritative APY target table** (33 districts x 4 years) |
| `interim/apy_yield.csv` | 2 KB | Mehsana-only APY subset |
| `interim/apy_2025_ground_truth.csv` | <1 KB | 2024-25 ingestion output — **header-only**: no official district-level 2024-25 APY exists (section 5) |
| `interim/cases.csv` | 20 KB | The 165 district-season cases |
| `interim/coordinate_registry.csv` | 19 KB | District lat/lon |
| `interim/image_metadata.csv` | 164 KB | Image inventory + stage labels |
| `raw/apy/*.csv` (3) | 5 KB | Hand-curated APY provenance/trace CSVs |
| `raw/apy/README_APY_CSV.md` | 1 KB | Provenance notes — read this before using any APY number |
| `raw/field_points/field_coordinates.csv` | <1 KB | Field/centroid coordinates |
| `raw/templates/*.csv` (4) | 1 KB | Empty schema stubs |

---

## 2. What is NOT in Git, and how to get it

### 2.1 Wheat growth-stage image dataset (CNN training data)

* **Local path:** `data/raw/images/wheat_stage_dataset/`
* **Size:** ~2.9 GB, **5,211 images**, 5 class folders
* **Excluded because:** thousands of large binaries; GitHub is the wrong place
  for an image corpus.
* **Expected structure:**

```
data/raw/images/wheat_stage_dataset/
├── 1_Tillering/    1037 images
├── 2_Jointing/     1202 images
├── 3_BH/           1192 images
├── 4_Flowering/    1164 images
└── 5_Filling/       616 images
```

* **Preprocessing:** none needed. `src/deep_learning/cnn_training_utils.py`
  builds the stratified 70/15/15 split, drops unreadable files (logged to
  `reports/model_results/corrupted_images.csv`), and writes
  `data/interim/cnn_dataset_split.csv`. Images are resized to the backbone's
  input size (224 or 299) at load time.
* **Needed for:** CNN training, CNN evaluation, CNN feature extraction. **Not**
  needed for the yield pipeline.

### 2.2 Mendeley WheatPhenology auxiliary image set (handcrafted features)

* **Local path:** `data/raw/images/auxiliary_wheat_images/WheatPhenology A Multi-Stage Field Image Dataset o/`
* **Source:** Mendeley Data — *WheatPhenology: A Multi-Stage Field Image Dataset*
  (recorded as `source_dataset = Mendeley_WheatPhenology` in
  `processed/image_features.csv`). Search Mendeley Data for the dataset title to
  obtain the DOI; record it here once confirmed.
* **Expected structure:** 5 folders — `1. Seedling & Plant`, `2. Wheat Flowers`,
  `3. Plant with Fruit`, `4. Fruit with Seeds`, `5. Wheat Fruits only` — mapped to
  agronomic stages by `configs/config.yaml: image_stage_label_map`.
* **Preprocessing:** `src/features/extract_image_features.py` computes green-pixel
  ratio, canopy-density proxy, vegetation-pixel ratio, brightness, saturation and
  a texture proxy per image; `src/data/build_image_stage_features.py` averages
  them per stage.

> **Integrity warning.** `processed/image_features.csv` references **560** images,
> but only **225** are present in this working copy (`1. Seedling & Plant` has 10
> of the 19 used, `4. Fruit with Seeds` 128 of 185, etc.). The committed
> `image_stage_features.csv` reflects all 560. **Re-running
> `extract_image_features.py` against an incomplete download will silently change
> the per-stage averages the saved models were trained on.** Restore the full
> dataset first, or leave `image_stage_features.csv` as-is.

### 2.3 Sentinel-2 exports

* **Local path:** `data/raw/satellite/`
* **Size:** 1.2 MB, **153 CSVs** named
  `sentinel2_GUJ_<DISTRICT>_<YYYY>_<YYYY>[_<FIELD|CENTROID>].csv`
* **Source:** Google Earth Engine, `COPERNICUS/S2_SR_HARMONIZED`. Regenerate with
  `python src/data/export_sentinel2_gee.py` (needs `earthengine-api` and an
  authenticated GEE account; the export task list is written to
  `reports/gee_export_tasks.md`). Download the resulting Drive exports into
  `data/raw/satellite/`.
* **Columns:** `case_id, field_id, district, date, ndvi_raw, ndre, evi, cloud_pct, source`
* **Excluded because:** regenerable, and gitignored alongside the other raw
  inputs for consistency. It is small enough to commit if you prefer — remove the
  `data/raw/satellite/` line from `.gitignore`.
* **Coverage note:** the 2024-25 season was exported for **Mehsana only**. The
  other 32 districts have satellite data for 2020-21 to 2023-24 only.

### 2.4 APY source workbooks and PDFs

* **Local path:** `data/raw/apy/`
* **Size:** 19 MB (`20-23.xlsx/.pdf`, `23-24.xlsx/.pdf`, `24-25.xlsx/.pdf`)
* **Source:** Directorate of Agriculture, Government of Gujarat —
  <https://dag.gujarat.gov.in/Home/AreaProductionAndYield>
  (mirror: <https://www.data.gov.in/catalog/area-production-and-yield-major-crops-gujarat-state>)
* **Excluded because:** redistributing government publication binaries is
  unnecessary — the extracted table (`interim/apy_yield_all_districts.csv`) is
  committed instead.
* **Preprocessing:** `python src/data/extract_apy_all_districts_xlsx.py`
  (requires `openpyxl`).
* **Licence:** Government of India Open Government Data licence. Cite the
  Directorate of Agriculture, Gujarat, as the source.
* **Coverage:** district-level 2020-21 to 2023-24. The 24-25 report is
  **state-level Final Advance Estimates only** — see section 5.

### 2.5 Weather

* **Local path:** none — `data/raw/weather/` was never populated; the downloader
  writes straight to `interim/`.
* **Source:** Open-Meteo Archive API (ERA5),
  <https://archive-api.open-meteo.com/v1/archive>. Free, no API key.
* **Regenerate:** `python src/data/download_weather.py` (needs network; a few
  minutes for 165 cases).
* **Excluded outputs:** `interim/weather_daily.csv` (2.4 MB, 20,856 rows) and
  `interim/weather_daily_points.csv` (3.1 MB).

### 2.6 Large derived tables

| Path | Size | Regenerate with |
|---|---|---|
| `processed/cnn_features.csv` | 618 MB | `src/deep_learning/extract_cnn_features.py` (GPU) |
| `processed/cnn_features/*.csv` (5) | 620 MB | same |
| `processed/cnn_stage_features.csv` | 785 KB | `src/deep_learning/build_cnn_stage_features.py` |
| `processed/stage_features_with_cnn.csv` | 11 MB | `src/data/build_stage_table.py` |
| `processed/stage_features_with_split.csv` | 11 MB | `src/data/build_stage_table.py` |
| `interim/cnn_dataset_split.csv` | 422 KB | any `src/deep_learning/train_*.py` |
| `interim/satellite_observations.csv` | 1.7 MB | `src/data/ingest_satellite.py` |

The two 11 MB `stage_features_with_*` files are 6,969 columns wide, of which
6,913 are the all-NaN `cnn_*` columns. The committed
`processed/stage_features.csv` carries the identical 55 real columns at 858 KB.

### 2.7 Not present at all

`data/raw/soil/`, `data/raw/drone/`, `data/raw/farm_logs/` are declared in
`configs/config.yaml` but were never populated. No soil, drone or farm-log
features exist in the modelling table. `data/raw/templates/` holds only empty
schema stubs.

---

## 3. Restoring a full local copy

```
data/raw/
├── apy/                       # 6 workbooks/PDFs from dag.gujarat.gov.in  (19 MB)
├── satellite/                 # 153 GEE export CSVs                       (1.2 MB)
├── images/
│   ├── wheat_stage_dataset/   # 5,211 images in 5 class folders           (2.9 GB)
│   └── auxiliary_wheat_images/WheatPhenology A Multi-Stage Field Image Dataset o/
│                              # 560 images in 5 folders
├── field_points/              # ships in Git
└── templates/                 # ships in Git
```

Then follow `docs/reproducibility.md` section 6.

---

## 4. `data/sample/`

Not created. `processed/stage_features.csv` is already small enough (858 KB) to
serve as the reproducibility sample, so a separate cut-down copy would only be a
second thing to keep in sync.

---

## 5. Ground-truth warning for 2024-25

**There is no official district-level 2024-25 Gujarat wheat APY.**
`interim/apy_2025_ground_truth.csv` is therefore header-only, and every 2024-25
row in `cases.csv` and the stage tables has a null `yield_kg_ha`,
`actual_available = 0` and `evaluation_status = pending_actual_apy`.

Until 2026-09-05 a carry-forward fallback in
`src/data/ingest_apy_2025_ground_truth.py` had written Mehsana's 2023-24 yield
(2796.06 kg/ha) into the 2024-25 cases as a real actual. That fallback has been
removed. Read
[`docs/2025_yield_reference_methodology.md`](../docs/2025_yield_reference_methodology.md)
before using any 2024-25 value.
