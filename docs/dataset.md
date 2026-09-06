# Dataset

All figures below were measured from the files in this repository, not copied
from an earlier document.

---

## 1. Unit of analysis

* **Case** = district x wheat season, e.g. `GUJ_MEHSANA_2024_2025`.
* **Row** = case x growth stage.
* 33 districts x 5 seasons = **165 cases**; 165 x 9 stages = **1485 rows**.
* The target is **final harvest yield (kg/ha)**, repeated across the 9 stages of
  a case. The models do not predict a per-stage yield; they predict the season's
  final yield from information available up to the end of each stage.

## 2. Temporal coverage and splits

| Split | Agri years | `season_year_start` | Cases | Rows | Target |
|---|---|---|---|---|---|
| train | 2020-21, 2021-22, 2022-23 | 2020, 2021, 2022 | 99 | 891 | District APY yield |
| validation | 2023-24 | 2023 | 33 | 297 | District APY yield |
| forecast | 2024-25 | 2024 | 33 | 297 | **Not available** — `pending_actual_apy` |

Per stage that is 99 training and 33 validation observations.

The split is assigned by `season_year_start`, hard-coded identically in
`build_stage_table.py`, `train_stagewise_regression.py`,
`train_quintile_classifier.py` and `train_lstm_sequence_model.py`.

## 3. District coverage (33)

Ahmedabad, Amreli, Anand, Arvalli, Banaskantha, Bharuch, Bhavnagar, Botad,
Chhota Udepur, Dahod, Dangs, Devbhoomi Dwarka, Gandhinagar, Gir Somnath,
Jamnagar, Junagadh, Kachchh, Kheda, Mahisagar, **Mehsana**, Morbi, Narmada,
Navsari, Panchmahal, Patan, Porbandar, Rajkot, Sabarkantha, Surat,
Surendranagar, Tapi, Vadodara, Valsad.

Models are trained on all 33; **Mehsana** is the focus district for the forecast
narrative.

## 4. Growth stages (9)

Defined in `configs/config.yaml` as days-after-sowing windows and applied by
`src/features/crop_stages.py`. Crop duration is 125-126 days.

| Order | Stage | DAS window |
|---|---|---|
| 1 | `sowing` | 0-10 |
| 2 | `early_vegetative` | 10-20 |
| 3 | `tillering` | 20-40 |
| 4 | `stem_elongation` | 40-60 |
| 5 | `booting_heading` | 60-80 |
| 6 | `flowering` | 80-95 |
| 7 | `grain_filling_initial` | 95-105 |
| 8 | `grain_filling` | 105-115 |
| 9 | `maturity_preharvest` | 115-130 |

Verified: no duplicate `(district, season_year_start, stage)` keys, and
`stage_end_das` is monotonically increasing within all 165 cases.

## 5. Feature modalities

`data/processed/stage_features.csv` has 55 columns: 22 keys/metadata/target and
33 candidate features. Per stage the regression selects ~40 columns after
one-hot encoding.

### 5.1 Weather — Open-Meteo Archive (12 features)

`rain_sum`, `cumulative_rain_mm`, `cumulative_rainfall`, `tmin_mean`,
`tmax_mean`, `mean_tmin_c`, `mean_tmax_c`, `max_tmax_c`, `heat_stress_days`
(days with Tmax >= 32 C), `gdd_cum`, `gdd_cumulative`, `cumulative_gdd`,
`gdd_norm`.

Source: `https://archive-api.open-meteo.com/v1/archive` (ERA5 reanalysis),
`source = open_meteo_point_mean` in `data/interim/weather_daily.csv`.
20,856 daily rows, 2020-11-20 to 2025-03-25, covering all 165 cases.

> **Correction to earlier project write-ups:** weather is Open-Meteo, **not**
> NASA POWER. `configs/config.yaml` declares `provider: open_meteo` and
> `download_weather.py` calls only that endpoint.

Note that several weather columns are exact duplicates of each other
(`rain_sum` = `cumulative_rain_mm` = `cumulative_rainfall`;
`tmin_mean` = `mean_tmin_c`; `gdd_cum` = `gdd_cumulative` = `cumulative_gdd`).
They are retained because the trained models were fitted with them present.

### 5.2 Satellite — Sentinel-2 via Google Earth Engine (12 features)

`ndvi_last`, `ndvi_mean`, `ndvi_max`, `ndvi_slope`, `ndvi_growth_rate`,
`ndre_last`, `ndre_mean`, `ndre_slope`, `evi_last`, `evi_mean`, `evi_slope`.

Source: `COPERNICUS/S2_SR_HARMONIZED`, exported per district-season by
`src/data/export_sentinel2_gee.py`; 153 CSVs under `data/raw/satellite/`,
ingested to 6,779 observations in `data/interim/satellite_observations.csv`
(`source = GEE_S2_SR_HARMONIZED`), 2020-11-20 to 2025-03-22.

**Coverage gap (important):**

| Split | Rows with NDVI |
|---|---|
| train | 891 / 891 |
| validation | 297 / 297 |
| forecast 2024-25 | **9 / 297 — Mehsana only** |

The 2024-25 GEE exports were produced for Mehsana only. For the other 32
districts the 2024-25 vegetation indices are NaN and the pipeline's median
imputer fills them with training-period values. Those 288 forecast rows are
therefore driven almost entirely by weather, location and calendar features.
Only the 9 Mehsana forecast rows are genuinely multi-modal.

### 5.3 Handcrafted image features (6 features) — carry no district or year signal

`image_green_pixel_ratio_stage_avg`, `image_canopy_density_proxy_stage_avg`,
`image_vegetation_pixel_ratio_stage_avg`, `image_brightness_mean_stage_avg`,
`image_saturation_mean_stage_avg`, `image_texture_proxy_stage_avg`, plus
`image_count_stage`, `image_coverage_flag`, `image_feature_source`.

Source: the **Mendeley "WheatPhenology: A Multi-Stage Field Image Dataset"**
auxiliary set (`source_dataset = Mendeley_WheatPhenology`), 560 images mapped to
5 of the 9 stages by `configs/config.yaml: image_stage_label_map`.

These are computed as a **single global mean per stage** and then broadcast to
every district and every year:

| Stage | `image_green_pixel_ratio_stage_avg` | `image_count_stage` |
|---|---|---|
| early_vegetative | 0.880794 | 19 |
| flowering | 0.667206 | 63 |
| grain_filling_initial | 0.642078 | 185 |
| grain_filling | 0.061284 | 238 |
| maturity_preharvest | 0.000841 | 55 |
| sowing, tillering, stem_elongation, booting_heading | — (44.4% of rows are NaN) | — |

Because the regression is fitted **per stage**, these six columns are *constant
within every model's training set* and therefore contribute exactly zero
information to the stage-wise regressors. The repository's own ablation study
confirms this: adding them moves validation RMSE from 651.52 to 653.33
(`reports/model_results/cnn_ablation_study.csv`).

**Reproducibility caveat:** 335 of the 560 image paths referenced in
`data/processed/image_features.csv` are no longer present on disk (225 remain).
Re-running `src/features/extract_image_features.py` would produce *different*
stage averages than the ones the saved models were trained on. Do not re-run it
without restoring the full Mendeley download — see `data/README.md`.

### 5.4 CNN features (6,913 columns) — present but 100% empty

`cnn_feat_0000` ... `cnn_feat_6911` and `cnn_image_count` exist in
`stage_features_with_cnn.csv` / `stage_features_with_split.csv` and are
**entirely NaN in all 1485 rows**. They are dropped by the ">90% missing" filter
before any model sees them. See `docs/models.md` section 1.

### 5.5 Location / calendar / crop metadata

`district` (one-hot), `lat`, `lon`, `stage`, `stage_order`, `stage_index`,
`stage_start_das`, `stage_end_das`, `days_after_sowing`, `crop_duration_days`,
`crop_name` (wheat), `season_type` (rabi).

`lat`/`lon` are present for only 14 of 33 districts (57.6% of rows are NaN):
Ahmedabad, Anand, Banaskantha, Bhavnagar, Gandhinagar, Jamnagar, Junagadh,
Kheda, Mehsana, Patan, Rajkot, Sabarkantha, Surat, Vadodara.

### 5.6 Soil

**No soil features are present.** No SoilGrids / ISRIC ingestion exists in the
code, `data/raw/soil/` is absent, and `data/raw/templates/soil_template.csv` is
an empty schema stub. Earlier project write-ups that list soil as a modality are
incorrect.

## 6. Target

`yield_kg_ha` — district-level wheat yield from Gujarat Area-Production-Yield
statistics. Authoritative table:
`data/interim/apy_yield_all_districts.csv` (132 rows = 33 districts x 4 years,
parsed from `20-23.xlsx` and `23-24.xlsx`).

| Split | Rows | Non-null target |
|---|---|---|
| train | 891 | 891 |
| validation | 297 | 297 |
| forecast 2024-25 | 297 | **should be 0** (currently 9, contaminated) |

Observed range 0 - 4486.98 kg/ha; train mean 3082.43, validation mean 3080.03.

> Some districts record a 0.00 yield in some years (non-wheat-growing or
> unreported districts). Those rows are retained; they inflate MAPE where they
> occur and are the reason MAPE is computed with a zero-safe denominator.

## 7. Missing-value summary (non-CNN columns)

| Column group | % missing | Reason |
|---|---|---|
| `area_ha` | 100.0 | Never populated; field-level areas were never collected |
| `lat`, `lon` | 57.6 | Coordinates registered for 14 of 33 districts |
| `image_*` | 44.4 | Mendeley images map to only 5 of 9 stages |
| `yield_kg_ha` | 19.4 | The 2024-25 forecast split has no published target |
| `ndvi_*`, `ndre_*`, `evi_*` | 19.4 | 2024-25 GEE exports exist for Mehsana only |
| everything else | 0.0 | — |

Missing numeric values are median-imputed inside each model pipeline
(`SimpleImputer(strategy="median")`), fitted on training rows only. Columns
above 90% missing are dropped before fitting.

## 8. Data sources — verified

| Modality | Source | Verified in |
|---|---|---|
| Yield (APY) | Directorate of Agriculture, Government of Gujarat — <https://dag.gujarat.gov.in/Home/AreaProductionAndYield> | `data/raw/apy/*.xlsx`, `README_APY_CSV.md` |
| Weather | Open-Meteo Archive (ERA5) — <https://archive-api.open-meteo.com/v1/archive> | `configs/config.yaml`, `weather_daily.csv` (`source` column) |
| Satellite | Sentinel-2 `COPERNICUS/S2_SR_HARMONIZED` via Google Earth Engine | `satellite_observations.csv` (`source` column) |
| Field images (CNN) | Wheat growth-stage image set, 5,211 images, 5 classes | `data/raw/images/wheat_stage_dataset/` |
| Auxiliary images (handcrafted) | Mendeley *WheatPhenology: A Multi-Stage Field Image Dataset* | `image_features.csv` (`source_dataset` column) |
| Soil | **not used** | — |
| Drone / farm logs | **not used** (templates only) | `data/raw/templates/` |
