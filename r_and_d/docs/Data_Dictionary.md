# Data Dictionary (What each input/output means)

This dictionary helps you keep the dataset consistent and beginner-readable.

---

## 1) Identifiers
- `case_id`: unique field-season ID (one field for one season)
- `farm_id`: farm identifier
- `field_id`: field identifier
- `season`: string like `2025-26`

---

## 2) Targets (Outputs)

### 2.1 Regression target
- `yield_kg_ha`: final wheat yield at harvest (kg/ha)

### 2.2 Quintile target (optional)
- `yield_quintile`: integer 1..5, where 1=lowest yields (Q1) and 5=highest yields (Q5)

Notes:
- Quintile thresholds must be computed on **training data only** to avoid leakage.

---

## 3) Static features (field-level; constant within a season)
- `soil_ph`
- `soil_ec_ds_m`
- `soil_organic_carbon_pct`
- `soil_n`, `soil_p`, `soil_k` (include units in your dataset notes)
- `soil_texture_class`
- `field_area_ha`
- `irrigation_type`
- `wheat_variety`

---

## 4) Temporal raw observations (stored by date)
If you keep a long-format table, you will have rows like (case_id, date).

### 4.1 Weather
- `rain_mm`
- `tmin_c`
- `tmax_c`
- `tmean_c`

### 4.2 Satellite/drone indices
- `ndvi_raw`
- `ndre_raw` (optional)
- `evi_raw` (optional)
- `canopy_cover_pct` (from drone or derived)
- `canopy_temp_c` (if thermal)

### 4.3 Farm visit / management logs
- `irrigation_event` (0/1) + optional `irrigation_amount_mm`
- `fertilizer_n_kg_ha`, `fertilizer_p_kg_ha`, `fertilizer_k_kg_ha`
- `weed_control_event` (0/1)
- `pest_score_0_5`
- `disease_score_0_5`

---

## 5) Weed correction features (“after removing weeds”)
- `weed_cover_pct`: % of field covered by weeds (estimated from imagery)
- `crop_cover_pct`: % covered by crop
- `ndvi_crop_only`: NDVI computed after masking weeds

---

## 6) Stage-wise engineered features (model-ready)
These are computed up to each stage end date, and used for stage-wise prediction.

### 6.1 Development clock
- `das`: days after sowing at stage end
- `gdd_cum`: cumulative Growing Degree Days at stage end

### 6.2 Weather aggregates (up to stage end)
- `rain_sum`
- `tmax_mean`
- `heat_days_count` (e.g., days with tmax above threshold)

### 6.3 Crop state aggregates (up to stage end)
- `ndvi_last`, `ndvi_mean`
- `ndvi_slope` (trend)
- `ndvi_crop_only_last`, `ndvi_crop_only_slope`
- `canopy_cover_last`, `canopy_cover_slope`
- `canopy_temp_anomaly_mean` (if thermal)

### 6.4 Management aggregates
- `irrigation_events_count`
- `fertilizer_n_total`
- `weed_control_events_count`

---

## 7) Optional: input quintiles
If required, create quintiles for selected inputs:
- Example: `rain_sum_quintile` in 1..5
- Keep continuous original too.
