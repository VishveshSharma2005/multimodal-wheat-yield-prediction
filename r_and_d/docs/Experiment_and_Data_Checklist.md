# Experiment + Data Checklist (Stage-wise Yield Updates)

Use this as a day-to-day checklist. If you follow this, your pipeline will be “PhD clean”.

---

## A) Define your cases (one field-season = one case)
- [ ] Choose case ID format (example): `FarmA_Field07_2025-26`
- [ ] For each case, confirm you have:
  - [ ] field boundary
  - [ ] sowing date
  - [ ] harvest date
  - [ ] final yield label (kg/ha)

---

## B) Data collection checklist (minimum viable)

### B1) Satellite
- [ ] Extract Sentinel‑2 time series at least every ~5 days (cloud permitting)
- [ ] Store: NDVI, NDRE (optional), EVI (optional)
- [ ] Cloud mask applied consistently

### B2) Drone
- [ ] Fly weekly or every 10–14 days from emergence → grain filling
- [ ] Store orthomosaic + (at least) canopy cover proxy
- [ ] If thermal available, store canopy temperature

### B3) Farm visits
- [ ] At least once per stage
- [ ] Record events with dates:
  - [ ] irrigation
  - [ ] fertilizer (type/dose)
  - [ ] weed control
  - [ ] pest/disease score (0–5)

### B4) Yield ground truth
- [ ] Choose method and apply consistently
- [ ] Record harvested area + total weight
- [ ] Moisture correction method recorded

---

## C) Preprocessing checklist
- [ ] All data timestamped and aligned to the season window
- [ ] Missing values strategy decided and written
- [ ] Outliers identified with a clear rule
- [ ] No target leakage (stage-wise features use only past data)

---

## D) Weed correction checklist (“after removing weeds”)
- [ ] Compute weed fraction per date: `weed_cover_%`
- [ ] Compute crop-only index per date: `NDVI_crop_only`
- [ ] Create trajectory features:
  - [ ] `NDVI_crop_only_last`
  - [ ] `NDVI_crop_only_mean`
  - [ ] `NDVI_crop_only_slope`
- [ ] Compare models with/without weed correction

---

## E) Stage-wise modeling checklist

### E1) Stage definitions
- [ ] Decide stage boundaries (calendar dates or GDD)
- [ ] For each case, fill a stage timeline table

### E2) Features per stage
For each stage $S$, compute features up to stage end date.
- [ ] Weather aggregates (sum/mean/max)
- [ ] Satellite/drone aggregates (mean/last/slope)
- [ ] Management event counts/totals
- [ ] Crop development clock: days-after-sowing, cumulative GDD

### E3) Models
- [ ] Baseline tabular: LightGBM/XGBoost
- [ ] Optional temporal DL: LSTM/TCN

---

## F) Evaluation checklist
- [ ] Choose split:
  - [ ] time split (earlier seasons train, later test)
  - [ ] spatial holdout (hold out fields)
- [ ] Report metrics for every stage:
  - [ ] RMSE, MAE, R2
  - [ ] Macro‑F1 for Q1–Q5
- [ ] Plot stage vs error (should decrease over stages)

---

## G) Ablation checklist (recommended)
- [ ] Weather-only vs Weather+Soil vs Weather+Soil+Imagery
- [ ] With NDVI vs Without NDVI
- [ ] With weed correction vs Without
- [ ] Tabular vs Temporal DL
- [ ] Continuous inputs vs Continuous + input-quintile features
