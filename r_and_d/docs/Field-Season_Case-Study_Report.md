# Farm & Field-Season Case Study Report (Beginner-Friendly Template)

> Purpose: This document is a **fill-in report** you can submit as a “whole report of the farm” and a clean R&D pipeline write-up.
>
> Rule of the project: **one field-season = one case** and you do **stage-wise yield updates** using **temporal features**.

---

## 0) Quick Summary (1 page)

### 0.1 Case ID
- Farm name / code: `[________]`
- Field name / ID: `[________]`
- Season: `[e.g., 2025–26]`
- Crop: `Wheat`
- Location (village/taluk/district/state): `[________]`
- GPS centroid (lat, lon): `[________]`
- Field area (ha): `[________]`

### 0.2 What you predicted
- Primary target: **Final yield at harvest (kg/ha)**
- Secondary target (optional): **Yield quintile class (Q1–Q5)**

### 0.3 Data sources used
Tick what you have:
- [ ] Soil lab report (pH, EC, OC, N/P/K, texture)
- [ ] Satellite (e.g., Sentinel‑2 indices time series)
- [ ] Drone flights (weekly / 10–14 days)
- [ ] Weather (station or gridded)
- [ ] Farm visit logs (management + pest/disease)
- [ ] Ground-truth yield (plot harvest / weighing + moisture correction)

### 0.4 Key result snapshot (fill later)
- Best stage-wise model: `[e.g., LightGBM]`
- Best stage to make confident decisions: `[e.g., booting/heading]`
- Final test metrics (example):
  - Regression: RMSE `[__]`, MAE `[__]`, $R^2$ `[__]`
  - Quintile: Macro‑F1 `[__]`

---

## 1) Background (Simple Explanation)

### 1.1 Why stage-wise (cycle-by-cycle) prediction?
A single static snapshot (one date of NDVI) misses the crop’s **trajectory**. Wheat yield depends on:
- early stand establishment,
- mid-season growth and stress,
- grain filling conditions.

So we predict final yield at **multiple stages**, and accuracy should improve as the season progresses.

### 1.2 What is one “case”?
A **case** is one complete field-season record:
- one field boundary
- one sowing date → harvest date
- all temporal data aligned to that season
- one final yield value

---

## 2) Farm Profile (Fill-In)

### 2.1 Farm overview
- Farmer name (optional): `[________]`
- Farm size (ha): `[________]`
- Irrigation type: `[canal / borewell / drip / flood / mixed]`
- Constraints noted by farmer: `[water shortage / salinity / labor / pests / etc.]`

### 2.2 Field boundary & map
- Field boundary file: `[GeoJSON / KML / Shapefile / drawn polygon]`
- How boundary was obtained: `[GPS walk / app drawing / drone ortho]`
- Include a map figure name: `[docs/figures/FieldBoundary.png]`

### 2.3 Soil & terrain (static features)
Attach soil report if available and summarize:
- pH: `[__]`
- EC (dS/m): `[__]`
- Organic carbon (%): `[__]`
- Available N/P/K (units): `[__]`
- Texture: `[sandy loam / clay / etc.]`
- Water holding capacity (if available): `[__]`
- Slope/drainage notes: `[flat / mild slope / waterlogging zones]`

---

## 3) Crop Lifecycle Timeline (Must-Have)

### 3.1 Key dates
- Sowing date: `[____-__-__]`
- Emergence (observed): `[____-__-__]`
- First irrigation: `[____-__-__]`
- First fertilizer: `[____-__-__]`
- Weed control event(s): `[date + method]`
- Pest/disease event(s): `[date + severity]`
- Harvest date: `[____-__-__]`

### 3.2 Stage definitions (use these unless you have better)
You can stage-align by **calendar days** or (better) by **Growing Degree Days (GDD)**.

Recommended stages to predict at:
1. Pre‑sowing / at sowing
2. Emergence / early vegetative
3. Tillering
4. Stem elongation
5. Booting / heading
6. Flowering
7. Grain filling
8. Pre‑harvest

For each stage, record:
- Stage start date: `[____-__-__]`
- Stage end date: `[____-__-__]`
- Approx. GDD at stage end (if you compute GDD): `[__]`

---

## 4) Data Sources (What, Where, How Often)

### 4.1 Weather (daily)
- Source: `[nearest station / IMD / ERA5 / other]`
- Variables collected:
  - rainfall, Tmin, Tmax
  - humidity or VPD proxy (optional)
  - wind, solar radiation (optional)
- Quality checks you did:
  - missing days handled by `[forward fill / interpolation / leave as NaN + model handles]`

### 4.2 Satellite time series (e.g., Sentinel‑2)
- Product/source: `[Sentinel‑2 L2A / GEE / other]`
- Frequency: `[~5 days cloud permitting]`
- Indices extracted (examples):
  - NDVI, EVI, NDRE
- Cloud masking: `[method: SCL mask / QA60 / threshold]`
- Spatial reduction over field: `[mean / median / percentile]`

### 4.3 Drone flights
- Drone & sensor: `[RGB / multispectral / thermal]`
- Flight schedule: `[weekly / every 10–14 days]`
- Flight altitude & GSD (if known): `[__]`
- Outputs you kept:
  - orthomosaic
  - canopy cover % (or proxy)
  - thermal canopy temperature (if available)

### 4.4 Farm visits (ground observations)
Minimum: at least once per stage.

Record these (simple, numeric scales are fine):
- Pest score: `[0–5]`
- Disease score: `[0–5]`
- Weed pressure score: `[0–5]`
- Plant height (cm): `[__]` (optional)
- LAI proxy: `[visual scale / app measurement]` (optional)
- Management events (date, type, dose):
  - irrigation
  - fertilizer (N/P/K)
  - herbicide/weeding

---

## 5) Ground Truth Yield (Critical Section)

### 5.1 Yield measurement method (pick one and stick to it)
Choose and describe:
- [ ] Harvest plot sampling + weighing + moisture correction
- [ ] Full-field combine yield (if reliable)
- [ ] Farmer report (least preferred)

### 5.2 Moisture correction (if you did it)
- Measured grain moisture (%): `[__]`
- Corrected to standard moisture (%): `[__]`
- Formula used (write here): `[________]`

### 5.3 Final label
- Final yield (kg/ha): `[________]`

---

## 6) Preprocessing & Quality Control (QC)

### 6.1 Data alignment (how everything is aligned)
Explain in plain words:
- You align all observations by **date** and compute features **only up to each stage**.
- No leakage: when predicting at stage $S$, you do not use any data after stage $S$.

### 6.2 Handling missing values
- Weather gaps: `[________]`
- Cloudy satellite gaps: `[________]`
- Missing drone dates: `[________]`

### 6.3 Outlier checks
- Yield outliers: `[rule used]`
- NDVI outliers: `[rule used]`
- Rainfall spikes: `[rule used]`

---

## 7) Weed Detection + “Crop-Only Vigor After Removing Weeds” (Non-Negotiable if required)

### 7.1 What “after removing weeds” means (simple definition)
You compute two versions of vegetation/vigor:
- **Raw** (crop + weeds): e.g., `NDVI_raw`
- **Crop-only** (mask weeds out): e.g., `NDVI_crop_only`

### 7.2 Step A: Weed fraction estimation
Pick one approach (keep it simple for beginners):
- Option 1 (best): segmentation model (crop vs weed vs soil)
- Option 2 (basic proxy): threshold + texture rules + manual calibration

What you store per date:
- `weed_cover_%`
- `crop_cover_%`

### 7.3 Step B: Crop-only indices
Using the weed mask:
- compute `NDVI_crop_only`
- compute growth rate features like `NDVI_crop_only_slope`

### 7.4 Weed method evaluation (small but honest)
- Labels created: `[how many images/plots?]`
- Metric: `[IoU / F1 / accuracy]`
- Result: `[__]`

---

## 8) Feature Engineering (Temporal Features Are the Core)

### 8.1 Feature groups (what goes into the model)
1) Static (field-level): soil + location + variety/irrigation type
2) Temporal: weather + satellite/drone indices + farm visits
3) Crop state: NDVI trajectory, canopy cover trajectory, thermal stress

### 8.2 GDD and phenology (beginner explanation)
Growing Degree Days helps compare seasons fairly.
- Choose base temperature $T_{base}$ (for wheat, often around 0–5°C; use what your advisor prefers).
- Daily mean temperature: $T_{mean} = (T_{max}+T_{min})/2$
- Daily GDD: $\max(0, T_{mean}-T_{base})$
- Cumulative GDD is the sum from sowing.

Write your chosen $T_{base}$ here: `[__]`

### 8.3 Stage-wise feature aggregation (example pattern)
For each stage $S$ you can compute, up to stage end date:
- weather totals/means: rainfall_sum, Tmax_mean, heat_days_count
- crop index level: NDVI_last, NDVI_mean
- crop growth rate: NDVI_slope (trend), NDVI_curvature (optional)
- stress: canopy_temp_anomaly_mean
- management counts: irrigation_events_count, fertilizer_N_total

### 8.4 Input quintiles (optional)
If you must use quintile inputs, do it like this:
- For each continuous variable $x$, create `x_quintile` with values 1..5.
- Keep the original continuous variable too (recommended).

---

## 9) Modeling Setup

### 9.1 Targets
- Regression target: yield (kg/ha)
- Classification target (optional): yield quintile Q1–Q5

How to compute yield quintiles:
- From training data yields only
- Split into 5 equal-frequency bins (Q1 low → Q5 high)

### 9.2 Stage-wise prediction design
Choose one (and state your choice):
- [ ] Option A: **one model per stage** (simplest to report)
- [ ] Option B: one temporal model (sequence input) for all stages

### 9.3 Models to implement (recommended order)
1) Baseline tabular model (must-have): XGBoost / LightGBM / RandomForest
2) Temporal deep model (R&D): LSTM or TCN (Transformer optional)

### 9.4 Preventing data leakage (write this clearly)
- When predicting at stage $S$, you only use features computed from dates **≤ stage end date**.
- You compute scaling/normalization **within training fold** only.

---

## 10) Evaluation Protocol (PhD-clean)

### 10.1 Data splits (choose what you can do)
- Time split: train on earlier seasons, test on later season
- Spatial holdout: hold out one farm/field for test (stronger)

Write your split rule:
- Train: `[seasons/fields]`
- Test: `[seasons/fields]`

### 10.2 Metrics (report at every stage)
Regression:
- RMSE, MAE, $R^2$

Quintile classification:
- Macro‑F1
- Confusion matrix

### 10.3 Stage-wise results table (fill after experiments)

| Stage | RMSE | MAE | R2 | Macro‑F1 (Q1–Q5) | Notes |
|------|------|-----|----|------------------|-------|
| Sowing | | | | | |
| Emergence | | | | | |
| Tillering | | | | | |
| Stem elongation | | | | | |
| Booting/Heading | | | | | |
| Flowering | | | | | |
| Grain filling | | | | | |
| Pre-harvest | | | | | |

---

## 11) Experiments & Ablations (These become your “research contribution”)

Run these comparisons (as many as you can):
1. Weather-only vs Weather+Soil vs Weather+Soil+Imagery
2. Without NDVI vs With NDVI
3. Without weed correction vs With crop-only indices
4. Tabular baseline vs Temporal DL
5. Continuous inputs vs Continuous + input-quintile features

For each ablation, fill a short summary:
- What changed: `[________]`
- What improved/worsened: `[________]`
- Why you think so: `[________]`

---

## 12) Explainability (Simple but strong)

### 12.1 Feature importance / SHAP (for baseline models)
For each stage, report top drivers:
- Top 5 features at stage: `[list]`
- Interpretation (plain language): `[________]`

---

## 13) Discussion

### 13.1 What worked
- `[________]`

### 13.2 What didn’t work
- `[________]`

### 13.3 Limitations (be honest)
Common limitations:
- small number of seasons/fields
- cloud gaps
- inconsistent farm logs
- noisy yield labels

---

## 14) Conclusion & Future Work

### 14.1 Conclusion (3–5 bullets)
- `[________]`

### 14.2 Future work (practical next steps)
- Multi-farm scaling
- More seasons (the biggest improvement)
- Better weed labels
- Uncertainty quantification (prediction intervals)

---

## Appendix A: What to Attach
- Field boundary map
- Soil report scan
- Drone flight log (dates, altitude, sensor)
- Satellite extraction summary
- Farm visit log sheets
- Yield measurement photos/notes
