# 2024-25 wheat yield — FORECAST ONLY

> **No accuracy metric exists for this season and none may be computed.**
> District-level Area-Production-Yield statistics for Gujarat wheat 2024-25 have
> not been published. Every value on this page is a model output, never an
> observation.

---

## 1. Ground-truth status

| | |
|---|---|
| District-level 2024-25 APY | **not available** |
| `yield_kg_ha` for `season_year_start == 2024` | **null**, all 297 rows |
| `actual_available` | 0, all 297 rows |
| `evaluation_status` | `pending_actual_apy`, all 297 rows |
| MAPE / RMSE / MAE / R² for 2024-25 | **N/A — not computable** |

The only 2024-25 figure in the repository is the Gujarat **state-level** Final
Advance Estimate of **3280.69 kg/ha**. It is a state aggregate and must never be
assigned to Mehsana or to any other district. Using it as a district target would
be fabricating ground truth.

A defect that previously carried Mehsana's 2023-24 yield (2796.06 kg/ha) forward
into the 2024-25 rows — producing an invalid "Mehsana 2025 MAPE 11.93" — was
removed on 2026-09-05. `scripts/smoke_test.py` now fails if any 2024-25 row ever
carries a target again. Full record:
[`docs/2025_yield_reference_methodology.md`](../../docs/2025_yield_reference_methodology.md).

## 2. Outputs

| File | Rows | Content |
|---|---|---|
| `data/processed/prediction_2025_stagewise_predictions.csv` | 297 | 33 districts × 9 stages, stage-wise GradientBoosting |
| `data/processed/mehsana_2025_stagewise_predictions.csv` | 9 | Mehsana view of the same forecast |
| `data/processed/lstm_predictions_2025.csv` | 33 | One case-level LSTM forecast per district |

All-district stage-wise forecast: mean 3167.7 kg/ha, range 310.7 – 4368.7 kg/ha.

## 3. Mehsana 2024-25 stage-wise forecast

**This is a forecast trajectory, not an accuracy curve.**

| Stage | DAS | Forecast yield (kg/ha) |
|---|---|---|
| sowing | 10 | 3186.4 |
| early_vegetative | 20 | 3080.4 |
| tillering | 40 | 3019.4 |
| stem_elongation | 60 | 3019.7 |
| booting_heading | 80 | 3067.0 |
| flowering | 95 | 3095.4 |
| grain_filling_initial | 105 | 3304.0 |
| grain_filling | 115 | 3139.3 |
| maturity_preharvest | 130 | 3205.2 |

LSTM case-level forecast for Mehsana: **3264.6 kg/ha**.

Figure: `reports/figures/mehsana_2025_stagewise_prediction.png` (forecast line
only — there is no "actual" line to draw).

### Plausibility context — not an error metric

Mehsana observed yields: 3389.73 (2020-21), 3292.80 (2021-22), 3169.33 (2022-23),
3255.21 (2023-24) kg/ha. The forecast trajectory sits inside that historical band,
which is a sanity check only. It says nothing about accuracy.

## 4. Limitation: 288 of 297 rows have imputed vegetation indices

Sentinel-2 exports for the 2024-25 season were produced for **Mehsana only**.

| | Rows with real NDVI / NDRE / EVI |
|---|---|
| Mehsana | 9 / 9 |
| Other 32 districts | **0 / 288** |

For those 288 rows the vegetation indices are filled by the pipeline's median
imputer using training-period values, so the forecast is driven almost entirely
by weather, location and calendar features. `satellite_available_flag` in the
prediction CSVs records this per row (1 for 9 rows, 0 for 288).

**Consequence:** only the 9 Mehsana rows are genuinely multi-modal forecasts. The
33-district forecast table should not be presented as 33 equally-informed
predictions.

## 5. How much confidence is warranted

Read this together with
[`research_ready_results.md`](research_ready_results.md):

* Under forward-chaining temporal validation no model beat a naive
  district-historical-mean baseline.
* The stage-wise model has negative within-district skill (anomaly R² = −4.43),
  meaning it does not predict season-to-season deviation from a district's norm.

A 2024-25 forecast from such a model is therefore best understood as **close to a
district climatology estimate**, not as an environmentally-informed in-season
forecast. Present it accordingly.

## 6. Wording to use

> Model forecasts for the 2024-25 wheat season are reported for 33 Gujarat
> districts across 9 growth stages. District-level Area-Production-Yield
> statistics for 2024-25 had not been released at the time of writing, so no
> forecast error can be computed for that season. Vegetation-index inputs were
> available for Mehsana only; the remaining districts use imputed values.

**Do not write** "2024-25 accuracy", "Mehsana 2025 MAPE", or "actual 2024-25
yield".
