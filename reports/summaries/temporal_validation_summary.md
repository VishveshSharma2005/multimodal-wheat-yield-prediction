# Temporal (walk-forward / leave-one-year-out) validation

> **This is an additional robustness experiment. It does not replace the
> authoritative 2023-24 hold-out result** in
> `reports/model_results/stagewise_regression_metrics.csv`. The two answer
> different questions and their numbers are not interchangeable.

## Protocol

Strictly forward-chaining: for a held-out season, the model sees only
seasons that precede it. No future information is used at any point.

| Fold | Train seasons | Validate |
|---|---|---|
| 2021 | 2020 | 2021 |
| 2022 | 2020,2021 | 2022 |
| 2023 | 2020,2021,2022 | 2023 |

2024-25 is never a fold: its district-level APY target has not been published.

Feature set: 32 numeric + 4 categorical (district, stage, crop_name, season_type), pooled across all 9 stages, using the shared
exclusion policy in `src/features/feature_policy.py`.

## Aggregate across folds (mean +/- sd)

| Kind | Name | Folds | MAE | RMSE | MAPE % | R2 |
|---|---|---|---|---|---|---|
| baseline | Baseline A: district historical mean | 3 | 113.61 ± 38.92 | 155.25 ± 56.39 | 3.62 ± 1.21 | 0.96 ± 0.03 |
| baseline | Baseline B: persistence (last known year) | 3 | 111.96 ± 32.84 | 158.96 ± 46.66 | 3.63 ± 0.97 | 0.96 ± 0.02 |
| baseline | Baseline C: district identity only | 3 | 158.74 ± 33.79 | 189.66 ± 41.72 | 5.02 ± 1.02 | 0.94 ± 0.02 |
| model | LightGBMRegressor | 3 | 124.51 ± 25.99 | 169.50 ± 33.02 | 4.00 ± 0.79 | 0.95 ± 0.02 |
| model | RandomForestRegressor | 3 | 156.70 ± 69.17 | 216.70 ± 90.72 | 4.98 ± 2.07 | 0.91 ± 0.07 |
| model | GradientBoostingRegressor | 3 | 173.37 ± 70.89 | 222.11 ± 98.38 | 5.46 ± 2.09 | 0.91 ± 0.07 |
| model | XGBoostRegressor | 3 | 190.28 ± 117.69 | 267.97 ± 175.01 | 6.05 ± 3.46 | 0.85 ± 0.17 |
| model | HistGradientBoostingRegressor | 3 | 342.99 ± 237.30 | 503.19 ± 343.15 | 9.94 ± 6.43 | 0.46 ± 0.52 |

## Per-fold detail

| Validate | Kind | Name | n | MAE | RMSE | MAPE % | R2 |
|---|---|---|---|---|---|---|---|
| 2021 | baseline | Baseline A: district historical mean | 297 | 124.93 | 181.32 | 3.92 | 0.95 |
| 2021 | baseline | Baseline B: persistence (last known year) | 297 | 124.93 | 181.32 | 3.92 | 0.95 |
| 2021 | baseline | Baseline C: district identity only | 297 | 177.77 | 211.27 | 5.52 | 0.93 |
| 2021 | model | LightGBMRegressor | 297 | 129.86 | 176.58 | 4.09 | 0.95 |
| 2021 | model | RandomForestRegressor | 297 | 233.47 | 317.46 | 7.28 | 0.84 |
| 2021 | model | GradientBoostingRegressor | 297 | 246.10 | 327.64 | 7.55 | 0.83 |
| 2021 | model | XGBoostRegressor | 297 | 323.55 | 467.62 | 9.95 | 0.66 |
| 2021 | model | HistGradientBoostingRegressor | 297 | 577.96 | 825.55 | 16.30 | -0.07 |
| 2022 | baseline | Baseline B: persistence (last known year) | 297 | 136.33 | 190.23 | 4.42 | 0.94 |
| 2022 | baseline | Baseline A: district historical mean | 297 | 145.61 | 193.90 | 4.64 | 0.94 |
| 2022 | baseline | Baseline C: district identity only | 297 | 178.72 | 216.14 | 5.70 | 0.92 |
| 2022 | model | RandomForestRegressor | 297 | 137.42 | 191.12 | 4.39 | 0.94 |
| 2022 | model | XGBoostRegressor | 297 | 146.70 | 195.18 | 4.83 | 0.93 |
| 2022 | model | LightGBMRegressor | 297 | 147.41 | 198.41 | 4.75 | 0.93 |
| 2022 | model | GradientBoostingRegressor | 297 | 169.54 | 205.74 | 5.45 | 0.93 |
| 2022 | model | HistGradientBoostingRegressor | 297 | 347.58 | 541.53 | 10.10 | 0.50 |
| 2023 | baseline | Baseline A: district historical mean | 297 | 70.29 | 90.54 | 2.28 | 0.99 |
| 2023 | baseline | Baseline B: persistence (last known year) | 297 | 74.61 | 105.32 | 2.55 | 0.98 |
| 2023 | baseline | Baseline C: district identity only | 297 | 119.72 | 141.57 | 3.85 | 0.97 |
| 2023 | model | GradientBoostingRegressor | 297 | 104.47 | 132.94 | 3.38 | 0.97 |
| 2023 | model | LightGBMRegressor | 297 | 96.26 | 133.52 | 3.17 | 0.97 |
| 2023 | model | XGBoostRegressor | 297 | 100.60 | 141.10 | 3.36 | 0.97 |
| 2023 | model | RandomForestRegressor | 297 | 99.22 | 141.52 | 3.26 | 0.97 |
| 2023 | model | HistGradientBoostingRegressor | 297 | 103.43 | 142.48 | 3.43 | 0.97 |

## Statistical caution

- Only **3 folds**. A standard deviation over 3 values is a weak estimate of
  spread and no significance test is meaningful at this sample size.
- Fold 2021 trains on a **single** season (33 cases, 297 stage rows). Its
  metrics are the least reliable and dominate the spread.
- Fold results are not independent: later folds contain all earlier training data.
- No claim of statistical significance between models is made or supported.
