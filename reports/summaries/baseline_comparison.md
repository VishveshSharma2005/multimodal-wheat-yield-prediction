# Baseline comparison

Do the multi-modal models learn anything beyond district identity and
historical yield? All baselines are evaluated on the **same** forward-chaining
folds as the models, and see only seasons before the held-out year.

| Baseline | Definition |
|---|---|
| A - district historical mean | Predict each district's mean yield over the training seasons. |
| B - persistence | Predict the district's most recent known yield before the held-out season. |
| C - district identity only | GradientBoosting on district + stage only: no weather, satellite or image features. |

## Results (mean +/- sd over folds)

| Kind | Name | MAE | RMSE | MAPE % | R2 |
|---|---|---|---|---|---|
| baseline | Baseline A: district historical mean | 113.61 ± 38.92 | 155.25 ± 56.39 | 3.62 ± 1.21 | 0.96 ± 0.03 |
| baseline | Baseline B: persistence (last known year) | 111.96 ± 32.84 | 158.96 ± 46.66 | 3.63 ± 0.97 | 0.96 ± 0.02 |
| baseline | Baseline C: district identity only | 158.74 ± 33.79 | 189.66 ± 41.72 | 5.02 ± 1.02 | 0.94 ± 0.02 |
| model | LightGBMRegressor | 124.51 ± 25.99 | 169.50 ± 33.02 | 4.00 ± 0.79 | 0.95 ± 0.02 |
| model | RandomForestRegressor | 156.70 ± 69.17 | 216.70 ± 90.72 | 4.98 ± 2.07 | 0.91 ± 0.07 |
| model | GradientBoostingRegressor | 173.37 ± 70.89 | 222.11 ± 98.38 | 5.46 ± 2.09 | 0.91 ± 0.07 |
| model | XGBoostRegressor | 190.28 ± 117.69 | 267.97 ± 175.01 | 6.05 ± 3.46 | 0.85 ± 0.17 |
| model | HistGradientBoostingRegressor | 342.99 ± 237.30 | 503.19 ± 343.15 | 9.94 ± 6.43 | 0.46 ± 0.52 |

## Interpretation

- Best model: **LightGBMRegressor**, RMSE 169.50 ± 33.02.
- Best baseline: **Baseline A: district historical mean**, RMSE 155.25 ± 56.39.
- **The best model does not beat the best baseline on mean RMSE.** This must be
  reported as-is: on this data and protocol, the multi-modal features do not
  demonstrably improve on a naive district-level predictor.

- Baseline C isolates district identity. The gap between Baseline C and the full
  models is the part of performance attributable to weather, satellite and calendar
  features rather than to knowing which district a row belongs to.
