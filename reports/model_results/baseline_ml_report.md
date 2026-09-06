# Baseline ML Report

- Dataset file: data\processed\stage_features.csv
- Train/validation split: train=2020-2022, validation=2023, prediction_2025=2024
- Training rows: 0
- Validation rows: 0
- Stages trained: none
- Models tried: GradientBoostingRegressor, HistGradientBoostingRegressor, LightGBMRegressor, RandomForestRegressor, XGBoostRegressor

## Best model per stage
- booting_heading: GradientBoostingRegressor
- early_vegetative: GradientBoostingRegressor
- flowering: GradientBoostingRegressor
- grain_filling: GradientBoostingRegressor
- grain_filling_initial: GradientBoostingRegressor
- maturity_preharvest: GradientBoostingRegressor
- sowing: GradientBoostingRegressor
- stem_elongation: GradientBoostingRegressor
- tillering: GradientBoostingRegressor

## Stage-wise regression metrics
| stage            | model                         |   n_train |   n_val |     mae |      mse |    rmse |     mape |       r2 | status   |   message |
|:-----------------|:------------------------------|----------:|--------:|--------:|---------:|--------:|---------:|---------:|:---------|----------:|
| booting_heading  | RandomForestRegressor         |        99 |      33 | 220.025 |  69645.1 | 263.904 |  7.45974 | 0.879999 | ok       |       nan |
| booting_heading  | GradientBoostingRegressor     |        99 |      33 | 161.15  |  35337.9 | 187.984 |  5.49355 | 0.939111 | ok       |       nan |
| booting_heading  | HistGradientBoostingRegressor |        99 |      33 | 456.982 | 311212   | 557.863 | 14.2408  | 0.46377  | ok       |       nan |
| booting_heading  | LightGBMRegressor             |        99 |      33 | 421.694 | 342633   | 585.349 | 13.5338  | 0.409629 | ok       |       nan |
| booting_heading  | XGBoostRegressor              |        99 |      33 | 194.077 |  77551.4 | 278.481 |  6.65859 | 0.866376 | ok       |       nan |
| early_vegetative | RandomForestRegressor         |        99 |      33 | 186.595 |  53844.9 | 232.045 |  5.98775 | 0.907223 | ok       |       nan |
| early_vegetative | GradientBoostingRegressor     |        99 |      33 | 154.992 |  38219.1 | 195.497 |  5.06738 | 0.934147 | ok       |       nan |
| early_vegetative | HistGradientBoostingRegressor |        99 |      33 | 451.047 | 396974   | 630.058 | 14.2027  | 0.315999 | ok       |       nan |
| early_vegetative | LightGBMRegressor             |        99 |      33 | 400.66  | 291170   | 539.602 | 14.2682  | 0.498302 | ok       |       nan |
| early_vegetative | XGBoostRegressor              |        99 |      33 | 173.162 |  53588.1 | 231.491 |  6.17128 | 0.907666 | ok       |       nan |
| flowering        | RandomForestRegressor         |        99 |      33 | 207.317 |  68678.3 | 262.065 |  7.12219 | 0.881665 | ok       |       nan |
| flowering        | GradientBoostingRegressor     |        99 |      33 | 149.822 |  40046.6 | 200.117 |  4.97626 | 0.930998 | ok       |       nan |
| flowering        | HistGradientBoostingRegressor |        99 |      33 | 396.752 | 289507   | 538.058 | 11.7687  | 0.501168 | ok       |       nan |
| flowering        | LightGBMRegressor             |        99 |      33 | 401.308 | 274134   | 523.578 | 13.0676  | 0.527656 | ok       |       nan |
| flowering        | XGBoostRegressor              |        99 |      33 | 217.4   |  98328.8 | 313.574 |  7.761   | 0.830576 | ok       |       nan |
| grain_filling    | RandomForestRegressor         |        99 |      33 | 203.789 |  71711.7 | 267.79  |  7.00822 | 0.876438 | ok       |       nan |
| grain_filling    | GradientBoostingRegressor     |        99 |      33 | 128.286 |  31729.7 | 178.128 |  4.07231 | 0.945329 | ok       |       nan |
| grain_filling    | HistGradientBoostingRegressor |        99 |      33 | 460.49  | 436143   | 660.411 | 13.257   | 0.248509 | ok       |       nan |
| grain_filling    | LightGBMRegressor             |        99 |      33 | 436.038 | 351058   | 592.502 | 13.8395  | 0.395113 | ok       |       nan |
| grain_filling    | XGBoostRegressor              |        99 |      33 | 196.427 |  73348.6 | 270.829 |  7.01109 | 0.873618 | ok       |       nan |

## Combined regression metrics
| model                         |   n_train |   n_val |      mae |     mse |    rmse |    mape |       r2 | status   |   message |
|:------------------------------|----------:|--------:|---------:|--------:|--------:|--------:|---------:|:---------|----------:|
| RandomForestRegressor         |       891 |     297 |  99.2165 | 20029   | 141.524 | 3.26386 | 0.965489 | ok       |       nan |
| GradientBoostingRegressor     |       891 |     297 | 104.471  | 17672.3 | 132.937 | 3.37625 | 0.96955  | ok       |       nan |
| HistGradientBoostingRegressor |       891 |     297 | 103.426  | 20300.2 | 142.479 | 3.43259 | 0.965022 | ok       |       nan |
| LightGBMRegressor             |       891 |     297 |  96.2553 | 17827.1 | 133.518 | 3.16677 | 0.969283 | ok       |       nan |
| XGBoostRegressor              |       891 |     297 | 100.601  | 19910   | 141.103 | 3.35561 | 0.965694 | ok       |       nan |

## Quintile classification metrics
| stage                 | model                      |   n_train |   n_val |   accuracy |   macro_f1 | status   |   message |
|:----------------------|:---------------------------|----------:|--------:|-----------:|-----------:|:---------|----------:|
| booting_heading       | RandomForestClassifier     |        99 |      33 |   0.757576 |   0.747949 | ok       |       nan |
| booting_heading       | GradientBoostingClassifier |        99 |      33 |   0.727273 |   0.704823 | ok       |       nan |
| early_vegetative      | RandomForestClassifier     |        99 |      33 |   0.727273 |   0.73098  | ok       |       nan |
| early_vegetative      | GradientBoostingClassifier |        99 |      33 |   0.818182 |   0.817542 | ok       |       nan |
| flowering             | RandomForestClassifier     |        99 |      33 |   0.757576 |   0.730745 | ok       |       nan |
| flowering             | GradientBoostingClassifier |        99 |      33 |   0.727273 |   0.712424 | ok       |       nan |
| grain_filling         | RandomForestClassifier     |        99 |      33 |   0.69697  |   0.677647 | ok       |       nan |
| grain_filling         | GradientBoostingClassifier |        99 |      33 |   0.727273 |   0.722403 | ok       |       nan |
| grain_filling_initial | RandomForestClassifier     |        99 |      33 |   0.606061 |   0.590043 | ok       |       nan |
| grain_filling_initial | GradientBoostingClassifier |        99 |      33 |   0.636364 |   0.611966 | ok       |       nan |
| maturity_preharvest   | RandomForestClassifier     |        99 |      33 |   0.69697  |   0.689675 | ok       |       nan |
| maturity_preharvest   | GradientBoostingClassifier |        99 |      33 |   0.69697  |   0.678264 | ok       |       nan |
| sowing                | RandomForestClassifier     |        99 |      33 |   0.424242 |   0.415296 | ok       |       nan |
| sowing                | GradientBoostingClassifier |        99 |      33 |   0.666667 |   0.66704  | ok       |       nan |
| stem_elongation       | RandomForestClassifier     |        99 |      33 |   0.787879 |   0.787473 | ok       |       nan |
| stem_elongation       | GradientBoostingClassifier |        99 |      33 |   0.69697  |   0.687857 | ok       |       nan |
| tillering             | RandomForestClassifier     |        99 |      33 |   0.69697  |   0.67913  | ok       |       nan |
| tillering             | GradientBoostingClassifier |        99 |      33 |   0.787879 |   0.78355  | ok       |       nan |

## Limitations
- Baselines do not model temporal sequences across stages.
- Some stages may be skipped if validation data is missing.
- Optional models (LightGBM/XGBoost) run only if installed.

## Next step
- Train LSTM temporal sequence model.
