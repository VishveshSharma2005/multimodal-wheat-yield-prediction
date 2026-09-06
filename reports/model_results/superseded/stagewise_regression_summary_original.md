# Stage-wise Regression Summary

- Dataset: /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/data/processed/stage_features_with_cnn.csv
- Total rows: 1485
- Stages found: booting_heading, early_vegetative, flowering, grain_filling, grain_filling_initial, maturity_preharvest, sowing, stem_elongation, tillering

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

## Feature groups used
- booting_heading: crop/meta, location, other, satellite, stage/time, weather
- early_vegetative: crop/meta, image, location, other, satellite, stage/time, weather
- flowering: crop/meta, image, location, other, satellite, stage/time, weather
- grain_filling: crop/meta, image, location, other, satellite, stage/time, weather
- grain_filling_initial: crop/meta, image, location, other, satellite, stage/time, weather
- maturity_preharvest: crop/meta, image, location, other, satellite, stage/time, weather
- sowing: crop/meta, location, other, satellite, stage/time, weather
- stem_elongation: crop/meta, location, other, satellite, stage/time, weather
- tillering: crop/meta, location, other, satellite, stage/time, weather
