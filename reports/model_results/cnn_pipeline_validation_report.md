# CNN Pipeline Validation Report

Pipeline status: PARTIAL

## Models trained
- xception: model_path=/home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/xception_best.keras, best_validation_accuracy=0.9948849081993103, best_validation_loss=0.02604047767817974
- densenet121: model_path=/home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/densenet121_best.keras, best_validation_accuracy=0.9859334826469421, best_validation_loss=0.033678941428661346
- vgg16: model_path=/home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/vgg16_best.keras, best_validation_accuracy=0.9820972084999084, best_validation_loss=0.06974948942661285
- inceptionv3: model_path=/home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/inceptionv3_best.keras, best_validation_accuracy=0.9616368412971497, best_validation_loss=0.10028336942195892
- mobilenetv2: model_path=/home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/mobilenetv2_best.keras, best_validation_accuracy=0.9130434989929199, best_validation_loss=0.28503984212875366

Image count: 5211
Stage count: 5

## Stage distribution
- 1_Tillering: 1037
- 2_Jointing: 1202
- 3_BH: 1192
- 4_Flowering: 1164
- 5_Filling: 616

Best model: xception
Validation accuracy: 0.9948849081993103
Extracted feature count: 5210
Stage feature rows: None

## Per-model feature files
- densenet121_features.csv: 5210
- inceptionv3_features.csv: 5210
- mobilenetv2_features.csv: 5210
- vgg16_features.csv: 5210
- xception_features.csv: 5210

## Generated files
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/xception_best.keras: exists
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/densenet121_best.keras: exists
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/vgg16_best.keras: exists
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/inceptionv3_best.keras: exists
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/models/cnn/mobilenetv2_best.keras: exists
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/reports/model_results/cnn_evaluation_metrics.csv: exists
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/reports/model_results/cnn_model_comparison.csv: exists
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/data/processed/cnn_features.csv: exists
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/data/processed/cnn_stage_features.csv: missing

## Missing files
- /home/student/Crop_Yield_Prediction/Crop_yeild_predictions-20260603T090933Z-3-001/Crop_yeild_predictions/data/processed/cnn_stage_features.csv
