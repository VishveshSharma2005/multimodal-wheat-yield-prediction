# Within-district (anomaly) skill

Season: **2023-24 validation**, 297 stage-level predictions (33 districts x 9 stages), stage-wise GradientBoosting models.

## What this measures

District yields differ systematically and persistently, and every district
appears in both the training and the validation split. Ordinary R2 therefore
cannot tell "knows the agronomy of this season" apart from "knows which
district this is". Removing each district's **training-season** mean from both
the observed and the predicted value isolates the season-to-season signal.

The centring uses training seasons only (2020-21..2022-23), so no
validation-year information enters it.

## Results

### Raw yield space

| Predictor | n | MAE | RMSE | R2 |
|---|---|---|---|---|
| Stage-wise model | 297 | 162.17 | 210.99 | 0.9233 |
| District historical mean (reference) | 297 | 70.29 | 90.54 | 0.9859 |

### Within-district anomaly space

| Predictor | n | MAE | RMSE | R2 |
|---|---|---|---|---|
| Stage-wise model | 297 | 162.17 | 210.99 | -4.4342 |
| Zero anomaly (= district mean) | 297 | 70.29 | 90.54 | -0.0007 |

Observed anomaly standard deviation: **90.51 kg/ha**.

### Per stage (anomaly space)

| Stage | n | MAE | RMSE | R2 |
|---|---|---|---|---|
| sowing | 33 | 159.74 | 220.50 | -4.9349 |
| early_vegetative | 33 | 154.99 | 195.50 | -3.6654 |
| tillering | 33 | 169.54 | 212.96 | -4.5359 |
| stem_elongation | 33 | 164.50 | 200.69 | -3.9166 |
| booting_heading | 33 | 161.15 | 187.98 | -3.3137 |
| flowering | 33 | 149.82 | 200.12 | -3.8884 |
| grain_filling_initial | 33 | 190.01 | 260.30 | -7.2710 |
| grain_filling | 33 | 128.29 | 178.13 | -2.8732 |
| maturity_preharvest | 33 | 181.49 | 230.91 | -5.5084 |

## Interpretation

- Anomaly R2 = **-4.4342 <= 0**: once each district's historical mean is
  removed, the model explains **none** of the remaining season-to-season
  variation. Its headline R2 in raw-yield space is carried by district identity,
  not by the weather / satellite / calendar features.
- For comparison, simply predicting each district's historical mean scores
  RMSE 90.54 in raw space against the model's 210.99.

## Limits of this analysis

- 33 districts x 1 validation season = 33 independent case-level anomalies; the 297
  stage rows are not independent (9 rows share one target per case).
- This is a predictive-skill decomposition, **not** a causal analysis. It does not
  say the environmental variables are irrelevant to yield, only that this model on
  this data does not extract usable season-level signal from them.
