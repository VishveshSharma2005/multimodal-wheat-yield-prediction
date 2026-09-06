# Stage-wise Improvement Report

## Validation metrics by stage (2023-24)
                stage  stage_end_das        mae       rmse     mape       r2
               sowing           10.0 159.742539 220.497943 5.587283 0.916227
     early_vegetative           20.0 154.992364 195.497018 5.067375 0.934147
            tillering           40.0 169.544212 212.957894 5.928419 0.921858
      stem_elongation           60.0 164.504061 200.691080 5.695951 0.930601
      booting_heading           80.0 161.149960 187.983762 5.493547 0.939111
            flowering           95.0 149.821914 200.116541 4.976259 0.930998
grain_filling_initial          105.0 190.006556 260.301480 6.433120 0.883252
        grain_filling          115.0 128.286163 178.128324 4.072308 0.945329
  maturity_preharvest          130.0 181.489323 230.906478 6.026178 0.908131

- RMSE trend from early to late stages: worsens.
- Best stage by RMSE: grain_filling (178.13).
- Worst stage by RMSE: grain_filling_initial (260.30).
- Stage-wise improvement is expected but not guaranteed if late-stage satellite/drone features are missing.
