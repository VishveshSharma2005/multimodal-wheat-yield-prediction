# Training Data Verification

- Number of districts: 33
- Districts: Ahmedabad, Amreli, Anand, Arvalli, Banaskantha, Bharuch, Bhavnagar, Botad, Chhota Udepur, Dahod, Dangs, Devbhoomi Dwarka, Gandhinagar, Gir Somnath, Jamnagar, Junagadh, Kachchh, Kheda, Mahisagar, Mehsana, Morbi, Narmada, Navsari, Panchmahal, Patan, Porbandar, Rajkot, Sabarkantha, Surat, Surendranagar, Tapi, Vadodara, Valsad

- Years available: 2020, 2021, 2022, 2023, 2024
- Number of cases: 165
- Number of stage rows: 1485
- Number of crop stages: 9
- Stages: booting_heading, early_vegetative, flowering, grain_filling, grain_filling_initial, maturity_preharvest, sowing, stem_elongation, tillering

- Mehsana exists: yes
- Mehsana years: 2020, 2021, 2022, 2023, 2024

- Missing yield_kg_ha count: 32

## Missing values by feature group
- Weather: 0 (0.00%)
- Satellite: 3168 (19.39%)
- Image: 7425 (50.00%)
- Weed/Crop concentration: 0 (0.00%)

## Year checks
- 2020-21 present: yes
- 2021-22 present: yes
- 2022-23 present: yes
- 2023-24 present: yes
- 2024-25 present: yes

## Split checks
- Split counts: {'train': 891, 'validation': 297, 'prediction_2025': 297}
- Rows with split=prediction_2025: 297
- season_year_start==2024 exists: yes
- targets season_year==2024 rows: 33

## Coverage summary
- Weather columns present: rain_sum, tmin_mean, tmax_mean, cumulative_rain_mm, mean_tmin_c, mean_tmax_c, max_tmax_c, gdd_cum, gdd_cumulative, gdd_norm, cumulative_gdd
- Satellite columns present: ndvi_last, ndvi_mean, ndvi_max, ndvi_slope, ndre_last, ndre_mean, ndre_slope, evi_last, evi_mean, evi_slope, ndvi_growth_rate
- Image columns present: image_green_pixel_ratio_stage_avg, image_canopy_density_proxy_stage_avg, image_vegetation_pixel_ratio_stage_avg, image_brightness_mean_stage_avg, image_saturation_mean_stage_avg, image_texture_proxy_stage_avg, image_count_stage, image_coverage_flag, image_feature_source, cnn_image_count
## Case-stage checks
- Cases with <2 stages: 0

## Leakage check
- Leakage columns in stage_features: yield_kg_ha
