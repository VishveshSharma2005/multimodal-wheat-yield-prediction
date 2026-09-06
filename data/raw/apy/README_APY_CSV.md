# APY CSVs created from uploaded Gujarat PDFs

Files created:

1. `apy_mehsana_wheat_district_2021_2024.csv`
   - Recommended main APY CSV for the project.
   - Contains Mehsana district Total Wheat APY values for 2021-22, 2022-23, 2023-24.
   - Location: `data/raw/apy/`

2. `apy_wheat_trace.csv`
   - Trace/reference CSV (not used for modeling).
   - Contains recommended Mehsana rows plus older 20-23 rows and Gujarat 2024-25 state-level wheat row.
   - Location: `data/raw/apy/`

3. `field_registry_mehsana_from_apy.csv`
   - Optional starter field registry (district-level yield as target).
   - Replace lat/lon/area_ha with real field details before final modeling.
   - Location: `data/raw/apy/`

Important notes:
- 2024-25 uploaded PDF is state-level Final Advance Estimates, not district-level Mehsana APY.
- Therefore, 2024-25 is included only in the trace CSV as a Gujarat-state benchmark.
- For true 2025 prediction/evaluation at Mehsana level, you need district-level 2024-25/2025 APY when released or real field harvest data.
- 2025 must not be used for training.
