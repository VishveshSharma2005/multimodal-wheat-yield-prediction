# data/interim

Cleaned, standardised tables — "clean but not yet engineered".

| File | Rows | In Git? | Produced by |
|---|---|---|---|
| `apy_yield_all_districts.csv` | 132 | yes | `src/data/extract_apy_all_districts_xlsx.py` — **authoritative yield target** (33 districts x 2020-21..2023-24) |
| `apy_yield.csv` | 3 | yes | `src/data/download_apy.py` — Mehsana subset |
| `apy_2025_ground_truth.csv` | 0 (header only) | yes | `src/data/ingest_apy_2025_ground_truth.py` — empty because no official district-level 2024-25 APY exists; see `../README.md` section 5 |
| `cases.csv` | 165 | yes | `src/data/build_stage_table.py` — the 33 districts x 5 seasons |
| `coordinate_registry.csv` | — | yes | `src/data/build_coordinate_registry.py` — 14 of 33 districts resolved |
| `image_metadata.csv` | — | yes | `src/data/ingest_images.py` |
| `cnn_dataset_split.csv` | 5,211 | no (422 KB) | any `src/deep_learning/train_*.py` |
| `satellite_observations.csv` | 6,779 | no (1.7 MB) | `src/data/ingest_satellite.py` |
| `weather_daily.csv` | 20,856 | no (2.4 MB) | `src/data/download_weather.py` |
| `weather_daily_points.csv` | — | no (3.1 MB) | `src/data/download_weather.py` |
