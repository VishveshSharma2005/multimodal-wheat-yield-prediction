# Data Sources Report

Fill or append to this report as data is ingested.

| Script | Dataset | Source URL | Region | Years | Purpose | Output File | Auto/Manual | Limitations |
|---|---|---|---|---|---|---|---|---|
| src/data/download_apy.py | Gujarat APY wheat yield | https://dag.gujarat.gov.in/Home/AreaProductionAndYield | Gujarat (Mehsana focus) | 2021-2024 | Yield ground truth | data/interim/apy_yield.csv | Manual | Convert source files to CSV before ingest |
| src/data/extract_apy_all_districts.py | Gujarat APY wheat yield (all districts) | https://dag.gujarat.gov.in/Home/AreaProductionAndYield | Gujarat (all districts) | 2020-21 to 2023-24 | Yield ground truth | data/interim/apy_yield_all_districts.csv | Manual | PDF table layouts can vary; verify extraction |
| src/data/download_weather.py | Open-Meteo Archive | https://archive-api.open-meteo.com/v1/archive | Mehsana | 2021-2025 | Daily weather | data/interim/weather_daily.csv | Auto | Dependent on coordinates |
| src/data/ingest_satellite.py | Sentinel-2 time series | Local/GEE export | Mehsana | 2021-2025 | Vegetation indices | data/interim/satellite_observations.csv | Manual | Requires valid cloud masking |
| src/data/ingest_images.py | Auxiliary image datasets | Local/Kaggle/Zenodo/etc | Global | Any | Stage/weed support | data/interim/image_metadata.csv | Manual | Optional support only |
