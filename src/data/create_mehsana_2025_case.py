from pathlib import Path
import sys
from datetime import datetime

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def fetch_weather(lat, lon, start_date, end_date, api_settings):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(api_settings["variables"]),
        "timezone": api_settings["timezone"],
    }
    resp = requests.get(api_settings["base_url"], params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    ensure_dir(interim_dir)

    cases_path = interim_dir / "cases.csv"
    if not cases_path.exists():
        raise FileNotFoundError("Missing cases.csv. Run preprocess_all.py first.")

    cases = pd.read_csv(cases_path)
    case_id = "GUJ_MEHSANA_2024_2025"

    exists = (cases["case_id"] == case_id).any()
    if not exists:
        new_row = {
            "case_id": case_id,
            "district": "Mehsana",
            "state": "Gujarat",
            "crop": "wheat",
            "region": "Gujarat",
            "season_year": 2024,
            "season_year_start": 2024,
            "season_year_end": 2025,
            "lat": 23.5880,
            "lon": 72.3693,
            "area_ha": None,
            "sowing_date": "2024-11-20",
            "harvest_date": "2025-03-25",
            "yield_kg_ha": None,
        }
        cases = pd.concat([cases, pd.DataFrame([new_row])], ignore_index=True)
        cases.to_csv(cases_path, index=False)
        logger.info("Added Mehsana 2024-2025 case to cases.csv")
    else:
        logger.info("Mehsana 2024-2025 case already exists in cases.csv")

    weather_path = interim_dir / "weather_daily.csv"
    weather_df = pd.read_csv(weather_path) if weather_path.exists() else pd.DataFrame()

    has_weather = False
    if not weather_df.empty and "case_id" in weather_df.columns:
        has_weather = case_id in weather_df["case_id"].unique()

    if not has_weather:
        api_settings = config["api_settings"]["weather"]
        try:
            data = fetch_weather(23.5880, 72.3693, "2024-11-20", "2025-03-25", api_settings)
        except Exception as exc:
            logger.warning("Weather fetch failed for Mehsana 2025 case: %s", exc)
            return

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        rows = []
        for i, day in enumerate(dates):
            rows.append(
                {
                    "case_id": case_id,
                    "date": day,
                    "precipitation_sum": daily.get("precipitation_sum", [None] * len(dates))[i],
                    "temperature_2m_min": daily.get("temperature_2m_min", [None] * len(dates))[i],
                    "temperature_2m_max": daily.get("temperature_2m_max", [None] * len(dates))[i],
                    "relative_humidity_2m_mean": daily.get("relative_humidity_2m_mean", [None] * len(dates))[i],
                    "shortwave_radiation_sum": daily.get("shortwave_radiation_sum", [None] * len(dates))[i],
                }
            )

        if rows:
            out_df = pd.concat([weather_df, pd.DataFrame(rows)], ignore_index=True)
            out_df.to_csv(weather_path, index=False)
            logger.info("Added weather rows for Mehsana 2024-2025 case")
        else:
            logger.warning("No weather rows fetched for Mehsana 2024-2025 case")
    else:
        logger.info("Weather data already present for Mehsana 2024-2025 case")

    sat_path = interim_dir / "satellite_observations.csv"
    if sat_path.exists():
        sat_df = pd.read_csv(sat_path)
        if "case_id" in sat_df.columns:
            if case_id not in sat_df["case_id"].unique():
                logger.warning("Satellite data missing for Mehsana 2024-2025 case")
    else:
        logger.warning("satellite_observations.csv not found; satellite features may be missing")


if __name__ == "__main__":
    main()
