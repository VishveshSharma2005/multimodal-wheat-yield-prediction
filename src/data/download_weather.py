from pathlib import Path
import sys

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


def _load_cases(interim_dir: Path) -> pd.DataFrame:
    cases_path = interim_dir / "cases.csv"
    if not cases_path.exists():
        raise FileNotFoundError("Missing cases.csv. Run preprocess_all.py first.")
    cases = pd.read_csv(cases_path)
    for col in ["sowing_date", "harvest_date"]:
        if col in cases.columns:
            cases[col] = pd.to_datetime(cases[col], errors="coerce")
    return cases


def _load_registry(interim_dir: Path) -> pd.DataFrame:
    registry_path = interim_dir / "coordinate_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError("Missing coordinate_registry.csv. Run build_coordinate_registry.py first.")
    registry = pd.read_csv(registry_path)
    return registry


def _split_label(year):
    if pd.isna(year):
        return "unknown"
    year = int(year)
    if year in {2020, 2021, 2022}:
        return "train"
    if year == 2023:
        return "validation"
    if year == 2024:
        return "prediction_2025"
    return "unknown"


def _write_coverage_report(
    report_path: Path,
    weather_df: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    cases = cases.copy()
    cases["split"] = cases["season_year_start"].apply(_split_label)
    weather_case_ids = set(weather_df["case_id"].dropna().unique().tolist()) if not weather_df.empty else set()

    lines = ["# Weather Coverage Report", ""]

    for split, group in cases.groupby("split"):
        case_ids = set(group["case_id"].dropna().unique().tolist())
        with_weather = case_ids.intersection(weather_case_ids)
        coverage_pct = (len(with_weather) / len(case_ids) * 100.0) if case_ids else 0.0
        lines.append(
            f"- {split}: {len(with_weather)}/{len(case_ids)} cases with weather ({coverage_pct:.2f}%)"
        )

    mehsana_cases = cases[cases["district"].str.contains("MEHSANA", case=False, na=False)]
    mehsana_case_ids = set(mehsana_cases["case_id"].dropna().unique().tolist())
    mehsana_weather = mehsana_case_ids.intersection(weather_case_ids)
    lines.append("")
    lines.append("## Mehsana coverage")
    if mehsana_case_ids:
        lines.append(f"- Cases with weather: {len(mehsana_weather)}/{len(mehsana_case_ids)}")
    else:
        lines.append("- No Mehsana cases found.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    ensure_dir(interim_dir)
    ensure_dir(reports_dir)

    cases = _load_cases(interim_dir)
    registry = _load_registry(interim_dir)
    if registry.empty:
        logger.warning("Coordinate registry is empty. Weather download skipped.")
        return

    merged = registry.merge(
        cases[["case_id", "district", "sowing_date", "harvest_date", "season_year_start", "season_year_end"]],
        on="case_id",
        how="left",
    )
    if "district" not in merged.columns:
        if "district_y" in merged.columns:
            merged["district"] = merged["district_y"]
        elif "district_x" in merged.columns:
            merged["district"] = merged["district_x"]

    api_settings = config["api_settings"]["weather"]
    provider = api_settings.get("provider", "open_meteo")

    points_path = interim_dir / "weather_daily_points.csv"
    existing_points = pd.read_csv(points_path) if points_path.exists() else pd.DataFrame()
    if not existing_points.empty:
        existing_points["date"] = pd.to_datetime(existing_points["date"], errors="coerce")

    weather_rows = []
    for _, row in merged.iterrows():
        case_id = row.get("case_id")
        field_id = row.get("field_id")
        district = row.get("district")
        lat = row.get("latitude")
        lon = row.get("longitude")
        sowing_date = row.get("sowing_date")
        harvest_date = row.get("harvest_date")

        if pd.isna(case_id) or pd.isna(field_id) or str(field_id).strip() == "":
            logger.warning("Missing case_id/field_id for registry row. Skipping.")
            continue
        if pd.isna(lat) or pd.isna(lon):
            logger.warning("Missing coordinates for %s (%s). Skipping.", case_id, field_id)
            continue
        if pd.isna(sowing_date) or pd.isna(harvest_date):
            logger.warning("Missing sowing/harvest dates for %s. Skipping.", case_id)
            continue

        start_date = str(pd.to_datetime(sowing_date).date())
        end_date = str(pd.to_datetime(harvest_date).date())

        if not existing_points.empty:
            existing = existing_points[
                (existing_points["case_id"] == case_id) & (existing_points["field_id"] == field_id)
            ]
            if not existing.empty:
                min_date = existing["date"].min()
                max_date = existing["date"].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    if min_date <= pd.to_datetime(start_date) and max_date >= pd.to_datetime(end_date):
                        continue

        try:
            data = fetch_weather(float(lat), float(lon), start_date, end_date, api_settings)
        except Exception as exc:
            logger.error("Weather fetch failed for %s/%s: %s", case_id, field_id, exc)
            continue

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        for i, day in enumerate(dates):
            rain_mm = daily.get("precipitation_sum", [None] * len(dates))[i]
            tmin_c = daily.get("temperature_2m_min", [None] * len(dates))[i]
            tmax_c = daily.get("temperature_2m_max", [None] * len(dates))[i]
            humidity_pct = daily.get("relative_humidity_2m_mean", [None] * len(dates))[i]
            solar_rad = daily.get("shortwave_radiation_sum", [None] * len(dates))[i]
            weather_rows.append(
                {
                    "case_id": case_id,
                    "field_id": field_id,
                    "district": district,
                    "date": day,
                    "latitude": lat,
                    "longitude": lon,
                    "rain_mm": rain_mm,
                    "tmin_c": tmin_c,
                    "tmax_c": tmax_c,
                    "humidity_pct": humidity_pct,
                    "solar_rad": solar_rad,
                    "precipitation_sum": rain_mm,
                    "temperature_2m_min": tmin_c,
                    "temperature_2m_max": tmax_c,
                    "relative_humidity_2m_mean": humidity_pct,
                    "shortwave_radiation_sum": solar_rad,
                    "source": provider,
                }
            )

    if not weather_rows and existing_points.empty:
        logger.warning("No weather rows produced.")
        return

    new_points = pd.DataFrame(weather_rows)
    if not existing_points.empty:
        points_df = pd.concat([existing_points, new_points], ignore_index=True)
    else:
        points_df = new_points.copy()
    if not points_df.empty:
        points_df = points_df.drop_duplicates(subset=["case_id", "field_id", "date"], keep="last")
    points_df.to_csv(points_path, index=False)
    logger.info("Wrote %s", points_path)

    if points_df.empty:
        logger.warning("weather_daily_points.csv is empty after merge.")
        return

    points_df["date"] = pd.to_datetime(points_df["date"], errors="coerce")
    if "district" in points_df.columns:
        points_df["district"] = points_df["district"].replace("", pd.NA)
        if "district" in cases.columns:
            district_map = cases.set_index("case_id")["district"]
            points_df["district"] = points_df["district"].fillna(points_df["case_id"].map(district_map))
        points_df["district"] = points_df["district"].fillna("unknown")
    numeric_cols = [
        "rain_mm",
        "tmin_c",
        "tmax_c",
        "humidity_pct",
        "solar_rad",
        "precipitation_sum",
        "temperature_2m_min",
        "temperature_2m_max",
        "relative_humidity_2m_mean",
        "shortwave_radiation_sum",
    ]
    numeric_cols = [col for col in numeric_cols if col in points_df.columns]

    group_cols = ["case_id", "district", "date"]
    agg_df = points_df[group_cols + numeric_cols].groupby(group_cols, dropna=False).mean().reset_index()
    agg_df["source"] = f"{provider}_point_mean"

    out_path = interim_dir / "weather_daily.csv"
    agg_df.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    report_path = reports_dir / "weather_coverage_report.md"
    _write_coverage_report(report_path, agg_df, cases)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
