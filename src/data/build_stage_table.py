from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger
from src.features.crop_stages import build_stage_windows
from src.features.weather_features import compute_gdd
from src.features.vegetation_indices import linear_slope


logger = get_logger(__name__)


def to_datetime(series):
    return pd.to_datetime(series, errors="coerce")


def compute_split(year):
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


def _aggregate_weather_daily(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df
    if "field_id" not in weather_df.columns:
        return weather_df

    df = weather_df.copy()
    df["date"] = to_datetime(df["date"])

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
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    group_cols = ["case_id", "date"]
    if "district" in df.columns:
        group_cols.insert(1, "district")

    df = df[group_cols + numeric_cols].groupby(group_cols).mean().reset_index()
    return df


def _aggregate_satellite_daily(sat_df: pd.DataFrame) -> pd.DataFrame:
    if sat_df.empty:
        return sat_df
    if "field_id" not in sat_df.columns:
        return sat_df

    df = sat_df.copy()
    df["date"] = to_datetime(df["date"])

    numeric_cols = ["ndvi", "ndvi_raw", "ndre", "evi", "cloud_pct"]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    group_cols = ["case_id", "date"]
    # Keep district in grouping only when it has usable values; otherwise
    # groupby would drop all rows because NaN keys are excluded.
    if "district" in df.columns and df["district"].notna().any():
        group_cols.insert(1, "district")

    df = df[group_cols + numeric_cols].groupby(group_cols).mean().reset_index()
    return df


def aggregate_weather(weather_df, case_id, start_date, end_date):
    df = weather_df[(weather_df["case_id"] == case_id)].copy()
    df["date"] = to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    if df.empty:
        return {}

    rain = df.get("rain_mm") if "rain_mm" in df.columns else df.get("precipitation_sum")
    tmin = df.get("tmin_c") if "tmin_c" in df.columns else df.get("temperature_2m_min")
    tmax = df.get("tmax_c") if "tmax_c" in df.columns else df.get("temperature_2m_max")

    rain_sum = rain.sum() if rain is not None else np.nan
    tmin_mean = tmin.mean() if tmin is not None else np.nan
    tmax_mean = tmax.mean() if tmax is not None else np.nan
    tmax_max = tmax.max() if tmax is not None else np.nan
    heat_stress_days = int((tmax >= 32.0).sum()) if tmax is not None else np.nan

    out = {
        "rain_sum": rain_sum,
        "tmin_mean": tmin_mean,
        "tmax_mean": tmax_mean,
        "cumulative_rain_mm": rain_sum,
        "mean_tmin_c": tmin_mean,
        "mean_tmax_c": tmax_mean,
        "max_tmax_c": tmax_max,
        "heat_stress_days": heat_stress_days,
    }

    if tmin is not None and tmax is not None:
        gdd = compute_gdd(tmin.values, tmax.values, t_base=0.0)
        gdd_cum = np.nansum(gdd)
        out["gdd_cum"] = gdd_cum
        out["gdd_cumulative"] = gdd_cum
        # normalized cumulative GDD (0-1) relative to season length
        period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days or 1
        out["gdd_norm"] = float(gdd_cum / (period_days * 30.0))
    else:
        out["gdd_cum"] = np.nan
        out["gdd_cumulative"] = np.nan

    return out


def aggregate_satellite(sat_df, case_id, end_date):
    out = {
        "ndvi_last": np.nan,
        "ndvi_mean": np.nan,
        "ndvi_max": np.nan,
        "ndvi_slope": np.nan,
        "ndre_last": np.nan,
        "ndre_mean": np.nan,
        "ndre_slope": np.nan,
        "evi_last": np.nan,
        "evi_mean": np.nan,
        "evi_slope": np.nan,
    }
    if sat_df.empty or "case_id" not in sat_df.columns:
        return out

    df = sat_df[(sat_df["case_id"] == case_id)].copy()
    if df.empty:
        return out
    df["date"] = to_datetime(df["date"])
    df = df[df["date"] <= end_date]
    if df.empty:
        return out

    df = df.sort_values("date")
    x_days = (df["date"] - df["date"].min()).dt.days.values

    if "ndvi_raw" in df.columns:
        ndvi_series = df["ndvi_raw"]
    elif "ndvi" in df.columns:
        ndvi_series = df["ndvi"]
    else:
        ndvi_series = None

    if ndvi_series is not None:
        out["ndvi_last"] = ndvi_series.iloc[-1]
        out["ndvi_mean"] = ndvi_series.mean()
        out["ndvi_max"] = ndvi_series.max()
        out["ndvi_slope"] = linear_slope(x_days, ndvi_series.values)

    if "ndre" in df.columns:
        ndre_series = df["ndre"]
        out["ndre_last"] = ndre_series.iloc[-1]
        out["ndre_mean"] = ndre_series.mean()
        out["ndre_slope"] = linear_slope(x_days, ndre_series.values)

    if "evi" in df.columns:
        evi_series = df["evi"]
        out["evi_last"] = evi_series.iloc[-1]
        out["evi_mean"] = evi_series.mean()
        out["evi_slope"] = linear_slope(x_days, evi_series.values)

    return out


def aggregate_drone(drone_df, case_id, end_date):
    if drone_df.empty or "case_id" not in drone_df.columns:
        return {}
    df = drone_df[(drone_df["case_id"] == case_id)].copy()
    if df.empty:
        return {}
    df["date"] = to_datetime(df["date"])
    df = df[df["date"] <= end_date]
    if df.empty:
        return {}

    out = {}
    for col in [
        "canopy_cover_pct",
        "weed_cover_pct",
        "crop_cover_pct",
        "ndvi_raw",
        "ndvi_crop_only",
        "canopy_temp_c",
    ]:
        if col in df.columns:
            out[f"drone_{col}_mean"] = df[col].mean()
            out[f"drone_{col}_last"] = df[col].iloc[-1]

    if "weed_cover_pct" in df.columns:
        out["weed_cover_mean"] = df["weed_cover_pct"].mean()
    if "crop_cover_pct" in df.columns:
        out["crop_cover_mean"] = df["crop_cover_pct"].mean()

    return out


def aggregate_farm_logs(farm_df, case_id, start_date, end_date):
    if farm_df.empty or "case_id" not in farm_df.columns:
        return {}
    df = farm_df[(farm_df["case_id"] == case_id)].copy()
    if df.empty:
        return {}
    df["date"] = to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    if df.empty:
        return {}

    out = {}
    if "irrigation_event" in df.columns:
        out["irrigation_events_count"] = df["irrigation_event"].fillna(0).sum()
    if "irrigation_mm" in df.columns:
        out["irrigation_mm_sum"] = df["irrigation_mm"].sum()

    for col in ["fertilizer_n_kg_ha", "fertilizer_p_kg_ha", "fertilizer_k_kg_ha"]:
        if col in df.columns:
            out[f"{col}_total"] = df[col].sum()

    if "weed_control_event" in df.columns:
        out["weed_control_events_count"] = df["weed_control_event"].fillna(0).sum()

    for col in ["pest_score_0_5", "disease_score_0_5", "crop_height_cm"]:
        if col in df.columns:
            out[f"{col}_mean"] = df[col].mean()

    return out


def _merge_handcrafted_image_features(stage_df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    image_stage_path = processed_dir / "image_stage_features.csv"
    if image_stage_path.exists():
        try:
            image_stage = pd.read_csv(image_stage_path)
            if not image_stage.empty:
                if "case_id" in image_stage.columns:
                    stage_df = stage_df.merge(image_stage, on=["case_id", "stage"], how="left")
                else:
                    stage_df = stage_df.merge(image_stage, on="stage", how="left")
        except Exception as exc:
            logger.warning("Failed to merge image features: %s", exc)
    return stage_df


def _merge_cnn_stage_features(stage_df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    cnn_stage_path = processed_dir / "cnn_stage_features.csv"
    if not cnn_stage_path.exists():
        logger.warning("CNN stage features missing: %s", cnn_stage_path)
        return stage_df

    try:
        cnn_stage = pd.read_csv(cnn_stage_path)
    except Exception as exc:
        logger.warning("Failed to read CNN stage features: %s", exc)
        return stage_df

    if cnn_stage.empty or "stage" not in cnn_stage.columns:
        logger.warning("CNN stage features are empty or missing the stage column.")
        return stage_df

    rename_cols = {}
    for col in cnn_stage.columns:
        if col == "stage":
            continue
        if col == "image_count":
            rename_cols[col] = "cnn_image_count"
        elif not col.startswith("cnn_"):
            rename_cols[col] = f"cnn_{col}"
    cnn_stage = cnn_stage.rename(columns=rename_cols)

    before_cols = set(stage_df.columns)
    stage_df = stage_df.merge(cnn_stage, on="stage", how="left")
    added_cols = sorted(set(stage_df.columns) - before_cols)
    logger.info("Merged %d CNN stage feature columns.", len(added_cols))
    return stage_df


def _ensure_image_flags(stage_df: pd.DataFrame) -> pd.DataFrame:
    image_flag_col = "image_coverage_flag"
    if image_flag_col not in stage_df.columns:
        image_cols = [
            "image_green_pixel_ratio_stage_avg",
            "image_canopy_density_proxy_stage_avg",
            "image_vegetation_pixel_ratio_stage_avg",
            "image_brightness_mean_stage_avg",
            "image_saturation_mean_stage_avg",
            "image_texture_proxy_stage_avg",
            "image_count_stage",
        ]
        available = [col for col in image_cols if col in stage_df.columns]
        if available:
            stage_df[image_flag_col] = stage_df[available].notna().any(axis=1).astype(int)
        else:
            stage_df[image_flag_col] = 0
    if "image_feature_source" not in stage_df.columns:
        stage_df["image_feature_source"] = stage_df[image_flag_col].apply(
            lambda x: "auxiliary_external_image_dataset" if x == 1 else ""
        )
    return stage_df


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    ensure_dir(processed_dir)

    cases_path = interim_dir / "cases.csv"
    if not cases_path.exists():
        raise FileNotFoundError("Missing cases.csv. Run preprocess_all.py first.")

    cases = pd.read_csv(cases_path)

    weather_path = interim_dir / "weather_daily.csv"
    weather_df = pd.read_csv(weather_path) if weather_path.exists() else pd.DataFrame()
    weather_df = _aggregate_weather_daily(weather_df)

    sat_path = interim_dir / "satellite_observations.csv"
    sat_df = pd.read_csv(sat_path) if sat_path.exists() else pd.DataFrame()
    sat_df = _aggregate_satellite_daily(sat_df)
    if sat_df.empty:
        logger.warning("Satellite observations missing; satellite features will be NaN.")

    drone_path = interim_dir / "drone_observations.csv"
    drone_df = pd.read_csv(drone_path) if drone_path.exists() else pd.DataFrame()

    farm_path = interim_dir / "farm_logs.csv"
    farm_df = pd.read_csv(farm_path) if farm_path.exists() else pd.DataFrame()

    stage_defs = config["stage_definitions"]
    stage_index_map = {stage["name"]: idx for idx, stage in enumerate(stage_defs)}
    rows = []

    for _, case in cases.iterrows():
        case_id = case.get("case_id")
        sowing_date = pd.to_datetime(case.get("sowing_date"), errors="coerce")
        harvest_date = pd.to_datetime(case.get("harvest_date"), errors="coerce")
        crop_name = case.get("crop_name") if pd.notna(case.get("crop_name")) else case.get("crop")
        season_type = case.get("season_type") if pd.notna(case.get("season_type")) else case.get("season")
        crop_name = str(crop_name).strip() if pd.notna(crop_name) else "wheat"
        season_type = str(season_type).strip() if pd.notna(season_type) else "Rabi"

        if pd.isna(case_id) or pd.isna(sowing_date) or pd.isna(harvest_date):
            logger.warning("Missing case_id/sowing/harvest for a row. Skipping.")
            continue

        windows = build_stage_windows(sowing_date, stage_defs)
        for window in windows:
            start_date = window["start_date"]
            end_date = window["end_date"]
            if end_date > harvest_date:
                end_date = harvest_date

            stage_idx = stage_index_map.get(window["stage"], np.nan)

            row = {
                "case_id": case_id,
                "crop_name": crop_name,
                "season_type": season_type,
                "stage": window["stage"],
                "stage_name": window["stage"],
                "stage_start_date": start_date.date().isoformat(),
                "stage_end_date": end_date.date().isoformat(),
                "stage_start_das": window["start_das"],
                "stage_end_das": window["end_das"],
                "stage_index": stage_idx,
                "stage_order": (stage_idx + 1) if pd.notna(stage_idx) else np.nan,
                "days_after_sowing": window["start_das"],
                "crop_duration_days": int((harvest_date - sowing_date).days) if pd.notna(harvest_date) and pd.notna(sowing_date) else np.nan,
                "district": case.get("district"),
                "lat": case.get("lat"),
                "lon": case.get("lon"),
                "area_ha": case.get("area_ha"),
                "season_year_start": case.get("season_year_start"),
                "season_year_end": case.get("season_year_end"),
                "yield_kg_ha": case.get("yield_kg_ha"),
                "actual_available": int(pd.notna(case.get("yield_kg_ha"))),
                "evaluation_status": "available" if pd.notna(case.get("yield_kg_ha")) else "pending_actual_apy",
            }

            row.update(aggregate_weather(weather_df, case_id, sowing_date, end_date))
            row.update(aggregate_satellite(sat_df, case_id, end_date))
            row.update(aggregate_drone(drone_df, case_id, end_date))
            row.update(aggregate_farm_logs(farm_df, case_id, sowing_date, end_date))
            row["cumulative_rainfall"] = row.get("cumulative_rain_mm", np.nan)
            row["cumulative_gdd"] = row.get("gdd_cumulative", np.nan)
            row["ndvi_growth_rate"] = row.get("ndvi_slope", np.nan)

            rows.append(row)

    stage_df = pd.DataFrame(rows)

    if "district" in cases.columns and "ndvi_last" in stage_df.columns:
        satellite_cols = [
            col
            for col in ["ndvi_last", "ndvi_mean", "ndvi_max", "ndvi_slope", "ndre_last", "ndre_mean", "ndre_slope", "evi_last", "evi_mean", "evi_slope"]
            if col in stage_df.columns
        ]
        mehsana_rows = stage_df[stage_df["district"].astype(str).str.contains("MEHSANA", case=False, na=False)].copy()
        if "season_year_start" in mehsana_rows.columns:
            mehsana_rows = mehsana_rows[mehsana_rows["season_year_start"] == 2024]

        if not mehsana_rows.empty and satellite_cols:
            has_satellite = bool(mehsana_rows[satellite_cols].notna().any().any())
            if not has_satellite:
                logger.warning("Satellite missing for Mehsana 2024-25 prediction rows; model will use weather + image + stage/location features only.")

    stage_df = _merge_handcrafted_image_features(stage_df, processed_dir)
    stage_df = _ensure_image_flags(stage_df)
    stage_out = processed_dir / "stage_features.csv"
    stage_df.to_csv(stage_out, index=False)
    logger.info("Wrote %s", stage_out)

    stage_df = _merge_cnn_stage_features(stage_df, processed_dir)
    if "split" not in stage_df.columns:
        stage_df["split"] = stage_df["season_year_start"].apply(compute_split)
    cnn_out = processed_dir / "stage_features_with_cnn.csv"
    stage_df.to_csv(cnn_out, index=False)
    logger.info("Wrote %s", cnn_out)

    split_out = processed_dir / "stage_features_with_split.csv"
    stage_df.to_csv(split_out, index=False)
    logger.info("Wrote %s", split_out)

    targets = cases[["case_id", "season_year", "yield_kg_ha"]].copy()

    train_years = set(config["train_years"])
    train_mask = targets["season_year"].isin(train_years)
    train_yields = targets.loc[train_mask, "yield_kg_ha"].dropna()

    if len(train_yields) >= 5:
        bins = pd.qcut(train_yields, 5, labels=[1, 2, 3, 4, 5])
        thresholds = pd.qcut(train_yields, 5, retbins=True)[1]
        targets["yield_quintile"] = pd.cut(
            targets["yield_kg_ha"], bins=thresholds, labels=[1, 2, 3, 4, 5], include_lowest=True
        )
    else:
        targets["yield_quintile"] = np.nan
        logger.warning("Insufficient training samples to compute quintiles.")

    targets_out = processed_dir / "targets.csv"
    targets.to_csv(targets_out, index=False)
    logger.info("Wrote %s", targets_out)


if __name__ == "__main__":
    main()
