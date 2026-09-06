from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _is_excluded(col: str) -> bool:
    excluded = {
        "yield_kg_ha",
        "yield_quintile",
        "actual_yield_kg_ha",
        "predicted_yield_kg_ha",
        "absolute_error",
        "percentage_error",
        "case_id",
        "source_file",
        "source_sheet",
        "notes",
        "split",
        "season_year_start",
        "season_year_end",
        "stage_start_date",
        "stage_end_date",
    }
    col_lower = col.lower()
    if col_lower in excluded:
        return True
    for keyword in ["source", "note", "comment", "remark", "description", "free_text", "text", "sheet", "file"]:
        if keyword in col_lower:
            return True
    return False


def _group_feature(col: str) -> str:
    col_lower = col.lower()
    if any(k in col_lower for k in ["crop_name", "season_type"]):
        return "crop/meta"
    if any(k in col_lower for k in ["rain", "precip", "tmin", "tmax", "gdd", "temp", "humidity", "radiation", "wind"]):
        return "weather"
    if any(k in col_lower for k in ["ndvi", "ndre", "evi", "cloud", "canopy", "satellite"]):
        return "satellite"
    if any(k in col_lower for k in ["image", "crop_cover", "weed", "ndvi_crop_only"]):
        return "image"
    if any(k in col_lower for k in ["stage", "das", "date"]):
        return "stage/time"
    if any(k in col_lower for k in ["district", "lat", "lon", "location", "area"]):
        return "district/location"
    if any(k in col_lower for k in ["yield", "target", "error", "pred"]):
        return "target/leakage"
    return "other"


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    ensure_dir(reports_dir)

    stage_path = processed_dir / "stage_features_with_split.csv"
    if not stage_path.exists():
        raise FileNotFoundError("Missing stage_features_with_split.csv")

    df = pd.read_csv(stage_path)

    image_features_path = processed_dir / "image_features.csv"
    image_stage_path = processed_dir / "image_stage_features.csv"
    image_features = pd.read_csv(image_features_path) if image_features_path.exists() else pd.DataFrame()
    image_stage = pd.read_csv(image_stage_path) if image_stage_path.exists() else pd.DataFrame()
    predictions_path = processed_dir / "mehsana_2025_stagewise_predictions.csv"
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()

    cases_path = interim_dir / "cases.csv"
    if cases_path.exists() and "district" not in df.columns:
        cases = pd.read_csv(cases_path)
        df = df.merge(cases[["case_id", "district"]], on="case_id", how="left")

    total_cols = len(df.columns)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]

    excluded_cols = [col for col in df.columns if _is_excluded(col)]
    candidate_numeric = [col for col in numeric_cols if col not in excluded_cols]

    missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    missing_summary = missing_pct.to_dict()

    unique_counts = {col: int(df[col].nunique(dropna=True)) for col in df.columns}

    dropped_high_missing = [col for col in candidate_numeric if missing_pct.get(col, 0) > 90.0]
    used_numeric = [col for col in candidate_numeric if col not in dropped_high_missing]

    has_district = "district" in df.columns
    used_features = used_numeric.copy()
    if has_district:
        used_features.append("district (one-hot)")

    group_map = {}
    for col in df.columns:
        group_map.setdefault(_group_feature(col), []).append(col)

    split_counts = df["split"].value_counts(dropna=False).to_dict() if "split" in df.columns else {}
    year_counts = df["season_year_start"].value_counts(dropna=False).to_dict() if "season_year_start" in df.columns else {}
    prediction_rows = int((df["split"] == "prediction_2025").sum()) if "split" in df.columns else 0

    weather_cols = [
        "rain_sum",
        "cumulative_rain_mm",
        "tmin_mean",
        "tmax_mean",
        "mean_tmin_c",
        "mean_tmax_c",
        "max_tmax_c",
        "heat_stress_days",
        "gdd_cum",
        "gdd_cumulative",
    ]
    satellite_cols = [
        "ndvi_last",
        "ndvi_mean",
        "ndvi_max",
        "ndvi_slope",
        "ndre_last",
        "ndre_mean",
        "ndre_slope",
        "evi_last",
        "evi_mean",
        "evi_slope",
    ]
    image_cols = [
        "image_green_pixel_ratio_stage_avg",
        "image_canopy_density_proxy_stage_avg",
        "image_vegetation_pixel_ratio_stage_avg",
        "image_brightness_mean_stage_avg",
        "image_saturation_mean_stage_avg",
        "image_texture_proxy_stage_avg",
        "image_count_stage",
        "image_coverage_flag",
    ]

    mehsana_rows_2024 = 0
    if "district" in df.columns and "season_year_start" in df.columns:
        mehsana_rows_2024 = int(
            df[(df["district"].str.contains("MEHSANA", case=False, na=False)) & (df["season_year_start"] == 2024)].shape[0]
        )

    report_path = reports_dir / "feature_diagnosis_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Feature Diagnosis Report\n\n")
        f.write(f"- Input file: {stage_path}\n")
        f.write(f"- Total columns: {total_cols}\n")
        f.write(f"- Numeric columns: {len(numeric_cols)}\n")
        f.write(f"- Categorical columns: {len(categorical_cols)}\n\n")

        f.write("## Columns excluded from modeling\n")
        f.write(f"- {', '.join(excluded_cols) if excluded_cols else 'none'}\n\n")

        f.write("## Columns used by LSTM\n")
        f.write(f"- Candidate numeric columns: {len(candidate_numeric)}\n")
        f.write(f"- Dropped (>90% missing): {', '.join(dropped_high_missing) if dropped_high_missing else 'none'}\n")
        f.write(f"- Final numeric columns: {len(used_numeric)}\n")
        f.write(f"- Include district encoding: {'yes' if has_district else 'no'}\n")
        f.write(f"- Final features: {', '.join(used_features) if used_features else 'none'}\n\n")

        f.write("## Missing percentage per column\n")
        for col, pct in missing_summary.items():
            f.write(f"- {col}: {pct:.2f}%\n")
        f.write("\n")

        f.write("## Unique values per column\n")
        for col, count in unique_counts.items():
            f.write(f"- {col}: {count}\n")
        f.write("\n")

        f.write("## Feature groups\n")
        for group, cols in group_map.items():
            f.write(f"- {group}: {', '.join(cols)}\n")
        f.write("\n")

        f.write("## Image feature coverage\n")
        if image_features.empty:
            f.write("- Image features file not found or empty.\n")
        else:
            f.write(f"- Image features rows: {len(image_features)}\n")
            stages = image_features.get("stage_label_standardized")
            if stages is not None:
                stage_list = sorted(stages.dropna().unique().tolist())
                f.write(f"- Image stages covered: {', '.join(stage_list)}\n")
        if image_stage.empty:
            f.write("- image_stage_features.csv not found or empty.\n")
        else:
            f.write(f"- image_stage_features rows: {len(image_stage)}\n")
            if "stage" in image_stage.columns:
                stage_list = sorted(image_stage["stage"].dropna().unique().tolist())
                f.write(f"- Stages merged into model data: {', '.join(stage_list)}\n")
        f.write("\n")

        f.write("## Split counts\n")
        f.write(f"- Rows per split: {split_counts}\n")
        f.write(f"- Rows per season_year_start: {year_counts}\n")
        f.write(f"- prediction_2025 rows: {prediction_rows}\n")
        f.write(f"- Mehsana rows for 2024: {mehsana_rows_2024}\n\n")

        f.write("## Coverage checks\n")
        for split, split_df in df.groupby("split"):
            weather_missing = split_df[weather_cols].isna().mean().mean() * 100.0 if set(weather_cols).issubset(split_df.columns) else 100.0
            satellite_missing = split_df[satellite_cols].isna().mean().mean() * 100.0 if set(satellite_cols).issubset(split_df.columns) else 100.0
            image_missing = split_df[image_cols].isna().mean().mean() * 100.0 if set(image_cols).issubset(split_df.columns) else 100.0
            f.write(f"- {split}: weather missing {weather_missing:.2f}%, satellite missing {satellite_missing:.2f}%, image missing {image_missing:.2f}%\n")
        f.write("\n")

        if "district" in df.columns:
            f.write("## Satellite coverage by district\n")
            if set(satellite_cols).issubset(df.columns):
                for district_name, group in df.groupby("district"):
                    missing_pct = group[satellite_cols].isna().mean().mean() * 100.0
                    f.write(f"- {district_name}: satellite missing {missing_pct:.2f}%\n")
            else:
                f.write("- Satellite columns missing from stage features.\n")
            f.write("\n")

        if "image_coverage_flag" in df.columns:
            f.write("## Image coverage by stage\n")
            stage_cov = df.groupby("stage")["image_coverage_flag"].mean().sort_index()
            for stage, val in stage_cov.items():
                f.write(f"- {stage}: {val:.2f}\n")
            f.write("\n")

        if not predictions.empty and "predicted_yield_kg_ha" in predictions.columns:
            for case_id, group in predictions.groupby("case_id"):
                if group["predicted_yield_kg_ha"].nunique(dropna=True) <= 1:
                    f.write(f"- Flat prediction detected for {case_id}.\n")

        if len(used_numeric) <= 4:
            f.write("## Diagnosis\n")
            f.write("- Only a few numeric features are available after exclusions and missing-value filtering.\n")
            f.write("- This limits model capacity and can degrade LSTM performance.\n")

    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
