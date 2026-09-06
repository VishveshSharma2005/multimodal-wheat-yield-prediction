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


def _match_columns(columns, patterns):
    matches = []
    for col in columns:
        col_lower = col.lower()
        for pat in patterns:
            if pat in col_lower:
                matches.append(col)
                break
    return matches


def _missing_summary(df, columns):
    if not columns:
        return {"columns": [], "missing_count": 0, "missing_pct": 0.0}
    missing_count = int(df[columns].isna().sum().sum())
    total = int(df[columns].shape[0] * df[columns].shape[1])
    missing_pct = (missing_count / total * 100.0) if total else 0.0
    return {"columns": columns, "missing_count": missing_count, "missing_pct": missing_pct}


def add_split_labels(stage_df: pd.DataFrame, cases_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if "season_year_start" not in stage_df.columns:
        merged = stage_df.merge(
            cases_df[["case_id", "season_year_start"]],
            on="case_id",
            how="left",
        )
    else:
        merged = stage_df.copy()
    def _split(year):
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

    merged["split"] = merged["season_year_start"].apply(_split)
    merged.to_csv(out_path, index=False)
    return merged


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    ensure_dir(reports_dir)

    cases_path = interim_dir / "cases.csv"
    stage_path = processed_dir / "stage_features_with_split.csv"
    if not stage_path.exists():
        stage_path = processed_dir / "stage_features.csv"
    targets_path = processed_dir / "targets.csv"

    if not cases_path.exists() or not stage_path.exists():
        raise FileNotFoundError("cases.csv or stage_features.csv not found.")

    cases = pd.read_csv(cases_path)
    stage = pd.read_csv(stage_path)
    targets = pd.read_csv(targets_path) if targets_path.exists() else pd.DataFrame()

    districts = sorted(cases["district"].dropna().unique().tolist()) if "district" in cases.columns else []
    years = sorted(cases["season_year_start"].dropna().unique().tolist()) if "season_year_start" in cases.columns else []
    stages = sorted(stage["stage"].dropna().unique().tolist()) if "stage" in stage.columns else []

    mehsana_mask = cases["district"].str.contains("MEHSANA", case=False, na=False) if "district" in cases.columns else pd.Series([], dtype=bool)
    mehsana_years = sorted(cases.loc[mehsana_mask, "season_year_start"].dropna().unique().tolist()) if not cases.empty else []

    missing_yield_cases = int(cases["yield_kg_ha"].isna().sum()) if "yield_kg_ha" in cases.columns else 0

    weather_cols = _match_columns(stage.columns, ["rain_", "tmin", "tmax", "gdd", "precip", "temp", "humidity", "radiation", "wind"])
    satellite_cols = _match_columns(stage.columns, ["ndvi", "ndre", "evi", "cloud", "canopy_cover"])
    image_cols = _match_columns(stage.columns, ["image_", "stage_label", "image_type", "image_coverage_flag"])
    weed_cols = _match_columns(stage.columns, ["weed", "crop_cover", "ndvi_crop_only"])

    weather_missing = _missing_summary(stage, weather_cols)
    satellite_missing = _missing_summary(stage, satellite_cols)
    image_missing = _missing_summary(stage, image_cols)
    weed_missing = _missing_summary(stage, weed_cols)

    case_stage_counts = stage.groupby("case_id")["stage"].nunique()
    single_stage_cases = case_stage_counts[case_stage_counts < 2].index.tolist()

    leakage_cols = [
        col for col in stage.columns
        if col.lower() in {"yield_kg_ha", "yield_quintile", "target"}
    ]

    # Write split file
    split_out = processed_dir / "stage_features_with_split.csv"
    split_df = add_split_labels(stage, cases, split_out)

    split_counts = split_df["split"].value_counts(dropna=False).to_dict() if "split" in split_df.columns else {}
    year_counts = split_df["season_year_start"].value_counts(dropna=False).to_dict() if "season_year_start" in split_df.columns else {}
    prediction_rows = int((split_df["split"] == "prediction_2025").sum()) if "split" in split_df.columns else 0
    has_2024 = 2024 in years

    targets_2024 = 0
    if not targets.empty and "season_year" in targets.columns:
        targets_2024 = int((targets["season_year"] == 2024).sum())

    report_path = reports_dir / "training_data_verification.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Training Data Verification\n\n")
        f.write(f"- Number of districts: {len(districts)}\n")
        f.write(f"- Districts: {', '.join(districts)}\n\n")
        f.write(f"- Years available: {', '.join(str(y) for y in years)}\n")
        f.write(f"- Number of cases: {len(cases)}\n")
        f.write(f"- Number of stage rows: {len(stage)}\n")
        f.write(f"- Number of crop stages: {len(stages)}\n")
        f.write(f"- Stages: {', '.join(stages)}\n\n")
        f.write(f"- Mehsana exists: {'yes' if mehsana_mask.any() else 'no'}\n")
        f.write(f"- Mehsana years: {', '.join(str(y) for y in mehsana_years)}\n\n")
        f.write(f"- Missing yield_kg_ha count: {missing_yield_cases}\n\n")
        f.write("## Missing values by feature group\n")
        f.write(f"- Weather: {weather_missing['missing_count']} ({weather_missing['missing_pct']:.2f}%)\n")
        f.write(f"- Satellite: {satellite_missing['missing_count']} ({satellite_missing['missing_pct']:.2f}%)\n")
        f.write(f"- Image: {image_missing['missing_count']} ({image_missing['missing_pct']:.2f}%)\n")
        f.write(f"- Weed/Crop concentration: {weed_missing['missing_count']} ({weed_missing['missing_pct']:.2f}%)\n\n")

        f.write("## Year checks\n")
        f.write(f"- 2020-21 present: {'yes' if 2020 in years else 'no'}\n")
        f.write(f"- 2021-22 present: {'yes' if 2021 in years else 'no'}\n")
        f.write(f"- 2022-23 present: {'yes' if 2022 in years else 'no'}\n")
        f.write(f"- 2023-24 present: {'yes' if 2023 in years else 'no'}\n")
        f.write(f"- 2024-25 present: {'yes' if 2024 in years else 'no'}\n\n")

        f.write("## Split checks\n")
        f.write(f"- Split counts: {split_counts}\n")
        f.write(f"- Rows with split=prediction_2025: {prediction_rows}\n")
        f.write(f"- season_year_start==2024 exists: {'yes' if has_2024 else 'no'}\n")
        f.write(f"- targets season_year==2024 rows: {targets_2024}\n")
        if prediction_rows == 0 or targets_2024 == 0:
            f.write("- 2025 prediction/evaluation rows not available because 2024-25 district APY/cases are missing.\n")
        f.write("\n")

        f.write("## Coverage summary\n")
        f.write(f"- Weather columns present: {', '.join(weather_cols) if weather_cols else 'none'}\n")
        f.write(f"- Satellite columns present: {', '.join(satellite_cols) if satellite_cols else 'none'}\n")
        f.write(f"- Image columns present: {', '.join(image_cols) if image_cols else 'none'}\n")

        f.write("## Case-stage checks\n")
        f.write(f"- Cases with <2 stages: {len(single_stage_cases)}\n")
        if single_stage_cases:
            f.write(f"- Case IDs: {', '.join(single_stage_cases)}\n")
        f.write("\n")

        f.write("## Leakage check\n")
        f.write(f"- Leakage columns in stage_features: {', '.join(leakage_cols) if leakage_cols else 'none'}\n")

    logger.info("Wrote %s", report_path)
    logger.info("Wrote %s", split_out)

    print("Training data verification complete.")
    print(f"Districts: {len(districts)} | Cases: {len(cases)} | Stage rows: {len(stage)}")
    print(f"Stages: {len(stages)} | Mehsana present: {'yes' if mehsana_mask.any() else 'no'}")
    print(f"Missing yield_kg_ha: {missing_yield_cases}")


if __name__ == "__main__":
    main()
