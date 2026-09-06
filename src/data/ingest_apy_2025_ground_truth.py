from __future__ import annotations

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

# The forecast season. Rows with these `season_year_start` values are the ones
# whose target this script is allowed to touch.
PREDICTION_SEASON_START_YEARS = {2024}

GROUND_TRUTH_COLUMNS = [
    "district",
    "state",
    "crop",
    "season",
    "agri_year",
    "season_year_start",
    "season_year_end",
    "yield_kg_ha",
    "source_file",
    "scope",
]

# Stage tables that carry the target and must stay consistent with cases.csv.
STAGE_TABLE_NAMES = [
    "stage_features.csv",
    "stage_features_with_split.csv",
    "stage_features_with_cnn.csv",
]

TARGET_COLUMNS = ["yield_kg_ha", "actual_available", "evaluation_status"]


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_norm = str(col).strip().lower().replace(" ", "_")
        rename_map[col] = col_norm
    return df.rename(columns=rename_map)


def _read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        frames = []
        for sheet_name, sheet_df in sheets.items():
            if sheet_df is None or sheet_df.empty:
                continue
            frame = _normalize_columns(sheet_df.copy())
            frame["source_sheet"] = sheet_name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _normalize_columns(pd.read_csv(path))


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _parse_agri_year(value: object) -> tuple[int | None, int | None, str]:
    text = _normalize_text(value)
    if not text:
        return None, None, ""
    digits = [part for part in text.replace("/", "-").split("-") if part]
    if len(digits) >= 2 and digits[0].isdigit() and digits[1].isdigit():
        start_year = int(digits[0])
        end_year = int(digits[1])
        return start_year, end_year, f"{start_year}-{str(end_year)[-2:]}"
    if text.isdigit():
        year = int(text)
        return year, year + 1, str(year)
    return None, None, text


def _is_gujarat_row(df: pd.DataFrame) -> pd.Series:
    state_col = _first_present(df, ["state", "state_name", "state_ut", "state_ut_name"])
    if state_col is None:
        return pd.Series([True] * len(df), index=df.index)
    return df[state_col].astype(str).str.contains("GUJARAT", case=False, na=False)


def _is_wheat_row(df: pd.DataFrame) -> pd.Series:
    crop_col = _first_present(df, ["crop", "crop_name", "commodity"])
    if crop_col is None:
        return pd.Series([True] * len(df), index=df.index)
    return df[crop_col].astype(str).str.contains("WHEAT", case=False, na=False)


def _is_rabi_row(df: pd.DataFrame) -> pd.Series:
    season_col = _first_present(df, ["season", "season_name", "crop_season"])
    if season_col is None:
        return pd.Series([True] * len(df), index=df.index)
    return df[season_col].astype(str).str.contains("RABI", case=False, na=False)


# District names that denote an aggregate rather than a real district. A row
# carrying one of these is a state/national total and must never become a
# district-level ground-truth target.
_AGGREGATE_DISTRICT_PATTERN = "GUJARAT|STATE|TOTAL|ALL INDIA|ALL_DISTRICTS|ALL DISTRICTS"


def _is_district_level(df: pd.DataFrame) -> pd.Series:
    scope_col = _first_present(df, ["scope", "level", "report_level", "granularity"])
    district_col = _first_present(df, ["district", "district_name"])
    if district_col is None:
        return pd.Series([False] * len(df), index=df.index)

    district_text = df[district_col].astype(str)
    named = district_text.str.len().gt(0) & ~district_text.str.contains(
        _AGGREGATE_DISTRICT_PATTERN, case=False, na=False
    )
    if scope_col is not None:
        scope = df[scope_col].astype(str).str.contains("district|state_estimate", case=False, na=False)
        return scope & named
    return named


def _extract_yield(df: pd.DataFrame) -> pd.Series:
    yield_col = _first_present(df, ["yield_kg_ha", "yield", "yield_value", "yield_per_ha"])
    if yield_col is not None:
        return pd.to_numeric(df[yield_col], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")


def _extract_apy_ground_truth(raw_dir: Path) -> pd.DataFrame:
    files = sorted([*raw_dir.glob("*.csv"), *raw_dir.glob("*.xlsx"), *raw_dir.glob("*.xls")])
    frames = []
    raw_frames_all = []
    for path in files:
        try:
            df = _read_tabular_file(path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            continue
        if df.empty:
            continue

        df["source_file"] = path.name
        district_col = _first_present(df, ["district", "district_name"])
        if district_col is None:
            continue

        district = df[district_col].astype(str).str.strip()
        df["district"] = district
        state_col = _first_present(df, ["state", "state_name", "state_ut", "state_ut_name"])
        if state_col is not None:
            df["state"] = df[state_col].astype(str).str.strip()
        else:
            df["state"] = "Gujarat"

        crop_col = _first_present(df, ["crop", "crop_name", "commodity"])
        if crop_col is not None:
            df["crop"] = df[crop_col].astype(str).str.strip()
        else:
            df["crop"] = ""

        season_col = _first_present(df, ["season", "season_name", "crop_season"])
        if season_col is not None:
            df["season"] = df[season_col].astype(str).str.strip()
        else:
            df["season"] = ""

        agri_col = _first_present(df, ["agri_year", "agri_years", "year", "crop_year"])
        if agri_col is not None:
            df["agri_year_raw"] = df[agri_col].astype(str).str.strip()
        else:
            df["agri_year_raw"] = ""
        df["yield_kg_ha"] = _extract_yield(df)
        # Preserve the source file's own scope column when it has one. Overwriting
        # it with "district" would let state-level aggregate rows (for example the
        # Gujarat 2024-25 Final Advance Estimate) pass the district-level filter.
        if "scope" not in df.columns:
            df["scope"] = "district"

        if "state" in df.columns:
            df = df[df["state"].astype(str).str.contains("GUJARAT", case=False, na=False) | df["state"].isna()]
        df = df[_is_gujarat_row(df)]
        df = df[_is_wheat_row(df)]
        df = df[_is_rabi_row(df)]
        district_mask = _is_district_level(df)
        df = df[district_mask]

        if df.empty:
            continue

        parsed_years = df["agri_year_raw"].apply(_parse_agri_year)
        df["season_year_start"] = parsed_years.apply(lambda item: item[0])
        df["season_year_end"] = parsed_years.apply(lambda item: item[1])
        df["agri_year"] = parsed_years.apply(lambda item: item[2])

        season_start_col = _first_present(df, ["season_year_start", "year_start", "start_year"])
        season_end_col = _first_present(df, ["season_year_end", "year_end", "end_year"])
        if season_start_col is not None:
            df["season_year_start"] = pd.to_numeric(df[season_start_col], errors="coerce").fillna(df["season_year_start"])
        if season_end_col is not None:
            df["season_year_end"] = pd.to_numeric(df[season_end_col], errors="coerce").fillna(df["season_year_end"])

        df = df[
            df["season_year_start"].isin([2024, 2025])
            & df["season_year_end"].isin([2025, 2026])
        ]
        if df.empty:
            continue

        frames.append(
            df[[
                "district",
                "state",
                "crop",
                "season",
                "agri_year",
                "season_year_start",
                "season_year_end",
                "yield_kg_ha",
                "source_file",
                "scope",
            ]].copy()
        )

    if not frames:
        # No district-level 2024-25 APY rows were found.
        #
        # RESEARCH INTEGRITY: do NOT fall back to the most recent district yield
        # from an earlier season. Carrying a 2023-24 value forward and labelling
        # it a 2024-25 actual produces an invalid evaluation (it scores a
        # 2024-25 forecast against a 2023-24 observation). The prediction-year
        # target must stay missing until the Directorate of Agriculture,
        # Government of Gujarat publishes district-level 2024-25 APY.
        # See docs/2025_yield_reference_methodology.md.
        logger.warning(
            "No district-level %s APY rows found. Prediction-year target stays "
            "missing (evaluation_status=pending_actual_apy). No proxy or "
            "carried-forward value is created.",
            "/".join(str(y) for y in sorted(PREDICTION_SEASON_START_YEARS)),
        )
        return pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out["district"] = out["district"].astype(str).str.strip()
    out = out[out["district"].str.len() > 0]
    out = out.drop_duplicates(subset=["district", "agri_year"], keep="last")
    return out


def _write_cases_updates(cases_path: Path, apy_df: pd.DataFrame) -> None:
    if not cases_path.exists():
        logger.warning("cases.csv not found at %s", cases_path)
        return

    cases = pd.read_csv(cases_path)
    if cases.empty:
        cases.to_csv(cases_path, index=False)
        return

    if "season_year_start" in cases.columns:
        pred_mask = cases["season_year_start"].isin(PREDICTION_SEASON_START_YEARS)
    else:
        pred_mask = cases.get("season_year", pd.Series(dtype=float)).isin(PREDICTION_SEASON_START_YEARS)

    for col in ["actual_available", "evaluation_status"]:
        if col not in cases.columns:
            cases[col] = pd.NA

    # Build the lookup from official prediction-season rows ONLY. A row from any
    # other season must never supply the prediction-season target.
    district_to_yield: dict[str, float] = {}
    if not apy_df.empty and "season_year_start" in apy_df.columns:
        official = apy_df[apy_df["season_year_start"].isin(PREDICTION_SEASON_START_YEARS)]
        official = official.dropna(subset=["district", "yield_kg_ha"])
        if not official.empty:
            lookup = official[["district", "yield_kg_ha"]].copy()
            lookup["district_norm"] = lookup["district"].astype(str).str.strip().str.upper()
            district_to_yield = (
                lookup.drop_duplicates(subset=["district_norm"], keep="last")
                .set_index("district_norm")["yield_kg_ha"]
                .to_dict()
            )

    # Assign unconditionally. When no official row exists for a district the map
    # yields NaN, which is the correct state: the target stays missing rather
    # than retaining a stale value from a previous run.
    cases.loc[pred_mask, "yield_kg_ha"] = (
        cases.loc[pred_mask, "district"].astype(str).str.strip().str.upper().map(district_to_yield)
    )
    logger.info(
        "Prediction-season rows: %d; official targets matched: %d",
        int(pred_mask.sum()),
        int(cases.loc[pred_mask, "yield_kg_ha"].notna().sum()),
    )

    cases["actual_available"] = cases["yield_kg_ha"].notna().astype(int)
    cases["evaluation_status"] = cases["actual_available"].map(lambda flag: "available" if int(flag) == 1 else "pending_actual_apy")

    cases.to_csv(cases_path, index=False)
    logger.info("Updated %s", cases_path)


def _write_stage_updates(stage_path: Path, cases: pd.DataFrame) -> None:
    """Refresh the target columns of one stage table from cases.csv.

    Column order is preserved so the refreshed table is a drop-in replacement.
    """
    if not stage_path.exists() or cases.empty:
        return

    stage = pd.read_csv(stage_path)
    if "case_id" not in stage.columns:
        return

    present = [col for col in TARGET_COLUMNS if col in cases.columns]
    if not present:
        return

    original_order = stage.columns.tolist()
    stage = stage.drop(columns=[col for col in TARGET_COLUMNS if col in stage.columns], errors="ignore")
    stage = stage.merge(cases[["case_id"] + present], on="case_id", how="left")

    # Restore the original column order; any target column the table did not
    # previously have is appended at the end.
    ordered = [col for col in original_order if col in stage.columns]
    ordered += [col for col in stage.columns if col not in ordered]
    stage = stage[ordered]

    stage.to_csv(stage_path, index=False)
    logger.info("Updated %s", stage_path)


def main() -> None:
    config = load_config()
    raw_dir = Path(config["paths"]["apy_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    ensure_dir(interim_dir)
    ensure_dir(reports_dir)

    output_path = interim_dir / "apy_2025_ground_truth.csv"
    # Always re-derive from the raw sources. Re-using the previous output would
    # keep any stale or carried-forward row alive across runs.
    apy_df = _extract_apy_ground_truth(raw_dir)
    apy_df.to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)

    cases_path = interim_dir / "cases.csv"
    _write_cases_updates(cases_path, apy_df)

    cases = pd.read_csv(cases_path) if cases_path.exists() else pd.DataFrame()
    for stage_name in STAGE_TABLE_NAMES:
        _write_stage_updates(processed_dir / stage_name, cases)

    if apy_df.empty:
        logger.warning("No district-level APY ground truth found. Evaluation will remain pending.")
    else:
        logger.info("Loaded %d district-level APY rows.", len(apy_df))

    print(f"APY 2025 ground truth rows: {len(apy_df)}")
    print(f"APY 2025 output: {output_path}")


if __name__ == "__main__":
    main()
