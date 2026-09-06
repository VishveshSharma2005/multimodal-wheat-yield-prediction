from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)

WHEAT_PREDICTION_START = pd.Timestamp("2024-11-20")
WHEAT_PREDICTION_END = pd.Timestamp("2025-03-25")


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper().replace("-", "_")


def _season_year_start_for_date(dt: pd.Timestamp | None) -> int | None:
    if dt is None or pd.isna(dt):
        return None
    return int(dt.year if dt.month >= 7 else dt.year - 1)


def _season_window(season_year_start: int | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if season_year_start is None:
        return None, None
    return pd.Timestamp(f"{season_year_start}-11-20"), pd.Timestamp(f"{season_year_start + 1}-03-25")


def _overlaps(start_a, end_a, start_b, end_b) -> bool:
    if pd.isna(start_a) or pd.isna(end_a) or pd.isna(start_b) or pd.isna(end_b):
        return False
    return start_a <= end_b and start_b <= end_a


def _parse_filename_metadata(file_name: str) -> dict[str, str]:
    stem = Path(file_name).stem
    cleaned = stem.replace("sentinel2_", "", 1)
    tokens = [token for token in re.split(r"[_\s]+", cleaned) if token]
    metadata = {"district": "", "field_id": "", "case_id": ""}

    for token in tokens:
        token_upper = token.upper()
        if token_upper.startswith("GUJ_") or token_upper.startswith("GUJ") and token_upper.count("_") >= 2:
            metadata["case_id"] = token_upper
        elif token_upper.startswith("FIELD"):
            metadata["field_id"] = token

    if not metadata["district"] and tokens:
        district_tokens = []
        for token in tokens:
            token_upper = token.upper()
            if token_upper.startswith("FIELD") or re.fullmatch(r"20\d{2}", token_upper):
                break
            district_tokens.append(token)
        if district_tokens:
            metadata["district"] = " ".join(district_tokens).replace("_", " ")
    if not metadata["field_id"] and tokens:
        for token in tokens:
            if token.upper().startswith("FIELD"):
                metadata["field_id"] = token
                break

    return metadata


def _prefer_filename_metadata(primary: str, metadata_value: str, *, allow_field_like: bool = False) -> str:
    primary_norm = _normalize_text(primary)
    metadata_norm = _normalize_text(metadata_value)
    if not primary_norm:
        return metadata_norm
    if metadata_norm and (
        primary_norm.startswith("FIELD")
        or primary_norm.endswith("CENTROID")
        or primary_norm == metadata_norm
        or (not allow_field_like and re.fullmatch(r"20\d{2}", primary_norm))
    ):
        return metadata_norm
    return primary_norm


def _prefer_filename_metadata(primary: str, metadata_value: str, *, allow_field_like: bool = False) -> str:
    primary_norm = _normalize_text(primary)
    metadata_norm = _normalize_text(metadata_value)
    if not primary_norm:
        return metadata_norm
    if metadata_norm and (
        primary_norm.startswith("FIELD")
        or primary_norm.endswith("CENTROID")
        or primary_norm == metadata_norm
        or (not allow_field_like and re.fullmatch(r"20\d{2}", primary_norm))
    ):
        return metadata_norm
    return primary_norm


def _load_cases(interim_dir: Path) -> pd.DataFrame:
    cases_path = interim_dir / "cases.csv"
    if not cases_path.exists():
        return pd.DataFrame()
    cases = pd.read_csv(cases_path)
    for col in ["sowing_date", "harvest_date"]:
        if col in cases.columns:
            cases[col] = pd.to_datetime(cases[col], errors="coerce")
    return cases


def _load_registry(interim_dir: Path) -> pd.DataFrame:
    registry_path = interim_dir / "coordinate_registry.csv"
    if not registry_path.exists():
        return pd.DataFrame()
    registry = pd.read_csv(registry_path)
    if "district" in registry.columns:
        registry["district_norm"] = registry["district"].apply(_normalize_text)
    if "field_id" in registry.columns:
        registry["field_id_norm"] = registry["field_id"].apply(_normalize_text)
    return registry


def _case_lookup(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return cases
    lookup = cases.copy()
    for col in ["district", "case_id"]:
        if col in lookup.columns:
            lookup[f"{col}_norm"] = lookup[col].apply(_normalize_text)
    if "season_year_start" in lookup.columns:
        lookup["season_year_start"] = pd.to_numeric(lookup["season_year_start"], errors="coerce")
    return lookup


def _file_matches_case(file_start, file_end, case_row: pd.Series) -> bool:
    sow = case_row.get("sowing_date")
    harv = case_row.get("harvest_date")
    if pd.notna(sow) and pd.notna(harv):
        return _overlaps(file_start, file_end, sow, harv)
    case_season_start = case_row.get("season_year_start")
    start, end = _season_window(int(case_season_start)) if pd.notna(case_season_start) else (None, None)
    return _overlaps(file_start, file_end, start, end)


def _match_file(df: pd.DataFrame, file_name: str, cases: pd.DataFrame, registry: pd.DataFrame) -> tuple[str | None, str, str, str, str, pd.Timestamp | None, pd.Timestamp | None]:
    meta = _parse_filename_metadata(file_name)
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    file_start = df["date"].min()
    file_end = df["date"].max()
    file_season_start = _season_year_start_for_date(file_start) if pd.notna(file_start) else None

    file_case_id = ""
    if "case_id" in df.columns and df["case_id"].notna().any():
        file_case_id = _normalize_text(df["case_id"].dropna().iloc[0])
    elif meta["case_id"]:
        file_case_id = _normalize_text(meta["case_id"])

    file_district = ""
    if "district" in df.columns and df["district"].notna().any():
            file_district = _prefer_filename_metadata(df["district"].dropna().iloc[0], meta["district"])
    elif meta["district"]:
        file_district = _normalize_text(meta["district"])

    file_field_id = ""
    if "field_id" in df.columns and df["field_id"].notna().any():
        file_field_id = _normalize_text(df["field_id"].dropna().iloc[0])
    elif meta["field_id"]:
        file_field_id = _normalize_text(meta["field_id"])

    if pd.isna(file_start) or pd.isna(file_end):
        return None, "missing date range", file_case_id, file_district, file_field_id, file_start, file_end

    if not _overlaps(file_start, file_end, WHEAT_PREDICTION_START, WHEAT_PREDICTION_END):
        return None, "date range does not overlap 2024-25 wheat season", file_case_id, file_district, file_field_id, file_start, file_end

    if not cases.empty:
        if file_case_id:
            exact = cases[cases["case_id_norm"] == file_case_id]
            if not exact.empty:
                case_row = exact.iloc[0]
                if _file_matches_case(file_start, file_end, case_row):
                    return str(case_row.get("case_id")), "exact case_id match", file_case_id, file_district, file_field_id, file_start, file_end
                return None, "exact case_id found but no crop-season overlap", file_case_id, file_district, file_field_id, file_start, file_end

        if file_district:
            district_cases = cases[cases["district_norm"] == file_district].copy()
            if not district_cases.empty:
                if file_season_start is not None and "season_year_start" in district_cases.columns:
                    season_cases = district_cases[district_cases["season_year_start"] == file_season_start]
                    for _, case_row in season_cases.iterrows():
                        if _file_matches_case(file_start, file_end, case_row):
                            return str(case_row.get("case_id")), "district + same season_year_start + overlap", file_case_id, file_district, file_field_id, file_start, file_end

                if file_field_id and not registry.empty and {"district_norm", "field_id_norm", "case_id"}.issubset(registry.columns):
                    registry_candidates = registry[
                        (registry["district_norm"] == file_district)
                        & (registry["field_id_norm"] == file_field_id)
                    ]
                    for _, reg_row in registry_candidates.iterrows():
                        case_id = _normalize_text(reg_row.get("case_id"))
                        case_row = cases[cases["case_id_norm"] == case_id]
                        if case_row.empty:
                            continue
                        case_row = case_row.iloc[0]
                        if _file_matches_case(file_start, file_end, case_row):
                            return str(case_row.get("case_id")), "district + field_id + overlap", file_case_id, file_district, file_field_id, file_start, file_end

                for _, case_row in district_cases.iterrows():
                    if _file_matches_case(file_start, file_end, case_row):
                        return str(case_row.get("case_id")), "district + date overlap", file_case_id, file_district, file_field_id, file_start, file_end

    return None, "no matching case after season/date checks", file_case_id, file_district, file_field_id, file_start, file_end


def _write_markdown_report(path: Path, title: str, sections: list[tuple[str, pd.DataFrame | list[str]]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for heading, content in sections:
            f.write(f"## {heading}\n\n")
            if isinstance(content, pd.DataFrame):
                if content.empty:
                    f.write("- None\n\n")
                else:
                    try:
                        f.write(content.to_markdown(index=False))
                    except Exception:
                        f.write(content.to_string(index=False))
                    f.write("\n\n")
            else:
                if not content:
                    f.write("- None\n\n")
                else:
                    for line in content:
                        f.write(f"- {line}\n")
                    f.write("\n")


def main() -> None:
    config = load_config()
    satellite_dir = Path(config["paths"]["satellite_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    ensure_dir(reports_dir)

    cases = _case_lookup(_load_cases(interim_dir))
    registry = _load_registry(interim_dir)

    csv_files = sorted(satellite_dir.glob("*.csv"))
    rows = []
    if not csv_files:
        logger.warning("No satellite CSV files found in %s", satellite_dir)
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            continue

        matched_case_id, reason, file_case_id, file_district, file_field_id, min_date, max_date = _match_file(
            df,
            path.name,
            cases,
            registry,
        )

        rows.append(
            {
                "file": path.name,
                "file_case_id": file_case_id,
                "file_district": file_district,
                "file_field_id": file_field_id,
                "min_date": str(min_date.date()) if pd.notna(min_date) else "",
                "max_date": str(max_date.date()) if pd.notna(max_date) else "",
                "file_season_year_start": _season_year_start_for_date(min_date) if pd.notna(min_date) else None,
                "matched_case_id": matched_case_id or "",
                "match_reason": reason,
                "status": "matched" if matched_case_id else "rejected",
            }
        )

    report_df = pd.DataFrame(rows)
    matched_df = report_df[report_df["status"] == "matched"].copy() if not report_df.empty else pd.DataFrame()
    rejected_df = report_df[report_df["status"] == "rejected"].copy() if not report_df.empty else pd.DataFrame()

    cases_raw_path = interim_dir / "cases.csv"
    if cases_raw_path.exists():
        cases_raw = pd.read_csv(cases_raw_path)
        if "season_year_start" in cases_raw.columns:
            prediction_cases = cases_raw[cases_raw["season_year_start"] == 2024].copy()
        else:
            prediction_cases = cases_raw[cases_raw.get("season_year", pd.Series(dtype=float)) == 2024].copy()
        prediction_case_ids = set(prediction_cases.get("case_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    else:
        prediction_cases = pd.DataFrame()
        prediction_case_ids = set()

    matched_case_ids = set(matched_df["matched_case_id"].dropna().astype(str).tolist()) if not matched_df.empty else set()
    covered_prediction_case_ids = prediction_case_ids.intersection(matched_case_ids)
    prediction_count = len(prediction_case_ids)

    mehsana_cases = prediction_cases[prediction_cases.get("district", pd.Series(dtype=str)).astype(str).str.contains("MEHSANA", case=False, na=False)] if not prediction_cases.empty else pd.DataFrame()
    mehsana_case_ids = set(mehsana_cases.get("case_id", pd.Series(dtype=str)).dropna().astype(str).tolist()) if not mehsana_cases.empty else set()
    mehsana_has_satellite = bool(mehsana_case_ids.intersection(matched_case_ids))

    stage_availability_rows = []
    if prediction_case_ids:
        for case_id in sorted(prediction_case_ids):
            stage_availability_rows.append(f"{case_id}: {'yes' if case_id in covered_prediction_case_ids else 'no'}")

    matched_report_cols = ["file", "file_case_id", "file_district", "file_field_id", "min_date", "max_date", "matched_case_id", "match_reason"]
    rejected_report_cols = ["file", "file_case_id", "file_district", "file_field_id", "min_date", "max_date", "match_reason"]

    report_path = reports_dir / "prediction_2025_satellite_audit.md"
    sections: list[tuple[str, pd.DataFrame | list[str]]] = [
        ("Prediction 2025 summary", [
            f"prediction_2025 cases count: {prediction_count}",
            f"prediction_2025 cases with satellite data: {len(covered_prediction_case_ids)}",
            f"Mehsana 2024-25 satellite status: {'yes' if mehsana_has_satellite else 'no'}",
        ]),
        ("Matched satellite files", matched_df[matched_report_cols] if not matched_df.empty else pd.DataFrame(columns=matched_report_cols)),
        ("Rejected satellite files", rejected_df[rejected_report_cols] if not rejected_df.empty else pd.DataFrame(columns=rejected_report_cols)),
        ("Prediction stage satellite feature availability", stage_availability_rows),
    ]
    _write_markdown_report(report_path, "Prediction 2025 Satellite Audit", sections)

    summary_lines = [
        f"Prediction 2025 satellite coverage: {len(covered_prediction_case_ids)} / {prediction_count} cases",
        f"Mehsana 2025 satellite coverage: {'yes' if mehsana_has_satellite else 'no'}",
    ]
    for line in summary_lines:
        print(line)
        logger.info(line)

    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
