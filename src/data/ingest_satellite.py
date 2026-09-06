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


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "date": "date",
        "ndvi": "ndvi",
        "ndre": "ndre",
        "evi": "evi",
        "cloud_pct": "cloud_pct",
        "cloud_percentage": "cloud_pct",
        "field": "field_id",
        "fieldid": "field_id",
        "field_id": "field_id",
        "district": "district",
    }
    cols = {c: rename_map.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    if "ndvi" in df.columns and "ndvi_raw" not in df.columns:
        df["ndvi_raw"] = df["ndvi"]
    for col in ["case_id", "field_id", "district", "date", "ndvi", "ndvi_raw", "ndre", "evi", "cloud_pct"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


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
    return pd.read_csv(registry_path)


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

    year_tokens = [idx for idx, token in enumerate(tokens) if re.fullmatch(r"20\d{2}", token)]
    if tokens and tokens[0].upper() == "GUJ" and len(year_tokens) >= 2:
        case_tokens = tokens[: year_tokens[0]]
        if len(case_tokens) >= 2:
            metadata["case_id"] = "_".join(token.upper() for token in case_tokens)
            district_tokens = case_tokens[1:]
            if district_tokens:
                metadata["district"] = " ".join(district_tokens).replace("_", " ")
    elif tokens:
        district_tokens = []
        for token in tokens:
            token_upper = token.upper()
            if token_upper.startswith("FIELD") or re.fullmatch(r"20\d{2}", token_upper):
                break
            district_tokens.append(token)
        if district_tokens:
            metadata["district"] = " ".join(district_tokens).replace("_", " ")

    for token in tokens:
        token_upper = token.upper()
        if token_upper.startswith("FIELD"):
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


def _load_case_lookup(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return cases
    lookup = cases.copy()
    for col in ["case_id", "district", "field_id"]:
        if col in lookup.columns:
            lookup[f"{col}_norm"] = lookup[col].apply(_normalize_text)
    if "season_year_start" in lookup.columns:
        lookup["season_year_start"] = pd.to_numeric(lookup["season_year_start"], errors="coerce")
    if "sowing_date" in lookup.columns:
        lookup["sowing_date"] = pd.to_datetime(lookup["sowing_date"], errors="coerce")
    if "harvest_date" in lookup.columns:
        lookup["harvest_date"] = pd.to_datetime(lookup["harvest_date"], errors="coerce")
    return lookup


def _file_matches_case(file_start, file_end, case_row: pd.Series) -> bool:
    sow = case_row.get("sowing_date")
    harv = case_row.get("harvest_date")
    if pd.notna(sow) and pd.notna(harv):
        return _overlaps(file_start, file_end, sow, harv)
    case_season_start = case_row.get("season_year_start")
    if pd.notna(case_season_start):
        season_start, season_end = _season_window(int(case_season_start))
        return _overlaps(file_start, file_end, season_start, season_end)
    return False


def _match_satellite_file(
    df: pd.DataFrame,
    file_name: str,
    cases: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[str | None, str, str, str, str, pd.Timestamp | None, pd.Timestamp | None]:
    meta = _parse_filename_metadata(file_name)
    work = df.copy()
    work["date"] = pd.to_datetime(work.get("date"), errors="coerce")
    file_start = work["date"].min()
    file_end = work["date"].max()
    file_season_start = _season_year_start_for_date(file_start) if pd.notna(file_start) else None

    file_case_id = ""
    if "case_id" in work.columns and work["case_id"].notna().any():
        file_case_id = _normalize_text(work["case_id"].dropna().iloc[0])
    elif meta["case_id"]:
        file_case_id = _normalize_text(meta["case_id"])

    file_district = ""
    if "district" in work.columns and work["district"].notna().any():
        file_district = _prefer_filename_metadata(work["district"].dropna().iloc[0], meta["district"])
    elif meta["district"]:
        file_district = _normalize_text(meta["district"])

    file_field_id = ""
    if "field_id" in work.columns and work["field_id"].notna().any():
        file_field_id = _normalize_text(work["field_id"].dropna().iloc[0])
    elif meta["field_id"]:
        file_field_id = _normalize_text(meta["field_id"])

    if pd.isna(file_start) or pd.isna(file_end):
        return None, "missing date range", file_case_id, file_district, file_field_id, file_start, file_end

    if not cases.empty:
        if file_case_id and "case_id_norm" in cases.columns:
            exact = cases[cases["case_id_norm"] == file_case_id]
            if not exact.empty:
                case_row = exact.iloc[0]
                if _file_matches_case(file_start, file_end, case_row):
                    return str(case_row.get("case_id")), "exact case_id match", file_case_id, file_district, file_field_id, file_start, file_end
                return None, "exact case_id found but no crop-season overlap", file_case_id, file_district, file_field_id, file_start, file_end

        if file_district and "district_norm" in cases.columns:
            district_cases = cases[cases["district_norm"] == file_district].copy()
            if not district_cases.empty:
                if file_season_start is not None and "season_year_start" in district_cases.columns:
                    season_cases = district_cases[district_cases["season_year_start"] == file_season_start]
                    for _, case_row in season_cases.iterrows():
                        if _file_matches_case(file_start, file_end, case_row):
                            return str(case_row.get("case_id")), "district + same season_year_start + overlap", file_case_id, file_district, file_field_id, file_start, file_end

                if file_field_id and not registry.empty and {"district", "field_id", "case_id"}.issubset(registry.columns):
                    reg = registry.copy()
                    reg["district_norm"] = reg["district"].apply(_normalize_text)
                    reg["field_id_norm"] = reg["field_id"].apply(_normalize_text)
                    registry_candidates = reg[(reg["district_norm"] == file_district) & (reg["field_id_norm"] == file_field_id)]
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


def _normalize(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    if pd.isna(a_start) or pd.isna(a_end) or pd.isna(b_start) or pd.isna(b_end):
        return False
    return a_start <= b_end and b_start <= a_end


def _build_registry_lookup(registry: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in registry.iterrows():
        district = _normalize(row.get("district"))
        field_id = _normalize(row.get("field_id"))
        case_id = row.get("case_id")
        if not district or not field_id or pd.isna(case_id):
            continue
        lookup.setdefault((district, field_id), set()).add(case_id)
    return lookup


def _match_case_id(
    sat_case_id: str,
    field_id: str,
    district: str,
    sat_start,
    sat_end,
    cases: pd.DataFrame,
    registry_lookup: dict,
) -> tuple[str | None, str]:
    if cases.empty:
        return None, "cases.csv missing"
    if isinstance(sat_case_id, str) and sat_case_id in cases["case_id"].values:
        return sat_case_id, "exact case_id match"

    district_norm = _normalize(district)
    field_norm = _normalize(field_id)
    if district_norm and field_norm:
        candidates = registry_lookup.get((district_norm, field_norm), set())
        for case_id in candidates:
            row = cases[cases["case_id"] == case_id]
            if row.empty:
                continue
            sowing = row.iloc[0].get("sowing_date")
            harvest = row.iloc[0].get("harvest_date")
            if _overlaps(sat_start, sat_end, sowing, harvest):
                return case_id, "field_id + district match with date overlap"
        return None, "field_id + district not in date range"

    return None, "missing case_id/field_id/district"


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


def _satellite_feature_flags(df: pd.DataFrame) -> dict[str, bool]:
    cols = [col for col in ["ndvi", "ndvi_raw", "ndre", "evi"] if col in df.columns]
    if not cols:
        return {"ndvi": False, "ndre": False, "evi": False}
    return {
        "ndvi": bool(df[[col for col in ["ndvi", "ndvi_raw"] if col in df.columns]].notna().any().any()),
        "ndre": bool(df[["ndre"]].notna().any().any()) if "ndre" in df.columns else False,
        "evi": bool(df[["evi"]].notna().any().any()) if "evi" in df.columns else False,
    }


def main() -> None:
    config = load_config()
    satellite_dir = Path(config["paths"]["satellite_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    ensure_dir(interim_dir)
    ensure_dir(reports_dir)

    cases = _load_case_lookup(_load_cases(interim_dir))
    registry = _load_registry(interim_dir)
    registry_lookup = _build_registry_lookup(registry)

    csv_files = sorted(satellite_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No satellite CSVs found in %s", satellite_dir)
        return

    frames = []
    report_rows = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            df = standardize_columns(df)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"].notna()].copy()
            df["source_file"] = path.name
            matched_case_id, reason, file_case_id, file_district, file_field_id, file_start, file_end = _match_satellite_file(
                df,
                path.name,
                cases,
                registry,
            )

            report_rows.append(
                {
                    "file": path.name,
                    "file_case_id": file_case_id,
                    "file_district": file_district,
                    "file_field_id": file_field_id,
                    "min_date": str(file_start.date()) if pd.notna(file_start) else "",
                    "max_date": str(file_end.date()) if pd.notna(file_end) else "",
                    "matched_case_id": matched_case_id or "",
                    "match_reason": reason,
                    "status": "matched" if matched_case_id else "rejected",
                }
            )

            if not matched_case_id:
                continue

            df = df.copy()
            df["source_case_id"] = file_case_id or matched_case_id
            df["matched_case_id"] = matched_case_id
            df["match_reason"] = reason
            df["case_id"] = matched_case_id
            if not file_district and not cases.empty and "district" in cases.columns:
                match = cases[cases["case_id"] == matched_case_id]
                if not match.empty:
                    df["district"] = match.iloc[0].get("district")
            if not file_field_id and "field_id" in df.columns:
                df["field_id"] = df["field_id"].fillna("")
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)

    if not frames:
        out_df = pd.DataFrame()
    else:
        out_df = pd.concat(frames, ignore_index=True)

    out_path = interim_dir / "satellite_observations.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    report_df = pd.DataFrame(report_rows)
    report_path = reports_dir / "satellite_ingestion_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Satellite Ingestion Report\n\n")
        if report_df.empty:
            f.write("- No satellite rows processed.\n")
        else:
            try:
                f.write(report_df.to_markdown(index=False))
            except Exception:
                f.write(report_df.to_string(index=False))
            f.write("\n\n")

        if not cases.empty and not out_df.empty:
            cases_for_summary = cases.copy()
            if "season_year_start" in cases_for_summary.columns:
                cases_for_summary["split"] = cases_for_summary["season_year_start"].apply(_split_label)
            elif "season_year" in cases_for_summary.columns:
                cases_for_summary["split"] = cases_for_summary["season_year"].apply(_split_label)
            else:
                cases_for_summary["split"] = "unknown"

            sat_case_ids = set(out_df["case_id"].dropna().astype(str).unique().tolist())
            f.write("## Satellite coverage by split\n")
            for split, group in cases_for_summary.groupby("split"):
                case_ids = set(group["case_id"].dropna().astype(str).unique().tolist())
                with_sat = case_ids.intersection(sat_case_ids)
                coverage_pct = (len(with_sat) / len(case_ids) * 100.0) if case_ids else 0.0
                f.write(f"- {split}: {len(with_sat)}/{len(case_ids)} cases with satellite ({coverage_pct:.2f}%)\n")

            f.write("\n## Satellite coverage by district\n")
            if "district" in cases_for_summary.columns:
                for district_name, group in cases_for_summary.groupby("district"):
                    case_ids = set(group["case_id"].dropna().astype(str).unique().tolist())
                    with_sat = case_ids.intersection(sat_case_ids)
                    coverage_pct = (len(with_sat) / len(case_ids) * 100.0) if case_ids else 0.0
                    f.write(f"- {district_name}: {len(with_sat)}/{len(case_ids)} cases with satellite ({coverage_pct:.2f}%)\n")

            mehsana_cases = cases_for_summary[cases_for_summary["district"].astype(str).str.contains("MEHSANA", case=False, na=False)] if "district" in cases_for_summary.columns else pd.DataFrame()
            mehsana_case_ids = set(mehsana_cases["case_id"].dropna().astype(str).unique().tolist()) if not mehsana_cases.empty else set()
            mehsana_sat = mehsana_case_ids.intersection(sat_case_ids)
            f.write("\n## Mehsana coverage\n")
            if mehsana_case_ids:
                f.write(f"- Cases with satellite: {len(mehsana_sat)}/{len(mehsana_case_ids)}\n")
            else:
                f.write("- No Mehsana cases found.\n")

    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
