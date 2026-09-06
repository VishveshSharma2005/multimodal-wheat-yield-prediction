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


def fallback_centroids() -> dict:
    return {
        "MEHSANA": (23.5880, 72.3693),
        "AHMEDABAD": (23.0225, 72.5714),
        "GANDHINAGAR": (23.2156, 72.6369),
        "PATAN": (23.8507, 72.1266),
        "BANASKANTHA": (24.1747, 72.4331),
        "SABARKANTHA": (23.6042, 72.9630),
        "SURAT": (21.1702, 72.8311),
        "VADODARA": (22.3072, 73.1812),
        "RAJKOT": (22.3039, 70.8022),
        "JAMNAGAR": (22.4707, 70.0577),
        "JUNAGADH": (21.5222, 70.4579),
        "BHAVNAGAR": (21.7645, 72.1519),
        "KUTCH": (23.7337, 69.8597),
        "KACHCHH": (23.7337, 69.8597),
        "ANAND": (22.5645, 72.9289),
        "KHEDA": (22.7500, 72.7000),
        "AMRELI": (21.6032, 71.2221),
        "ARVALLI": (23.1156, 73.0017),
        "BHARUCH": (21.7051, 72.9959),
        "BOTAD": (22.1692, 71.6663),
        "CHHOTA UDEPUR": (22.3085, 74.0123),
        "DAHOD": (22.8379, 74.2531),
        "DANGS": (20.7577, 73.6869),
        "DEVBHOOMI DWARKA": (22.2380, 69.3430),
        "GIR SOMNATH": (20.9129, 70.3679),
        "MAHISAGAR": (22.8463, 73.6080),
        "MORBI": (22.8119, 70.8236),
        "NARMADA": (21.8690, 73.7125),
        "NAVSARI": (20.9467, 72.9520),
        "PANCHMAHAL": (22.7726, 73.6149),
        "PORBANDAR": (21.6425, 69.6093),
        "SURENDRANAGAR": (22.7271, 71.6480),
        "TAPI": (21.2780, 73.5832),
        "VALSAD": (20.6230, 72.9271),
    }


def _normalize_district(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def main() -> None:
    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    ensure_dir(interim_dir)
    ensure_dir(reports_dir)

    cases_path = interim_dir / "cases.csv"
    if not cases_path.exists():
        raise FileNotFoundError("Missing cases.csv. Run preprocess_all.py first.")

    cases = pd.read_csv(cases_path)
    if cases.empty:
        logger.warning("cases.csv is empty. Coordinate registry not created.")
        return

    field_points_path = raw_dir / "field_points" / "field_coordinates.csv"
    if field_points_path.exists():
        field_points = pd.read_csv(field_points_path)
    else:
        field_points = pd.DataFrame(
            columns=["field_id", "district", "state", "latitude", "longitude", "source", "notes"]
        )

    field_points["district_norm"] = field_points["district"].apply(_normalize_district)

    rows = []
    for _, case in cases.iterrows():
        case_id = case.get("case_id")
        district = case.get("district")
        district_norm = _normalize_district(district)
        season_year_start = case.get("season_year_start")
        season_year_end = case.get("season_year_end")

        if pd.isna(case_id):
            continue

        matches = field_points[field_points["district_norm"] == district_norm]
        if not matches.empty:
            for _, match in matches.iterrows():
                rows.append(
                    {
                        "case_id": case_id,
                        "district": district,
                        "season_year_start": season_year_start,
                        "season_year_end": season_year_end,
                        "field_id": match.get("field_id"),
                        "latitude": match.get("latitude"),
                        "longitude": match.get("longitude"),
                        "coordinate_source": "real_farm_coordinate",
                    }
                )
            continue

        centroid = fallback_centroids().get(district_norm)
        if centroid is not None:
            rows.append(
                {
                    "case_id": case_id,
                    "district": district,
                    "season_year_start": season_year_start,
                    "season_year_end": season_year_end,
                    "field_id": f"{district_norm}_CENTROID",
                    "latitude": centroid[0],
                    "longitude": centroid[1],
                    "coordinate_source": "district_centroid_fallback",
                }
            )
        else:
            rows.append(
                {
                    "case_id": case_id,
                    "district": district,
                    "season_year_start": season_year_start,
                    "season_year_end": season_year_end,
                    "field_id": "",
                    "latitude": pd.NA,
                    "longitude": pd.NA,
                    "coordinate_source": "missing",
                }
            )

    registry = pd.DataFrame(rows)
    out_path = interim_dir / "coordinate_registry.csv"
    registry.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    case_sources = {}
    for case_id, group in registry.groupby("case_id"):
        if (group["coordinate_source"] == "real_farm_coordinate").any():
            case_sources[case_id] = "real_farm_coordinate"
        elif (group["coordinate_source"] == "district_centroid_fallback").any():
            case_sources[case_id] = "district_centroid_fallback"
        else:
            case_sources[case_id] = "missing"

    real_cases = [cid for cid, src in case_sources.items() if src == "real_farm_coordinate"]
    fallback_cases = [cid for cid, src in case_sources.items() if src == "district_centroid_fallback"]
    missing_cases = [cid for cid, src in case_sources.items() if src == "missing"]
    real_districts = sorted(field_points["district"].dropna().unique().tolist())

    report_path = reports_dir / "coordinate_registry_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Coordinate Registry Report\n\n")
        f.write(f"- Total cases: {len(case_sources)}\n")
        f.write(f"- Cases with real farm coordinates: {len(real_cases)}\n")
        f.write(f"- Cases with centroid fallback: {len(fallback_cases)}\n")
        f.write(f"- Cases with missing coordinates: {len(missing_cases)}\n\n")

        f.write("## Districts with real field points\n")
        if real_districts:
            f.write(f"- {', '.join(real_districts)}\n\n")
        else:
            f.write("- None\n\n")

        f.write("## Case coverage by coordinate source\n")
        if real_cases:
            f.write(f"- Real farm coordinates: {', '.join(real_cases)}\n")
        if fallback_cases:
            f.write(f"- Centroid fallback: {', '.join(fallback_cases)}\n")
        if missing_cases:
            f.write(f"- Missing: {', '.join(missing_cases)}\n")

    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
