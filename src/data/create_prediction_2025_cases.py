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


def _pick_lat_lon(cases: pd.DataFrame, district: str) -> tuple[float | None, float | None]:
    subset = cases[cases["district"].str.contains(district, case=False, na=False)]
    if subset.empty:
        return None, None
    lat = subset["lat"].dropna().iloc[0] if "lat" in subset.columns and subset["lat"].notna().any() else None
    lon = subset["lon"].dropna().iloc[0] if "lon" in subset.columns and subset["lon"].notna().any() else None
    return lat, lon


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    ensure_dir(interim_dir)

    cases_path = interim_dir / "cases.csv"
    if not cases_path.exists():
        raise FileNotFoundError("Missing cases.csv. Run preprocess_all.py first.")

    cases = pd.read_csv(cases_path)
    if cases.empty:
        logger.warning("cases.csv is empty. No prediction cases created.")
        return

    districts = sorted(cases["district"].dropna().unique().tolist()) if "district" in cases.columns else []
    if not districts:
        logger.warning("No districts found in cases.csv. No prediction cases created.")
        return

    target_start = 2024
    target_end = 2025

    new_rows = []
    for district in districts:
        district_key = str(district).strip().upper().replace(" ", "_")
        case_id = f"GUJ_{district_key}_{target_start}_{target_end}"
        if (cases["case_id"] == case_id).any():
            continue

        lat, lon = _pick_lat_lon(cases, district)
        new_rows.append(
            {
                "case_id": case_id,
                "district": district,
                "state": "Gujarat",
                "crop": "wheat",
                "region": "Gujarat",
                "season_year": target_start,
                "season_year_start": target_start,
                "season_year_end": target_end,
                "lat": lat,
                "lon": lon,
                "area_ha": None,
                "sowing_date": f"{target_start}-11-20",
                "harvest_date": f"{target_end}-03-25",
                "yield_kg_ha": None,
            }
        )

    if not new_rows:
        logger.info("Prediction cases already present for %s-%s.", target_start, target_end)
        return

    out_df = pd.concat([cases, pd.DataFrame(new_rows)], ignore_index=True)
    out_df.to_csv(cases_path, index=False)
    logger.info("Added %d prediction cases for %s-%s", len(new_rows), target_start, target_end)


if __name__ == "__main__":
    main()
