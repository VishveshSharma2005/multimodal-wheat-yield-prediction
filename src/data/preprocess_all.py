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


def concat_csvs(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
            df["source_file"] = path.name
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    templates_dir = Path(config["paths"]["templates_dir"])
    farm_logs_dir = Path(config["paths"]["farm_logs_dir"])
    drone_dir = Path(config["paths"]["drone_dir"])
    soil_dir = Path(config["paths"]["soil_dir"])

    ensure_dir(interim_dir)

    apy_all_path = interim_dir / "apy_yield_all_districts.csv"
    if apy_all_path.exists():
        apy_all = pd.read_csv(apy_all_path)

        centroid_map = {
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
            "ANAND": (22.5645, 72.9289),
            "KHEDA": (22.7500, 72.7000),
        }

        cases_rows = []
        for _, row in apy_all.iterrows():
            district = str(row.get("district", "")).strip()
            if not district:
                continue
            district_key = district.upper()
            lat, lon = centroid_map.get(district_key, (None, None))

            year_start = str(row.get("season_year_start", "")).strip()
            year_end = str(row.get("season_year_end", "")).strip()
            if not year_start or not year_end:
                continue

            case_id = f"GUJ_{district_key.replace(' ', '_')}_{year_start}_{year_end}"
            sowing_date = f"{year_start}-11-20"
            harvest_date = f"{year_end}-03-25"

            cases_rows.append(
                {
                    "case_id": case_id,
                    "district": district,
                    "state": "Gujarat",
                    "crop": "wheat",
                    "region": "Gujarat",
                    "season_year": int(year_start),
                    "season_year_start": int(year_start),
                    "season_year_end": int(year_end),
                    "lat": lat,
                    "lon": lon,
                    "area_ha": None,
                    "sowing_date": sowing_date,
                    "harvest_date": harvest_date,
                    "yield_kg_ha": row.get("yield_kg_ha"),
                }
            )

        cases = pd.DataFrame(cases_rows)
        cases.to_csv(interim_dir / "cases.csv", index=False)
        logger.info("Wrote cases.csv from apy_yield_all_districts.csv")
    else:
        registry_candidates = [
            templates_dir / "field_registry.csv",
            templates_dir / "field_registry_template.csv",
        ]
        registry_path = next((p for p in registry_candidates if p.exists()), None)
        if registry_path is None:
            raise FileNotFoundError("Field registry CSV not found.")

        cases = pd.read_csv(registry_path)
        cases.to_csv(interim_dir / "cases.csv", index=False)
        logger.info("Wrote cases.csv")

    farm_logs = concat_csvs(farm_logs_dir)
    if not farm_logs.empty:
        farm_logs.to_csv(interim_dir / "farm_logs.csv", index=False)
        logger.info("Wrote farm_logs.csv")

    drone_obs = concat_csvs(drone_dir)
    if not drone_obs.empty:
        drone_obs.to_csv(interim_dir / "drone_observations.csv", index=False)
        logger.info("Wrote drone_observations.csv")

    soil = concat_csvs(soil_dir)
    if not soil.empty:
        soil.to_csv(interim_dir / "soil.csv", index=False)
        logger.info("Wrote soil.csv")


if __name__ == "__main__":
    main()
