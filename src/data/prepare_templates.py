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


def write_template(path: Path, columns: list) -> None:
    ensure_dir(path.parent)
    df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)
    logger.info("Wrote template: %s", path)


def main() -> None:
    config = load_config()
    templates_dir = Path(config["paths"]["templates_dir"])

    write_template(
        templates_dir / "field_registry_template.csv",
        [
            "case_id",
            "farm_id",
            "field_id",
            "season_year",
            "crop",
            "region",
            "lat",
            "lon",
            "area_ha",
            "sowing_date",
            "harvest_date",
            "yield_kg_ha",
        ],
    )

    write_template(
        templates_dir / "drone_observations_template.csv",
        [
            "case_id",
            "date",
            "image_id",
            "canopy_cover_pct",
            "weed_cover_pct",
            "crop_cover_pct",
            "ndvi_raw",
            "ndvi_crop_only",
            "canopy_temp_c",
            "crop_stage_observed",
            "notes",
        ],
    )

    write_template(
        templates_dir / "farm_visit_logs_template.csv",
        [
            "case_id",
            "date",
            "irrigation_event",
            "irrigation_mm",
            "fertilizer_n_kg_ha",
            "fertilizer_p_kg_ha",
            "fertilizer_k_kg_ha",
            "weed_control_event",
            "pest_score_0_5",
            "disease_score_0_5",
            "crop_height_cm",
            "crop_stage_observed",
            "notes",
        ],
    )

    write_template(
        templates_dir / "soil_template.csv",
        [
            "case_id",
            "soil_ph",
            "ec",
            "organic_carbon",
            "n_kg_ha",
            "p_kg_ha",
            "k_kg_ha",
            "texture",
            "water_holding_capacity",
            "notes",
        ],
    )


if __name__ == "__main__":
    main()
