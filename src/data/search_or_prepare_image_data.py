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


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    images_dir = Path(config["paths"]["images_dir"])

    ensure_dir(interim_dir)
    ensure_dir(reports_dir)
    ensure_dir(images_dir)

    out_path = interim_dir / "image_metadata.csv"
    if out_path.exists():
        logger.info("Image metadata already exists: %s", out_path)
        return

    columns = [
        "image_id",
        "image_path",
        "source_dataset",
        "region",
        "crop",
        "stage_label_original",
        "stage_label_standardized",
        "image_type",
        "acquisition_year",
        "split_allowed",
        "notes",
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(out_path, index=False)
    logger.info("Wrote empty image metadata template: %s", out_path)


if __name__ == "__main__":
    main()
