from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger
from src.features.image_stage_features import map_stage_label


logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    images_dir = Path(config["paths"]["images_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    ensure_dir(interim_dir)

    metadata_files = list(images_dir.rglob("*.csv"))
    if not metadata_files:
        logger.warning("No image metadata CSVs found in %s", images_dir)
        return

    frames = []
    label_map = config.get("image_stage_label_map", {})

    for path in metadata_files:
        try:
            df = pd.read_csv(path)
            if "stage_label_original" in df.columns:
                df["stage_label_standardized"] = df["stage_label_original"].apply(
                    lambda x: map_stage_label(x, label_map)
                )
            df["source_file"] = path.name
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)

    if not frames:
        logger.warning("No image metadata loaded.")
        return

    out_df = pd.concat(frames, ignore_index=True)
    out_path = interim_dir / "image_metadata.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
