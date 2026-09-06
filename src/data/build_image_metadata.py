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
    images_dir = Path(config["paths"]["images_dir"])
    out_path = images_dir / "mendeley_image_metadata.csv"
    ensure_dir(images_dir)

    root_abs = ROOT.resolve()
    base = (images_dir / "auxiliary_wheat_images" / "WheatPhenology A Multi-Stage Field Image Dataset o").resolve()
    if not base.exists():
        logger.error("Dataset folder not found: %s", base)
        return

    stage_map = {
        "Seedling & Plant": "early_vegetative",
        "Wheat Flowers": "flowering",
        "Plant with Fruit": "grain_filling_initial",
        "Fruit with Seeds": "grain_filling",
        "Wheat Fruits only": "maturity_preharvest",
    }

    rows = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        # Folder format: "1. Seedling & Plant"
        parts = folder.name.split(".", 1)
        stage_original = parts[1].strip() if len(parts) > 1 else folder.name.strip()
        stage_standard = stage_map.get(stage_original)
        if stage_standard is None:
            logger.warning("Unmapped stage folder: %s", folder.name)
            continue

        for img_path in sorted(folder.glob("*.jpg")):
            img_path = img_path.resolve()
            image_id = f"mendeley_{stage_standard}_{img_path.stem.replace(' ', '_')}"
            rel_path = img_path.relative_to(root_abs).as_posix()
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": rel_path,
                    "source_dataset": "Mendeley_WheatPhenology",
                    "region": "India",
                    "crop": "wheat",
                    "stage_label_original": stage_original,
                    "stage_label_standardized": stage_standard,
                    "image_type": "RGB",
                    "acquisition_year": "",
                    "split_allowed": "yes",
                    "notes": "",
                }
            )

    if not rows:
        logger.warning("No images found under %s", base)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
