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


def _stage_label_map() -> dict:
    return {
        "Seedling & Plant": "early_vegetative",
        "Wheat Flowers": "flowering",
        "Plant with Fruit": "grain_filling_initial",
        "Fruit with Seeds": "grain_filling",
        "Wheat Fruits only": "maturity_preharvest",
    }


def _normalize_stage(label: str) -> str:
    if label is None:
        return ""
    label = str(label).strip()
    mapping = _stage_label_map()
    return mapping.get(label, label)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    ensure_dir(processed_dir)

    features_path = processed_dir / "image_features.csv"
    meta_path = interim_dir / "image_metadata.csv"

    if features_path.exists():
        df = pd.read_csv(features_path)
    elif meta_path.exists():
        df = pd.read_csv(meta_path)
    else:
        logger.warning("No image metadata or features available.")
        return

    if df.empty:
        logger.warning("Image data is empty.")
        return

    if "stage_label_standardized" not in df.columns and "stage_label_original" in df.columns:
        df["stage_label_standardized"] = df["stage_label_original"].map(_normalize_stage)

    if "stage_label_standardized" not in df.columns:
        logger.warning("No stage label column found in image metadata.")
        return

    df["stage"] = df["stage_label_standardized"].astype(str)

    agg_cols = [
        "green_pixel_ratio",
        "canopy_density_proxy",
        "vegetation_pixel_ratio",
        "brightness_mean",
        "saturation_mean",
        "texture_proxy",
    ]

    available_cols = [c for c in agg_cols if c in df.columns]
    if not available_cols:
        logger.warning("No numeric image features available for aggregation.")
        return

    has_case = "case_id" in df.columns and df["case_id"].notna().any()
    group_cols = ["stage"]
    if has_case:
        group_cols.insert(0, "case_id")

    grouped = df.groupby(group_cols)
    agg = grouped[available_cols].mean().reset_index()
    agg["image_count_stage"] = grouped.size().values

    rename_map = {
        "green_pixel_ratio": "image_green_pixel_ratio_stage_avg",
        "canopy_density_proxy": "image_canopy_density_proxy_stage_avg",
        "vegetation_pixel_ratio": "image_vegetation_pixel_ratio_stage_avg",
        "brightness_mean": "image_brightness_mean_stage_avg",
        "saturation_mean": "image_saturation_mean_stage_avg",
        "texture_proxy": "image_texture_proxy_stage_avg",
    }
    agg = agg.rename(columns=rename_map)

    agg["image_coverage_flag"] = 1
    agg["image_feature_source"] = "auxiliary_external_image_dataset"

    out_path = processed_dir / "image_stage_features.csv"
    agg.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(agg))


if __name__ == "__main__":
    main()
