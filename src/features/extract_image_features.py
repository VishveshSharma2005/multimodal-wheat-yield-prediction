from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_image(path: Path):
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("Pillow is required for image feature extraction. Install with pip install pillow")

    return Image.open(path).convert("RGB")


def _compute_features(img) -> dict:
    import numpy as np

    img_small = img.resize((256, 256))
    arr = np.asarray(img_small).astype(np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green_mask = (g > r) & (g > b) & (g > 40)
    green_ratio = float(np.mean(green_mask))

    hsv = img_small.convert("HSV")
    hsv_arr = np.asarray(hsv).astype(np.float32)
    h = hsv_arr[:, :, 0]
    s = hsv_arr[:, :, 1]
    v = hsv_arr[:, :, 2]

    hue_min = 42.0
    hue_max = 127.0
    veg_mask = (h >= hue_min) & (h <= hue_max) & (s > 50) & (v > 50)
    veg_ratio = float(np.mean(veg_mask))

    brightness_mean = float(np.mean(v))
    saturation_mean = float(np.mean(s))

    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    texture_proxy = float(np.std(gray))

    canopy_density_proxy = float(veg_ratio * (1.0 - brightness_mean / 255.0))

    return {
        "green_pixel_ratio": green_ratio,
        "canopy_density_proxy": canopy_density_proxy,
        "vegetation_pixel_ratio": veg_ratio,
        "brightness_mean": brightness_mean,
        "saturation_mean": saturation_mean,
        "texture_proxy": texture_proxy,
    }


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    ensure_dir(processed_dir)

    meta_path = interim_dir / "image_metadata.csv"
    if not meta_path.exists():
        logger.warning("Missing image_metadata.csv in %s", interim_dir)
        return

    df = pd.read_csv(meta_path)
    if df.empty:
        logger.warning("image_metadata.csv is empty")
        return

    rows = []
    for _, row in df.iterrows():
        img_path = row.get("image_path")
        if not isinstance(img_path, str) or not img_path:
            continue
        full_path = ROOT / img_path
        if not full_path.exists():
            logger.warning("Image not found: %s", full_path)
            continue

        try:
            img = _load_image(full_path)
            feats = _compute_features(img)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", full_path, exc)
            continue

        out_row = {
            "image_id": row.get("image_id"),
            "image_path": img_path,
            "stage_label_standardized": row.get("stage_label_standardized") or row.get("stage"),
            "case_id": row.get("case_id"),
            "district": row.get("district"),
            "source_dataset": row.get("source_dataset"),
        }
        out_row.update(feats)
        rows.append(out_row)

    if not rows:
        logger.warning("No image features extracted.")
        return

    out_df = pd.DataFrame(rows)
    out_path = processed_dir / "image_features.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(out_df))


if __name__ == "__main__":
    main()
