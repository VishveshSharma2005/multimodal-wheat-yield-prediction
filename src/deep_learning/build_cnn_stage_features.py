from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)


def _parse_vector(text: str) -> list[float]:
    if not isinstance(text, str) or not text:
        return []
    return [float(x) for x in json.loads(text)]


def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    ensure_dir(processed_dir)

    features_path = processed_dir / "cnn_features.csv"
    if not features_path.exists():
        raise FileNotFoundError("Missing cnn_features.csv. Run extract_cnn_features.py first.")

    df = pd.read_csv(features_path)
    if df.empty:
        logger.warning("cnn_features.csv is empty.")
        return

    vectors = df["feature_vector"].apply(_parse_vector)
    vector_lengths = vectors.apply(len)
    if vector_lengths.nunique() != 1:
        raise ValueError("Inconsistent feature vector lengths in cnn_features.csv.")

    feature_count = int(vector_lengths.iloc[0])
    features = np.vstack(vectors.to_list())

    feature_cols = [f"cnn_feat_{i:04d}" for i in range(feature_count)]
    feat_df = pd.DataFrame(features, columns=feature_cols)
    feat_df["stage"] = df["stage"].values

    agg = feat_df.groupby("stage").mean(numeric_only=True).reset_index()
    agg["image_count"] = feat_df.groupby("stage").size().values

    out_path = processed_dir / "cnn_stage_features.csv"
    agg.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
