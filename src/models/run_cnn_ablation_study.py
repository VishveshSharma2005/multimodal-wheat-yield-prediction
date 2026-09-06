from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.features.feature_policy import is_excluded
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_stage_features(processed_dir: Path) -> tuple[pd.DataFrame, Path]:
    stage_path = processed_dir / "stage_features_with_cnn.csv"
    if not stage_path.exists():
        stage_path = processed_dir / "stage_features_with_split.csv"
    if not stage_path.exists():
        raise FileNotFoundError("Missing stage_features_with_cnn.csv or stage_features_with_split.csv.")
    return pd.read_csv(stage_path), stage_path


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": _safe_mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _feature_group(col: str) -> str:
    col_lower = col.lower()
    if col_lower.startswith("cnn_") or "cnn_feat" in col_lower:
        return "cnn"
    if any(k in col_lower for k in ["rain", "precip", "tmin", "tmax", "gdd", "temp", "humidity", "radiation", "wind"]):
        return "weather"
    if any(k in col_lower for k in ["ndvi", "ndre", "evi", "cloud", "satellite"]):
        return "satellite"
    if any(k in col_lower for k in ["image_", "crop_cover", "weed", "texture", "vegetation_pixel", "canopy_density"]):
        return "image"
    return "other"


def _candidate_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns permitted as model inputs by the shared feature policy."""
    return [
        col
        for col in df.columns
        if (not is_excluded(col)) and pd.api.types.is_numeric_dtype(df[col])
    ]


def _columns_for_groups(df: pd.DataFrame, groups: set[str]) -> list[str]:
    return [
        col
        for col in _candidate_numeric_columns(df)
        if _feature_group(col) in groups and float(df[col].isna().mean()) <= 0.90
    ]


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    ensure_dir(reports_dir)

    df, stage_path = _load_stage_features(processed_dir)
    if "split" not in df.columns or "yield_kg_ha" not in df.columns:
        raise ValueError("Ablation study requires split and yield_kg_ha columns.")

    train_df = df[df["split"] == "train"].dropna(subset=["yield_kg_ha"]).copy()
    val_df = df[df["split"] == "validation"].dropna(subset=["yield_kg_ha"]).copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Ablation study requires non-empty train and validation rows.")

    ablations = [
        ("Weather Only", {"weather"}),
        ("Weather + Satellite", {"weather", "satellite"}),
        ("Weather + Satellite + Handcrafted Images", {"weather", "satellite", "image"}),
        ("Weather + Satellite + Handcrafted Images + CNN", {"weather", "satellite", "image", "cnn"}),
    ]

    rows = []
    for name, groups in ablations:
        feature_cols = _columns_for_groups(df, groups)
        if not feature_cols:
            rows.append(
                {
                    "ablation": name,
                    "feature_count": 0,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE": np.nan,
                    "R2": np.nan,
                    "status": "skipped",
                    "message": "No feature columns available.",
                    "dataset": str(stage_path),
                }
            )
            continue

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(random_state=42)),
            ]
        )
        model.fit(train_df[feature_cols], train_df["yield_kg_ha"].to_numpy(dtype=float))
        preds = model.predict(val_df[feature_cols])
        rows.append(
            {
                "ablation": name,
                "feature_count": len(feature_cols),
                **_metrics(val_df["yield_kg_ha"].to_numpy(dtype=float), preds),
                "status": "ok",
                "message": "",
                "dataset": str(stage_path),
            }
        )

    out_path = reports_dir / "cnn_ablation_study.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
