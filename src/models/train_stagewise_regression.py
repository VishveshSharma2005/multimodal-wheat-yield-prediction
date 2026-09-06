from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.features.feature_policy import is_excluded
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _safe_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _compute_split(series: pd.Series) -> pd.Series:
    def _split(year):
        if pd.isna(year):
            return "unknown"
        year = int(year)
        if year in {2020, 2021, 2022}:
            return "train"
        if year == 2023:
            return "validation"
        if year == 2024:
            return "prediction_2025"
        return "unknown"

    return series.apply(_split)


def _is_excluded(col: str) -> bool:
    """Delegate to the shared policy in src/features/feature_policy.py.

    Excludes identifiers, split keys, free text, target-derived columns
    (including `actual_available`) and exact-duplicate aliases.
    """
    return is_excluded(col)


def _missing_pct(series: pd.Series) -> float:
    return float(series.isna().mean() * 100.0)


def _feature_group(col: str) -> str:
    col_lower = col.lower()
    if col_lower.startswith("cnn_") or "cnn_feat" in col_lower:
        return "cnn"
    if any(k in col_lower for k in ["crop_name", "season_type"]):
        return "crop/meta"
    if any(k in col_lower for k in ["rain", "precip", "tmin", "tmax", "gdd", "temp", "humidity", "radiation", "wind"]):
        return "weather"
    if any(k in col_lower for k in ["ndvi", "ndre", "evi", "cloud", "canopy", "satellite"]):
        return "satellite"
    if any(k in col_lower for k in ["image_", "crop_cover", "weed", "texture", "vegetation_pixel", "canopy_density"]):
        return "image"
    if any(k in col_lower for k in ["stage", "das", "date", "index"]):
        return "stage/time"
    if any(k in col_lower for k in ["district", "lat", "lon", "area"]):
        return "location"
    return "other"


def _select_features(df: pd.DataFrame, include_stage: bool) -> tuple[list[str], list[str], list[dict]]:
    categorical_cols = []
    if "district" in df.columns:
        categorical_cols.append("district")
    if include_stage and "stage" in df.columns:
        categorical_cols.append("stage")
    for col in ["crop_name", "season_type"]:
        if col in df.columns:
            categorical_cols.append(col)

    numeric_cols = []
    dropped = []
    for col in df.columns:
        if _is_excluded(col) or col in categorical_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if _missing_pct(df[col]) > 90.0:
                dropped.append({"feature": col, "reason": "missing > 90%"})
            else:
                numeric_cols.append(col)

    return numeric_cols, categorical_cols, dropped


def _build_pipeline(model, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _safe_onehot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "mape": mape,
        "r2": float(r2),
    }


def _sanitize_stage(stage: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stage)


def _extract_feature_importance(pipeline: Pipeline, out_path: Path) -> bool:
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocess")
    if model is None or preprocessor is None:
        return False
    if not hasattr(model, "feature_importances_"):
        return False
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return False

    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return False

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(out_path, index=False)
    return True


def _load_dataset() -> tuple[pd.DataFrame, Path, Path, Path, Path]:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    models_dir = ROOT / "models" / "regression"

    ensure_dir(reports_dir)
    ensure_dir(models_dir)

    stage_path = processed_dir / "stage_features_with_cnn.csv"
    if not stage_path.exists():
        stage_path = processed_dir / "stage_features_with_split.csv"
    if not stage_path.exists():
        raise FileNotFoundError("Missing stage_features_with_cnn.csv or stage_features_with_split.csv. Run build_stage_table.py first.")

    df = pd.read_csv(stage_path)

    cases_path = interim_dir / "cases.csv"
    if cases_path.exists() and "district" not in df.columns:
        cases = pd.read_csv(cases_path)
        df = df.merge(cases[["case_id", "district"]], on="case_id", how="left")

    targets_path = processed_dir / "targets.csv"
    if targets_path.exists() and "yield_kg_ha" not in df.columns:
        targets = pd.read_csv(targets_path)
        df = df.merge(targets[["case_id", "yield_kg_ha", "yield_quintile"]], on="case_id", how="left")

    if "split" not in df.columns or df["split"].isna().all():
        if "season_year_start" not in df.columns:
            raise ValueError("Missing split and season_year_start columns.")
        df["split"] = _compute_split(df["season_year_start"])

    return df, stage_path, reports_dir, models_dir, processed_dir


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    df, stage_path, reports_dir, models_dir, _ = _load_dataset()

    if "yield_kg_ha" not in df.columns:
        raise ValueError("Missing yield_kg_ha after merging targets.")

    df = df[df["split"].isin(["train", "validation", "prediction_2025"])].copy()

    model_candidates = [
        ("RandomForestRegressor", RandomForestRegressor(random_state=42)),
        ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42)),
        ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(random_state=42)),
    ]

    optional_missing = []
    try:
        from lightgbm import LGBMRegressor

        model_candidates.append(
            (
                "LightGBMRegressor",
                LGBMRegressor(
                    random_state=42,
                    n_estimators=200,
                    min_data_in_leaf=5,
                    min_data_in_bin=1,
                    num_leaves=31,
                    verbose=-1,
                ),
            )
        )
    except Exception:
        optional_missing.append("LightGBMRegressor (lightgbm not installed)")

    try:
        from xgboost import XGBRegressor

        model_candidates.append(
            (
                "XGBoostRegressor",
                XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror",
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    n_jobs=-1,
                    verbosity=0,
                ),
            )
        )
    except Exception:
        optional_missing.append("XGBoostRegressor (xgboost not installed)")

    stages = sorted(df["stage"].dropna().unique().tolist()) if "stage" in df.columns else []
    stage_metrics = []
    best_models = {}
    skipped_stages = {}
    selected_rows = []

    for stage in stages:
        stage_df = df[df["stage"] == stage].copy()
        train_df = stage_df[stage_df["split"] == "train"].copy()
        val_df = stage_df[stage_df["split"] == "validation"].copy()

        if train_df.empty:
            skipped_stages[stage] = "No training rows"
            continue
        if val_df.empty:
            skipped_stages[stage] = "No validation rows"
            continue

        numeric_cols, categorical_cols, dropped_cols = _select_features(stage_df, include_stage=False)
        if not numeric_cols and not categorical_cols:
            skipped_stages[stage] = "No usable feature columns"
            continue

        for col in numeric_cols:
            selected_rows.append(
                {
                    "stage": stage,
                    "feature": col,
                    "feature_type": "numeric",
                    "feature_group": _feature_group(col),
                    "status": "selected",
                    "reason": "numeric",
                }
            )
        for col in categorical_cols:
            selected_rows.append(
                {
                    "stage": stage,
                    "feature": col,
                    "feature_type": "categorical",
                    "feature_group": _feature_group(col),
                    "status": "selected",
                    "reason": "categorical",
                }
            )
        for item in dropped_cols:
            selected_rows.append(
                {
                    "stage": stage,
                    "feature": item["feature"],
                    "feature_type": "numeric",
                    "feature_group": _feature_group(item["feature"]),
                    "status": "dropped",
                    "reason": item["reason"],
                }
            )

        X_train = train_df[numeric_cols + categorical_cols]
        y_train = train_df["yield_kg_ha"].values
        X_val = val_df[numeric_cols + categorical_cols]
        y_val = val_df["yield_kg_ha"].values

        best_rmse = np.inf
        best_pipeline = None
        best_model_name = None

        for model_name, model in model_candidates:
            pipeline = _build_pipeline(model, numeric_cols, categorical_cols)
            try:
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_val)
                metrics = _evaluate(y_val, preds)
                stage_metrics.append(
                    {
                        "stage": stage,
                        "model": model_name,
                        "n_train": len(train_df),
                        "n_val": len(val_df),
                        **metrics,
                        "status": "ok",
                        "message": "",
                    }
                )
                if metrics["rmse"] < best_rmse:
                    best_rmse = metrics["rmse"]
                    best_pipeline = pipeline
                    best_model_name = model_name
            except Exception as exc:
                stage_metrics.append(
                    {
                        "stage": stage,
                        "model": model_name,
                        "n_train": len(train_df),
                        "n_val": len(val_df),
                        "mae": np.nan,
                        "mse": np.nan,
                        "rmse": np.nan,
                        "mape": np.nan,
                        "r2": np.nan,
                        "status": "failed",
                        "message": str(exc),
                    }
                )
                logger.warning("Stage %s model %s failed: %s", stage, model_name, exc)

        if best_pipeline is None:
            skipped_stages[stage] = "All models failed"
            continue

        model_path = models_dir / f"best_model_{_sanitize_stage(stage)}.joblib"
        dump(best_pipeline, model_path)
        best_models[stage] = best_model_name

        fi_path = reports_dir / f"feature_importance_{_sanitize_stage(stage)}.csv"
        if not _extract_feature_importance(best_pipeline, fi_path):
            if fi_path.exists():
                fi_path.unlink(missing_ok=True)

    metrics_df = pd.DataFrame(stage_metrics)
    metrics_path = reports_dir / "stagewise_regression_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary_path = reports_dir / "stagewise_regression_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Stage-wise Regression Summary\n\n")
        f.write(f"- Dataset: {stage_path}\n")
        f.write(f"- Total rows: {len(df)}\n")
        f.write(f"- Stages found: {', '.join(stages) if stages else 'none'}\n")
        if optional_missing:
            f.write(f"- Optional models skipped: {', '.join(optional_missing)}\n")
        f.write("\n## Best model per stage\n")
        if best_models:
            for stage, model in best_models.items():
                f.write(f"- {stage}: {model}\n")
        else:
            f.write("- No stages trained.\n")
        if selected_rows:
            f.write("\n## Feature groups used\n")
            for stage in sorted({row["stage"] for row in selected_rows}):
                groups = sorted(
                    {
                        row["feature_group"]
                        for row in selected_rows
                        if row["stage"] == stage and row["status"] == "selected"
                    }
                )
                f.write(f"- {stage}: {', '.join(groups) if groups else 'none'}\n")
                if not any(group in {"weather", "satellite", "image", "cnn"} for group in groups):
                    f.write(f"  - Warning: no weather/satellite/image/cnn features for {stage}\n")
        if skipped_stages:
            f.write("\n## Skipped stages\n")
            for stage, reason in skipped_stages.items():
                f.write(f"- {stage}: {reason}\n")

    selected_path = reports_dir / "ml_selected_features.csv"
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)

    logger.info("Wrote %s", metrics_path)
    logger.info("Wrote %s", summary_path)

    _train_combined(df, model_candidates, reports_dir, models_dir)


def _train_combined(
    df: pd.DataFrame,
    model_candidates: list[tuple[str, object]],
    reports_dir: Path,
    models_dir: Path,
) -> None:
    combined_df = df[df["split"].isin(["train", "validation"])].copy()
    train_df = combined_df[combined_df["split"] == "train"].copy()
    val_df = combined_df[combined_df["split"] == "validation"].copy()

    metrics_rows = []

    if train_df.empty or val_df.empty:
        summary_path = reports_dir / "combined_regression_metrics.csv"
        pd.DataFrame(metrics_rows).to_csv(summary_path, index=False)
        return

    numeric_cols, categorical_cols, _ = _select_features(combined_df, include_stage=True)
    if not numeric_cols and not categorical_cols:
        pd.DataFrame(metrics_rows).to_csv(reports_dir / "combined_regression_metrics.csv", index=False)
        return

    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df["yield_kg_ha"].values
    X_val = val_df[numeric_cols + categorical_cols]
    y_val = val_df["yield_kg_ha"].values

    best_rmse = np.inf
    best_pipeline = None
    best_model_name = None

    for model_name, model in model_candidates:
        pipeline = _build_pipeline(model, numeric_cols, categorical_cols)
        try:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            metrics = _evaluate(y_val, preds)
            metrics_rows.append(
                {
                    "model": model_name,
                    "n_train": len(train_df),
                    "n_val": len(val_df),
                    **metrics,
                    "status": "ok",
                    "message": "",
                }
            )
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_pipeline = pipeline
                best_model_name = model_name
        except Exception as exc:
            metrics_rows.append(
                {
                    "model": model_name,
                    "n_train": len(train_df),
                    "n_val": len(val_df),
                    "mae": np.nan,
                    "mse": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                    "r2": np.nan,
                    "status": "failed",
                    "message": str(exc),
                }
            )
            logger.warning("Combined model %s failed: %s", model_name, exc)

    metrics_path = reports_dir / "combined_regression_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    if best_pipeline is not None:
        model_path = models_dir / "best_combined_model.joblib"
        dump(best_pipeline, model_path)
        fi_path = reports_dir / "feature_importance_combined.csv"
        if not _extract_feature_importance(best_pipeline, fi_path):
            if fi_path.exists():
                fi_path.unlink(missing_ok=True)

        summary_path = reports_dir / "combined_regression_summary.md"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("# Combined Regression Summary\n\n")
            f.write(f"- Best model: {best_model_name}\n")
            f.write(f"- Training rows: {len(train_df)}\n")
            f.write(f"- Validation rows: {len(val_df)}\n")


if __name__ == "__main__":
    main()
