from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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


def _select_features(df: pd.DataFrame, include_stage: bool) -> tuple[list[str], list[str]]:
    categorical_cols = []
    if "district" in df.columns:
        categorical_cols.append("district")
    if include_stage and "stage" in df.columns:
        categorical_cols.append("stage")

    numeric_cols = []
    for col in df.columns:
        if _is_excluded(col) or col in categorical_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols, categorical_cols


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


def _sanitize_stage(stage: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stage)


def _load_dataset() -> tuple[pd.DataFrame, Path, Path, Path]:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    models_dir = ROOT / "models" / "classification"

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

    return df, stage_path, reports_dir, models_dir


def _compute_quintiles(train_y: pd.Series) -> tuple[np.ndarray, list[str]]:
    unique_count = train_y.nunique(dropna=True)
    if unique_count < 2:
        return np.array([]), []
    bins = min(5, unique_count)
    _, bin_edges = pd.qcut(train_y, q=bins, retbins=True, duplicates="drop")
    if len(bin_edges) < 3:
        return np.array([]), []
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    labels = [f"Q{i}" for i in range(1, len(bin_edges))]
    return bin_edges, labels


def _assign_quintiles(series: pd.Series, bin_edges: np.ndarray, labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bin_edges, labels=labels, include_lowest=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    df, stage_path, reports_dir, models_dir = _load_dataset()
    if "yield_kg_ha" not in df.columns:
        raise ValueError("Missing yield_kg_ha after merging targets.")

    df = df[df["split"].isin(["train", "validation", "prediction_2025"])].copy()

    train_df = df[df["split"] == "train"].copy()
    bin_edges, labels = _compute_quintiles(train_df["yield_kg_ha"].dropna())
    if len(bin_edges) == 0:
        summary_path = reports_dir / "quintile_classification_summary.md"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("# Quintile Classification Summary\n\n")
            f.write("- Not enough unique yields to build quintiles. Classification skipped.\n")
        return

    df["yield_quintile"] = _assign_quintiles(df["yield_kg_ha"], bin_edges, labels)

    stages = sorted(df["stage"].dropna().unique().tolist()) if "stage" in df.columns else []
    model_candidates = [
        ("RandomForestClassifier", RandomForestClassifier(random_state=42)),
        ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42)),
    ]

    metrics_rows = []
    best_models = {}
    skipped_stages = {}

    for stage in stages:
        stage_df = df[df["stage"] == stage].copy()
        train_stage = stage_df[stage_df["split"] == "train"].copy()
        val_stage = stage_df[stage_df["split"] == "validation"].copy()

        if train_stage.empty:
            skipped_stages[stage] = "No training rows"
            continue
        if val_stage.empty:
            skipped_stages[stage] = "No validation rows"
            continue

        train_stage = train_stage.dropna(subset=["yield_quintile"])
        val_stage = val_stage.dropna(subset=["yield_quintile"])

        if train_stage.empty or val_stage.empty:
            skipped_stages[stage] = "Missing quintile labels"
            continue

        numeric_cols, categorical_cols = _select_features(stage_df, include_stage=False)
        if not numeric_cols and not categorical_cols:
            skipped_stages[stage] = "No usable feature columns"
            continue

        X_train = train_stage[numeric_cols + categorical_cols]
        y_train = train_stage["yield_quintile"].values
        X_val = val_stage[numeric_cols + categorical_cols]
        y_val = val_stage["yield_quintile"].values

        best_score = -np.inf
        best_pipeline = None
        best_model_name = None

        for model_name, model in model_candidates:
            pipeline = _build_pipeline(model, numeric_cols, categorical_cols)
            try:
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_val)
                acc = accuracy_score(y_val, preds)
                macro_f1 = f1_score(y_val, preds, average="macro")
                metrics_rows.append(
                    {
                        "stage": stage,
                        "model": model_name,
                        "n_train": len(train_stage),
                        "n_val": len(val_stage),
                        "accuracy": float(acc),
                        "macro_f1": float(macro_f1),
                        "status": "ok",
                        "message": "",
                    }
                )
                if macro_f1 > best_score:
                    best_score = macro_f1
                    best_pipeline = pipeline
                    best_model_name = model_name
            except Exception as exc:
                metrics_rows.append(
                    {
                        "stage": stage,
                        "model": model_name,
                        "n_train": len(train_stage),
                        "n_val": len(val_stage),
                        "accuracy": np.nan,
                        "macro_f1": np.nan,
                        "status": "failed",
                        "message": str(exc),
                    }
                )
                logger.warning("Stage %s classifier %s failed: %s", stage, model_name, exc)

        if best_pipeline is None:
            skipped_stages[stage] = "All models failed"
            continue

        model_path = models_dir / f"best_classifier_{_sanitize_stage(stage)}.joblib"
        dump(best_pipeline, model_path)
        best_models[stage] = best_model_name

        cm = confusion_matrix(y_val, best_pipeline.predict(X_val), labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_path = reports_dir / f"confusion_matrix_{_sanitize_stage(stage)}.csv"
        cm_df.to_csv(cm_path)

    metrics_path = reports_dir / "quintile_classification_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    summary_path = reports_dir / "quintile_classification_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Quintile Classification Summary\n\n")
        f.write(f"- Dataset: {stage_path}\n")
        f.write("- Quintiles computed from training data only.\n")
        f.write(f"- Labels: {', '.join(labels)}\n\n")
        f.write("## Best model per stage\n")
        if best_models:
            for stage, model in best_models.items():
                f.write(f"- {stage}: {model}\n")
        else:
            f.write("- No stages trained.\n")
        if skipped_stages:
            f.write("\n## Skipped stages\n")
            for stage, reason in skipped_stages.items():
                f.write(f"- {stage}: {reason}\n")

    logger.info("Wrote %s", metrics_path)
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
