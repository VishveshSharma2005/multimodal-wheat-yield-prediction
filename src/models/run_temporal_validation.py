"""Walk-forward (leave-one-year-out) temporal validation plus naive baselines.

This is an ADDITIONAL robustness experiment. It does not replace, and does not
overwrite, the authoritative 2023-24 hold-out result produced by
`train_stagewise_regression.py`.

Why it exists
-------------
The headline result selects the best regressor on the 2023-24 validation season
and then reports that same season, on 33 district cases. Two questions follow:

1. Does the result hold across more than one held-out year?
2. Does the model beat a naive baseline at all? Because the target is a
   district-level yield and districts differ systematically, a high R2 can be
   produced by district identity alone.

Protocol
--------
Strictly forward-chaining. For a held-out year Y the model sees only seasons
earlier than Y - no future information, ever.

    fold 1: train 2020-21                          -> validate 2021-22
    fold 2: train 2020-21..2021-22                 -> validate 2022-23
    fold 3: train 2020-21..2022-23                 -> validate 2023-24

2024-25 is never a fold: its target has not been published.

Feature engineering and the exclusion policy are identical to the main
pipeline (`src/features/feature_policy.py`).

Outputs (all new files)
-----------------------
    reports/model_results/temporal_cv_fold_metrics.csv
    reports/model_results/temporal_cv_metrics.csv
    reports/model_results/baseline_comparison.csv
    reports/summaries/temporal_validation_summary.md
    reports/summaries/baseline_comparison.md
    reports/figures/temporal_cv_model_vs_baseline.png
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.features.feature_policy import is_excluded
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)

RANDOM_SEED = 42

# Held-out years, in forward-chaining order. Each is validated using only
# seasons strictly before it. 2020 is training-only (nothing precedes it);
# 2024 is excluded because its target does not exist.
FOLD_YEARS = [2021, 2022, 2023]

STAGE_ORDER = [
    "sowing",
    "early_vegetative",
    "tillering",
    "stem_elongation",
    "booting_heading",
    "flowering",
    "grain_filling_initial",
    "grain_filling",
    "maturity_preharvest",
]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # Zero-safe MAPE: some districts report a 0.00 yield in some seasons.
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100.0)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"n": int(y_true.size), "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# data / features
# ---------------------------------------------------------------------------
def _load() -> pd.DataFrame:
    config = load_config()
    processed = Path(config["paths"]["processed_dir"])
    path = processed / "stage_features.csv"
    if not path.exists():
        path = processed / "stage_features_with_cnn.csv"
    if not path.exists():
        raise FileNotFoundError("No stage feature table found. Run build_stage_table.py first.")
    df = pd.read_csv(path)
    df = df[df["yield_kg_ha"].notna()].copy()
    logger.info("Loaded %s: %d labelled rows, years %s",
                path.name, len(df), sorted(df["season_year_start"].unique().tolist()))
    return df


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in ["district", "stage", "crop_name", "season_type"] if c in df.columns]
    numeric = [
        c for c in df.columns
        if (not is_excluded(c)) and c not in categorical and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Drop columns that are unusable everywhere (e.g. area_ha is 100% empty).
    numeric = [c for c in numeric if df[c].notna().mean() > 0.10]
    return numeric, categorical


def _safe_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _pipeline(model, numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", _safe_onehot())]),
            categorical,
        ))
    return Pipeline([("prep", ColumnTransformer(transformers, remainder="drop")), ("model", model)])


def _model_zoo() -> list[tuple[str, object]]:
    zoo: list[tuple[str, object]] = [
        ("RandomForestRegressor", RandomForestRegressor(random_state=RANDOM_SEED)),
        ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=RANDOM_SEED)),
        ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(random_state=RANDOM_SEED)),
    ]
    try:
        from lightgbm import LGBMRegressor

        zoo.append(("LightGBMRegressor", LGBMRegressor(
            random_state=RANDOM_SEED, n_estimators=200, min_data_in_leaf=5,
            min_data_in_bin=1, num_leaves=31, verbose=-1)))
    except Exception:
        logger.warning("lightgbm not installed - skipping LightGBMRegressor")
    try:
        from xgboost import XGBRegressor

        zoo.append(("XGBoostRegressor", XGBRegressor(
            random_state=RANDOM_SEED, objective="reg:squarederror", n_estimators=200,
            learning_rate=0.05, max_depth=6, n_jobs=-1, verbosity=0)))
    except Exception:
        logger.warning("xgboost not installed - skipping XGBoostRegressor")
    return zoo


# ---------------------------------------------------------------------------
# baselines - each sees only seasons strictly before the held-out year
# ---------------------------------------------------------------------------
def _baseline_district_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline A: each district's mean yield over the training seasons."""
    per_case = train.drop_duplicates(["district", "season_year_start"])
    means = per_case.groupby("district")["yield_kg_ha"].mean()
    fallback = float(per_case["yield_kg_ha"].mean())
    return test["district"].map(means).fillna(fallback).to_numpy(dtype=float)


def _baseline_persistence(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline B: the district's most recent known yield before the held-out year."""
    per_case = train.drop_duplicates(["district", "season_year_start"])
    last = (per_case.sort_values("season_year_start")
                    .groupby("district")["yield_kg_ha"].last())
    fallback = float(per_case["yield_kg_ha"].mean())
    return test["district"].map(last).fillna(fallback).to_numpy(dtype=float)


def _baseline_district_identity(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline C: a model given ONLY district identity and stage index.

    No weather, no satellite, no image features. If the full model cannot beat
    this, it has not learned anything environmental.
    """
    cols = [c for c in ["district", "stage"] if c in train.columns]
    pipe = _pipeline(GradientBoostingRegressor(random_state=RANDOM_SEED), [], cols)
    pipe.fit(train[cols], train["yield_kg_ha"].to_numpy(dtype=float))
    return pipe.predict(test[cols])


BASELINES = {
    "Baseline A: district historical mean": _baseline_district_mean,
    "Baseline B: persistence (last known year)": _baseline_persistence,
    "Baseline C: district identity only": _baseline_district_identity,
}


# ---------------------------------------------------------------------------
def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_SEED)

    config = load_config()
    reports = Path(config["paths"]["reports_dir"]) / "model_results"
    summaries = Path(config["paths"]["reports_dir"]) / "summaries"
    figures = Path(config["paths"]["reports_dir"]) / "figures"
    for d in (reports, summaries, figures):
        ensure_dir(d)

    df = _load()
    numeric, categorical = _feature_columns(df)
    logger.info("Feature set: %d numeric + %d categorical", len(numeric), len(categorical))

    zoo = _model_zoo()
    rows: list[dict] = []

    for year in FOLD_YEARS:
        train = df[df["season_year_start"] < year]
        test = df[df["season_year_start"] == year]
        if train.empty or test.empty:
            logger.warning("Fold %d skipped (train=%d test=%d)", year, len(train), len(test))
            continue
        train_years = sorted(train["season_year_start"].unique().tolist())
        logger.info("Fold %d: train years %s (%d rows) -> validate %d (%d rows)",
                    year, train_years, len(train), year, len(test))

        y_true = test["yield_kg_ha"].to_numpy(dtype=float)
        X_train, X_test = train[numeric + categorical], test[numeric + categorical]
        y_train = train["yield_kg_ha"].to_numpy(dtype=float)

        for name, estimator in zoo:
            try:
                pipe = _pipeline(estimator, numeric, categorical)
                pipe.fit(X_train, y_train)
                m = _metrics(y_true, pipe.predict(X_test))
                status, message = "ok", ""
            except Exception as exc:  # noqa: BLE001
                m = {"n": len(test), "mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
                status, message = "failed", str(exc)
                logger.warning("Fold %d model %s failed: %s", year, name, exc)
            rows.append({"fold_validation_year": year, "train_years": ",".join(map(str, train_years)),
                         "kind": "model", "name": name, "n_train": len(train), **m,
                         "status": status, "message": message})

        for name, fn in BASELINES.items():
            try:
                m = _metrics(y_true, fn(train, test))
                status, message = "ok", ""
            except Exception as exc:  # noqa: BLE001
                m = {"n": len(test), "mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
                status, message = "failed", str(exc)
                logger.warning("Fold %d baseline %s failed: %s", year, name, exc)
            rows.append({"fold_validation_year": year, "train_years": ",".join(map(str, train_years)),
                         "kind": "baseline", "name": name, "n_train": len(train), **m,
                         "status": status, "message": message})

    fold_df = pd.DataFrame(rows)
    fold_path = reports / "temporal_cv_fold_metrics.csv"
    fold_df.to_csv(fold_path, index=False)
    logger.info("Wrote %s (%d rows)", fold_path, len(fold_df))

    # ---- aggregate across folds -------------------------------------------
    ok = fold_df[fold_df["status"] == "ok"]
    agg = (ok.groupby(["kind", "name"])
             .agg(n_folds=("fold_validation_year", "count"),
                  mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                  rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                  mape_mean=("mape", "mean"), mape_std=("mape", "std"),
                  r2_mean=("r2", "mean"), r2_std=("r2", "std"))
             .reset_index()
             .sort_values(["kind", "rmse_mean"]))
    agg = agg.round(4)
    agg_path = reports / "temporal_cv_metrics.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("Wrote %s", agg_path)

    baseline_path = reports / "baseline_comparison.csv"
    best_model = agg[agg.kind == "model"].iloc[0] if (agg.kind == "model").any() else None
    comp = agg.copy()
    if best_model is not None:
        comp["rmse_vs_best_model_pct"] = (
            (comp["rmse_mean"] - best_model["rmse_mean"]) / best_model["rmse_mean"] * 100.0
        ).round(2)
    comp.to_csv(baseline_path, index=False)
    logger.info("Wrote %s", baseline_path)

    _write_summaries(summaries, fold_df, agg, best_model, numeric, categorical)
    _write_figure(figures, agg)


def _fmt(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.2f}"


def _write_summaries(summaries: Path, fold_df: pd.DataFrame, agg: pd.DataFrame,
                     best_model, numeric: list[str], categorical: list[str]) -> None:
    tcv = summaries / "temporal_validation_summary.md"
    with tcv.open("w", encoding="utf-8") as f:
        f.write("# Temporal (walk-forward / leave-one-year-out) validation\n\n")
        f.write("> **This is an additional robustness experiment. It does not replace the\n"
                "> authoritative 2023-24 hold-out result** in\n"
                "> `reports/model_results/stagewise_regression_metrics.csv`. The two answer\n"
                "> different questions and their numbers are not interchangeable.\n\n")
        f.write("## Protocol\n\n")
        f.write("Strictly forward-chaining: for a held-out season, the model sees only\n"
                "seasons that precede it. No future information is used at any point.\n\n")
        f.write("| Fold | Train seasons | Validate |\n|---|---|---|\n")
        for year in FOLD_YEARS:
            sub = fold_df[fold_df.fold_validation_year == year]
            if sub.empty:
                continue
            f.write(f"| {year} | {sub.iloc[0]['train_years']} | {year} |\n")
        f.write("\n2024-25 is never a fold: its district-level APY target has not been published.\n\n")
        f.write(f"Feature set: {len(numeric)} numeric + {len(categorical)} categorical "
                f"({', '.join(categorical)}), pooled across all 9 stages, using the shared\n"
                "exclusion policy in `src/features/feature_policy.py`.\n\n")

        f.write("## Aggregate across folds (mean +/- sd)\n\n")
        f.write("| Kind | Name | Folds | MAE | RMSE | MAPE % | R2 |\n|---|---|---|---|---|---|---|\n")
        for _, r in agg.iterrows():
            f.write(f"| {r['kind']} | {r['name']} | {int(r['n_folds'])} | "
                    f"{_fmt(r['mae_mean'])} ± {_fmt(r['mae_std'])} | "
                    f"{_fmt(r['rmse_mean'])} ± {_fmt(r['rmse_std'])} | "
                    f"{_fmt(r['mape_mean'])} ± {_fmt(r['mape_std'])} | "
                    f"{_fmt(r['r2_mean'])} ± {_fmt(r['r2_std'])} |\n")

        f.write("\n## Per-fold detail\n\n")
        f.write("| Validate | Kind | Name | n | MAE | RMSE | MAPE % | R2 |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for _, r in fold_df[fold_df.status == "ok"].sort_values(
                ["fold_validation_year", "kind", "rmse"]).iterrows():
            f.write(f"| {int(r['fold_validation_year'])} | {r['kind']} | {r['name']} | {int(r['n'])} | "
                    f"{_fmt(r['mae'])} | {_fmt(r['rmse'])} | {_fmt(r['mape'])} | {_fmt(r['r2'])} |\n")

        f.write("\n## Statistical caution\n\n")
        f.write("- Only **3 folds**. A standard deviation over 3 values is a weak estimate of\n"
                "  spread and no significance test is meaningful at this sample size.\n"
                "- Fold 2021 trains on a **single** season (33 cases, 297 stage rows). Its\n"
                "  metrics are the least reliable and dominate the spread.\n"
                "- Fold results are not independent: later folds contain all earlier training data.\n"
                "- No claim of statistical significance between models is made or supported.\n")

    logger.info("Wrote %s", tcv)

    bl = summaries / "baseline_comparison.md"
    with bl.open("w", encoding="utf-8") as f:
        f.write("# Baseline comparison\n\n")
        f.write("Do the multi-modal models learn anything beyond district identity and\n"
                "historical yield? All baselines are evaluated on the **same** forward-chaining\n"
                "folds as the models, and see only seasons before the held-out year.\n\n")
        f.write("| Baseline | Definition |\n|---|---|\n")
        f.write("| A - district historical mean | Predict each district's mean yield over the training seasons. |\n")
        f.write("| B - persistence | Predict the district's most recent known yield before the held-out season. |\n")
        f.write("| C - district identity only | GradientBoosting on district + stage only: no weather, satellite or image features. |\n\n")

        f.write("## Results (mean +/- sd over folds)\n\n")
        f.write("| Kind | Name | MAE | RMSE | MAPE % | R2 |\n|---|---|---|---|---|---|\n")
        for _, r in agg.iterrows():
            f.write(f"| {r['kind']} | {r['name']} | "
                    f"{_fmt(r['mae_mean'])} ± {_fmt(r['mae_std'])} | "
                    f"{_fmt(r['rmse_mean'])} ± {_fmt(r['rmse_std'])} | "
                    f"{_fmt(r['mape_mean'])} ± {_fmt(r['mape_std'])} | "
                    f"{_fmt(r['r2_mean'])} ± {_fmt(r['r2_std'])} |\n")

        f.write("\n## Interpretation\n\n")
        models = agg[agg.kind == "model"]
        bases = agg[agg.kind == "baseline"]
        if not models.empty and not bases.empty:
            bm, bb = models.iloc[0], bases.iloc[0]
            delta = bb["rmse_mean"] - bm["rmse_mean"]
            pct = delta / bb["rmse_mean"] * 100.0 if bb["rmse_mean"] else np.nan
            f.write(f"- Best model: **{bm['name']}**, RMSE {_fmt(bm['rmse_mean'])} ± {_fmt(bm['rmse_std'])}.\n")
            f.write(f"- Best baseline: **{bb['name']}**, RMSE {_fmt(bb['rmse_mean'])} ± {_fmt(bb['rmse_std'])}.\n")
            if pd.notna(delta) and delta > 0:
                f.write(f"- The model reduces RMSE by {_fmt(delta)} kg/ha ({pct:.1f}%) relative to the\n"
                        "  best baseline. With 3 folds this is an indication, not a significant result.\n")
            else:
                f.write("- **The best model does not beat the best baseline on mean RMSE.** This must be\n"
                        "  reported as-is: on this data and protocol, the multi-modal features do not\n"
                        "  demonstrably improve on a naive district-level predictor.\n")
        f.write("\n- Baseline C isolates district identity. The gap between Baseline C and the full\n"
                "  models is the part of performance attributable to weather, satellite and calendar\n"
                "  features rather than to knowing which district a row belongs to.\n")
    logger.info("Wrote %s", bl)


def _write_figure(figures: Path, agg: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    if agg.empty:
        return
    d = agg.sort_values("rmse_mean")
    colors = ["#4C78A8" if k == "model" else "#E45756" for k in d["kind"]]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(d) + 1.5)))
    ax.barh(d["name"], d["rmse_mean"], xerr=d["rmse_std"].fillna(0), color=colors, capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (kg/ha), mean ± sd over walk-forward folds")
    ax.set_title("Temporal CV: models (blue) vs naive baselines (red)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = figures / "temporal_cv_model_vs_baseline.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
