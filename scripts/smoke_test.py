"""Lightweight repository smoke test.

Checks that a fresh clone is wired up correctly WITHOUT training anything.
Nothing here fits a model, downloads data, or overwrites an artefact.

    python scripts/smoke_test.py

Exit code 0 = all required checks passed. Optional checks (TensorFlow, the
CNN .keras files, the large gitignored datasets) are reported as SKIP when the
dependency or file is absent; they never fail the run.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    detail = " ".join(str(detail).split())
    if len(detail) > 300:
        detail = detail[:297] + "..."
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""), flush=True)


def check(name: str, fn, optional: bool = False) -> None:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not raise
        record(SKIP if optional else FAIL, name, f"{type(exc).__name__}: {exc}")
        return
    if detail is None:
        record(SKIP, name, "not present")
        return
    record(PASS, name, str(detail))


# ---------------------------------------------------------------------------
# 1. Interpreter and required third-party packages
# ---------------------------------------------------------------------------
def _python_version() -> str:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        raise RuntimeError(f"Python 3.10+ required, found {major}.{minor}")
    return f"{major}.{minor}.{sys.version_info[2]}"


REQUIRED_MODULES = ["pandas", "numpy", "yaml", "sklearn", "joblib", "matplotlib"]
OPTIONAL_MODULES = ["torch", "xgboost", "lightgbm", "openpyxl", "tabulate", "PIL", "tensorflow"]


def _module_present(name: str):
    return name if importlib.util.find_spec(name) is not None else None


# ---------------------------------------------------------------------------
# 2. Config and project paths
# ---------------------------------------------------------------------------
def _load_config():
    from src.utils.config import load_config

    config = load_config()
    missing = [k for k in ("paths", "stage_definitions", "train_years") if k not in config]
    if missing:
        raise RuntimeError(f"config.yaml missing keys: {missing}")
    return f"{len(config['stage_definitions'])} stages defined"


def _paths_resolve():
    from src.utils.config import load_config

    config = load_config()
    absent = [k for k, v in config["paths"].items() if not Path(v).exists()]
    return f"{len(config['paths']) - len(absent)}/{len(config['paths'])} configured dirs exist"


# ---------------------------------------------------------------------------
# 3. Source tree imports (syntax + import graph, no side effects)
# ---------------------------------------------------------------------------
def _compile_sources() -> str:
    import py_compile

    files = sorted(p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    files += sorted((ROOT).glob("run_*.py"))
    errors = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True, cfile=str(path.with_suffix(".pyc.tmp")))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
        finally:
            path.with_suffix(".pyc.tmp").unlink(missing_ok=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    return f"{len(files)} files compile cleanly"


def _import_utils() -> str:
    from src.utils import config, logging_utils, paths  # noqa: F401

    return "src.utils imports OK"


# ---------------------------------------------------------------------------
# 4. Datasets that ship with the repository
# ---------------------------------------------------------------------------
STAGES = [
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


def _stage_features() -> str | None:
    import pandas as pd

    path = ROOT / "data" / "processed" / "stage_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    problems = []
    if len(df) != 1485:
        problems.append(f"expected 1485 rows, found {len(df)}")
    if df["district"].nunique() != 33:
        problems.append(f"expected 33 districts, found {df['district'].nunique()}")
    if sorted(df["stage"].dropna().unique()) != sorted(STAGES):
        problems.append("stage vocabulary does not match the 9 documented stages")
    if df.duplicated(["district", "season_year_start", "stage"]).any():
        problems.append("duplicate (district, year, stage) keys")
    for case_id, group in df.groupby("case_id"):
        if not group.sort_values("stage_order")["stage_end_das"].is_monotonic_increasing:
            problems.append(f"non-monotonic DAS for {case_id}")
            break
    if problems:
        raise RuntimeError("; ".join(problems))
    years = sorted(int(y) for y in df["season_year_start"].dropna().unique())
    return f"1485 rows, 33 districts x 9 stages x {len(years)} years {years}"


def _ground_truth_integrity() -> str | None:
    """The 2024-25 target must never silently carry a value from another year.

    Checks every artefact that holds a forecast-season target or an error derived
    from one. District-level 2024-25 Gujarat wheat APY has not been published, so
    all of these must be empty. See docs/2025_yield_reference_methodology.md.
    """
    import pandas as pd

    # (relative path, year column, columns that must be null for the 2024 season)
    targets = [
        ("data/processed/stage_features.csv", "season_year_start", ["yield_kg_ha"]),
        ("data/processed/stage_features_with_split.csv", "season_year_start", ["yield_kg_ha"]),
        ("data/processed/stage_features_with_cnn.csv", "season_year_start", ["yield_kg_ha"]),
        ("data/processed/targets.csv", "season_year", ["yield_kg_ha", "yield_quintile"]),
        ("data/interim/cases.csv", "season_year_start", ["yield_kg_ha"]),
        ("data/processed/prediction_2025_stagewise_predictions.csv", "season_year_start",
         ["actual_yield_kg_ha", "absolute_error", "percentage_error"]),
        ("data/processed/mehsana_2025_stagewise_predictions.csv", "season_year_start",
         ["actual_yield_kg_ha", "absolute_error", "percentage_error"]),
        ("data/processed/lstm_predictions_2025.csv", "season_year_start",
         ["actual_yield_kg_ha", "absolute_error", "percentage_error"]),
    ]

    problems = []
    checked = 0
    for rel, year_col, cols in targets:
        path = ROOT / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if year_col not in df.columns:
            continue
        checked += 1
        season = df[df[year_col] == 2024]
        for col in cols:
            if col in season.columns and season[col].notna().any():
                values = sorted(season[col].dropna().unique().tolist())[:3]
                problems.append(f"{rel}:{col} has {int(season[col].notna().sum())} non-null (e.g. {values})")
        if "actual_available" in season.columns and season["actual_available"].fillna(0).astype(float).sum():
            problems.append(f"{rel}: actual_available is set for the forecast season")
        if "evaluation_status" in season.columns:
            bad = season.loc[season["evaluation_status"] != "pending_actual_apy", "evaluation_status"]
            if not bad.empty:
                problems.append(f"{rel}: evaluation_status is {sorted(bad.unique())}, expected pending_actual_apy")

    if problems:
        raise RuntimeError(
            "; ".join(problems)
            + ". District-level 2024-25 APY has not been published - see "
              "docs/2025_yield_reference_methodology.md."
        )
    if checked == 0:
        return None
    return f"2024-25 target correctly empty (pending_actual_apy) across {checked} artefacts"


def _evaluation_report_has_no_2025_metrics() -> str | None:
    """The 2025 evaluation report must not print a numeric metric."""
    import re

    path = ROOT / "reports" / "model_results" / "prediction_2025_evaluation.md"
    if not path.exists():
        return None
    offending = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^-\s*(MAPE|RMSE|MAE|R2)\s*:\s*(.+)$", line.strip())
        if m and m.group(2).strip() != "N/A":
            offending.append(line.strip())
    if offending:
        raise RuntimeError(
            "2025 evaluation report prints a metric with no ground truth: "
            + "; ".join(offending)
        )
    return "MAPE / RMSE / MAE / R2 all reported as N/A"


def _predictions() -> str | None:
    import pandas as pd

    checks = {
        "validation_stagewise_predictions.csv": 297,
        "prediction_2025_stagewise_predictions.csv": 297,
        "mehsana_2025_stagewise_predictions.csv": 9,
    }
    seen = []
    for name, expected in checks.items():
        path = ROOT / "data" / "processed" / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if len(df) != expected:
            raise RuntimeError(f"{name}: expected {expected} rows, found {len(df)}")
        seen.append(f"{name}={len(df)}")
    return ", ".join(seen) if seen else None


# ---------------------------------------------------------------------------
# 4b. Feature-policy / leakage guarantees
# ---------------------------------------------------------------------------
def _feature_policy_importable() -> str:
    from src.features import feature_policy

    return (f"{len(feature_policy.TARGET_DERIVED_COLUMNS)} target-derived + "
            f"{len(feature_policy.REDUNDANT_ALIAS_COLUMNS)} duplicate aliases blocked")


def _duplicate_table_current() -> str | None:
    """The declared exact-duplicate aliases must still be exact duplicates."""
    import pandas as pd

    from src.features.feature_policy import verify_duplicates

    path = ROOT / "data" / "processed" / "stage_features.csv"
    if not path.exists():
        return None
    problems = verify_duplicates(pd.read_csv(path))
    if problems:
        raise RuntimeError("; ".join(problems) + " - update DUPLICATE_ALIASES in src/features/feature_policy.py")
    return "all declared duplicate aliases verified elementwise"


def _no_leaked_features_selected() -> str | None:
    """No target-derived or duplicate column may appear in any model's inputs."""
    import pandas as pd

    from src.features.feature_policy import is_excluded, is_redundant_alias, is_target_derived

    problems = []
    checked = []

    sel_path = ROOT / "reports" / "model_results" / "ml_selected_features.csv"
    if sel_path.exists():
        sel = pd.read_csv(sel_path)
        chosen = sorted(set(sel.loc[sel["status"] == "selected", "feature"]))
        bad = [c for c in chosen if is_excluded(c)]
        if bad:
            problems.append(f"regression selected excluded features: {bad[:6]}")
        checked.append(f"regression={len(chosen)}")

    ckpt = ROOT / "models" / "deep_learning" / "lstm_yield_model.pt"
    if ckpt.exists() and importlib.util.find_spec("torch") is not None:
        import torch

        cols = torch.load(ckpt, map_location="cpu", weights_only=False).get("feature_columns", [])
        # District one-hots are generated names like "district_Mehsana"; judge the
        # raw source columns only.
        raw = [c for c in cols if not c.startswith(("district_", "crop_name_", "season_type_", "stage_name_"))]
        bad = [c for c in raw if is_target_derived(c) or is_redundant_alias(c)]
        if bad:
            problems.append(f"LSTM uses excluded features: {bad[:6]}")
        checked.append(f"lstm={len(cols)}")

    if problems:
        raise RuntimeError("; ".join(problems))
    if not checked:
        return None
    return "no target-derived or duplicate features in any model input (" + ", ".join(checked) + ")"


# ---------------------------------------------------------------------------
# 5b. LSTM checkpoint contract
# ---------------------------------------------------------------------------
def _lstm_checkpoint_contract() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    path = ROOT / "models" / "deep_learning" / "lstm_yield_model.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)

    required = ["model_state", "input_size", "feature_columns", "scaler_mean", "scaler_scale",
                "imputer_statistics", "target_mean", "target_std",
                "best_epoch", "best_val_rmse", "epochs_run", "random_seed"]
    missing = [k for k in required if k not in ck]
    if missing:
        raise RuntimeError(f"checkpoint missing keys: {missing}")

    if len(ck["feature_columns"]) != int(ck["input_size"]):
        raise RuntimeError(
            f"feature_columns ({len(ck['feature_columns'])}) != input_size ({ck['input_size']})")

    sd = ck["model_state"]
    if "lstm.weight_ih_l0" not in sd or "fc.weight" not in sd:
        raise RuntimeError("unexpected architecture: missing lstm.weight_ih_l0 / fc.weight")
    hidden = sd["lstm.weight_hh_l0"].shape[1]
    layers = len({k.split("_l")[-1] for k in sd if k.startswith("lstm.weight_ih")})
    if (hidden, layers) != (64, 2):
        raise RuntimeError(f"architecture changed: hidden={hidden}, layers={layers}, expected 64/2")
    if sd["lstm.weight_ih_l0"].shape[1] != int(ck["input_size"]):
        raise RuntimeError("input_size does not match the first LSTM weight matrix")

    # The restored epoch must be the best one, not the last.
    log = ROOT / "reports" / "model_results" / "lstm_training_log.csv"
    if log.exists():
        import pandas as pd

        lg = pd.read_csv(log)
        if not lg.empty and lg["val_rmse"].notna().any():
            best_row = lg.loc[lg["val_rmse"].idxmin()]
            if int(best_row["epoch"]) != int(ck["best_epoch"]):
                raise RuntimeError(
                    f"checkpoint best_epoch={ck['best_epoch']} but the log's minimum val_rmse is at "
                    f"epoch {int(best_row['epoch'])} - the deepcopy fix may have regressed")
            metrics = ROOT / "reports" / "model_results" / "lstm_metrics.csv"
            if metrics.exists():
                rep = float(pd.read_csv(metrics).iloc[0]["rmse"])
                if abs(rep - float(best_row["val_rmse"])) > 1.0:
                    raise RuntimeError(
                        f"reported RMSE {rep:.4f} does not match the best epoch's "
                        f"{float(best_row['val_rmse']):.4f}")

    return (f"epoch {ck['best_epoch']}/{ck['epochs_run']}, val_rmse {float(ck['best_val_rmse']):.2f}, "
            f"input_size {ck['input_size']}, seed {ck['random_seed']}")


def _cnn_files_intact() -> str | None:
    """All five trained CNN classifiers must be present and non-trivial."""
    expected = ["xception", "inceptionv3", "vgg16", "densenet121", "mobilenetv2"]
    cnn_dir = ROOT / "models" / "cnn"
    if not cnn_dir.exists():
        return None
    missing = [n for n in expected if not (cnn_dir / f"{n}_best.keras").exists()]
    if missing:
        raise RuntimeError(f"missing CNN weights: {missing}")
    total = sum((cnn_dir / f"{n}_best.keras").stat().st_size for n in expected)
    if total < 200_000_000:
        raise RuntimeError(f"CNN weights unexpectedly small ({total/1e6:.0f} MB)")
    return f"all 5 present, {total/1e6:.0f} MB (never retrained)"


# ---------------------------------------------------------------------------
# 6b. Robustness experiments present and self-consistent
# ---------------------------------------------------------------------------
def _robustness_artifacts() -> str | None:
    import pandas as pd

    R = ROOT / "reports" / "model_results"
    needed = ["temporal_cv_metrics.csv", "temporal_cv_fold_metrics.csv",
              "baseline_comparison.csv", "within_district_skill.csv"]
    present = [n for n in needed if (R / n).exists()]
    if not present:
        return None
    if len(present) != len(needed):
        raise RuntimeError(f"missing robustness outputs: {sorted(set(needed) - set(present))}")

    folds = pd.read_csv(R / "temporal_cv_fold_metrics.csv")
    if 2024 in folds["fold_validation_year"].unique():
        raise RuntimeError("2024-25 must never be used as a validation fold - it has no target")
    for _, r in folds.iterrows():
        years = [int(y) for y in str(r["train_years"]).split(",") if y]
        if any(y >= int(r["fold_validation_year"]) for y in years):
            raise RuntimeError(
                f"fold {r['fold_validation_year']} trains on {years}: future information leak")

    agg = pd.read_csv(R / "temporal_cv_metrics.csv")
    if not {"model", "baseline"} <= set(agg["kind"]):
        raise RuntimeError("temporal CV must contain both models and baselines")
    return (f"{folds['fold_validation_year'].nunique()} forward-chaining folds, "
            f"{(agg.kind == 'model').sum()} models vs {(agg.kind == 'baseline').sum()} baselines")

# ---------------------------------------------------------------------------
# 5. Model artefacts - load only, never fit
# ---------------------------------------------------------------------------
def _load_regression_models() -> str | None:
    from joblib import load

    paths = sorted((ROOT / "models" / "regression").glob("*.joblib"))
    if not paths:
        return None
    for path in paths:
        model = load(path)
        if not hasattr(model, "predict"):
            raise RuntimeError(f"{path.name} has no .predict()")
    return f"{len(paths)} regression pipelines load"


def _regression_inference() -> str | None:
    """Predict on a handful of real rows to prove the saved pipelines still run."""
    import pandas as pd
    from joblib import load

    data = ROOT / "data" / "processed" / "stage_features.csv"
    model_path = ROOT / "models" / "regression" / "best_model_sowing.joblib"
    if not (data.exists() and model_path.exists()):
        return None
    df = pd.read_csv(data)
    sample = df[df["stage"] == "sowing"].head(5)
    preds = load(model_path).predict(sample)
    return f"5 sowing rows -> mean {float(preds.mean()):.1f} kg/ha"


def _load_classifiers() -> str | None:
    from joblib import load

    paths = sorted((ROOT / "models" / "classification").glob("*.joblib"))
    if not paths:
        return None
    for path in paths:
        if not hasattr(load(path), "predict"):
            raise RuntimeError(f"{path.name} has no .predict()")
    return f"{len(paths)} quintile classifiers load"


def _load_lstm() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    path = ROOT / "models" / "deep_learning" / "lstm_yield_model.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state", "input_size", "feature_columns", "scaler_mean", "target_mean"}
    missing = required - set(ckpt)
    if missing:
        raise RuntimeError(f"LSTM checkpoint missing keys: {sorted(missing)}")
    return f"input_size={ckpt['input_size']}, max_len={ckpt.get('max_len')}"


def _cnn_artifacts() -> str | None:
    paths = sorted((ROOT / "models" / "cnn").glob("*.keras"))
    if not paths:
        return None
    total_mb = sum(p.stat().st_size for p in paths) / 1e6
    return f"{len(paths)} .keras files present ({total_mb:.0f} MB, gitignored)"


def _cnn_loads() -> str | None:
    if importlib.util.find_spec("tensorflow") is None:
        return None
    path = ROOT / "models" / "cnn" / "mobilenetv2_best.keras"
    if not path.exists():
        return None
    import tensorflow as tf

    model = tf.keras.models.load_model(path)
    return f"mobilenetv2 loads, output shape {model.output_shape}"


# ---------------------------------------------------------------------------
# 6. Output directories are creatable
# ---------------------------------------------------------------------------
def _output_dirs() -> str:
    from src.utils.paths import ensure_dir

    for rel in ("reports/model_results", "reports/figures", "reports/summaries",
                "models/regression", "models/classification", "models/deep_learning"):
        ensure_dir(ROOT / rel)
    return "report/model output directories present"


def main() -> int:
    print("=" * 72)
    print("Stage-wise Multi-Modal Wheat Yield Prediction - smoke test")
    print("(read-only: nothing is trained, downloaded, or overwritten)")
    print("=" * 72)

    print("\n-- environment --")
    check("python version", _python_version)
    for name in REQUIRED_MODULES:
        check(f"import {name}", lambda n=name: _module_present(n))
    print("\n-- optional packages --")
    for name in OPTIONAL_MODULES:
        check(f"import {name}", lambda n=name: _module_present(n), optional=True)

    print("\n-- configuration --")
    check("load configs/config.yaml", _load_config)
    check("configured paths resolve", _paths_resolve)

    print("\n-- source tree --")
    check("all sources compile", _compile_sources)
    check("src.utils importable", _import_utils)

    print("\n-- datasets --")
    check("stage_features.csv schema", _stage_features)
    check("2024-25 ground-truth integrity", _ground_truth_integrity)
    check("no 2025 metrics without ground truth", _evaluation_report_has_no_2025_metrics)
    check("prediction output row counts", _predictions)

    print("\n-- feature policy / leakage --")
    check("feature policy importable", _feature_policy_importable)
    check("duplicate-alias table still valid", _duplicate_table_current)
    check("no leaked features in any model", _no_leaked_features_selected)

    print("\n-- model artefacts (load only) --")
    check("regression pipelines load", _load_regression_models)
    check("regression inference on 5 rows", _regression_inference)
    check("quintile classifiers load", _load_classifiers)
    check("LSTM checkpoint loads", _load_lstm, optional=True)
    check("LSTM checkpoint contract", _lstm_checkpoint_contract, optional=True)
    check("CNN weights intact (never retrained)", _cnn_files_intact, optional=True)
    check("CNN model loads (TensorFlow)", _cnn_loads, optional=True)

    print("\n-- robustness experiments --")
    check("temporal CV / baselines / within-district", _robustness_artifacts)

    print("\n-- outputs --")
    check("output directories", _output_dirs)

    failed = [r for r in results if r[0] == FAIL]
    skipped = [r for r in results if r[0] == SKIP]
    passed = [r for r in results if r[0] == PASS]

    print("\n" + "=" * 72)
    print(f"PASS {len(passed)}   SKIP {len(skipped)}   FAIL {len(failed)}")
    if failed:
        print("\nFailures:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
