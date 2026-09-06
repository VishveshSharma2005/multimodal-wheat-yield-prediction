from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def _venv_python() -> Path | None:
    """Return the interpreter of a project-local .venv, if one is usable."""
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return candidate
    return None


def _run_script(script_path: Path) -> int:
    venv_python = _venv_python()
    python_exe = venv_python if venv_python is not None else Path(sys.executable)
    print(f"Running: {python_exe} {script_path}")
    result = subprocess.run([str(python_exe), str(script_path)], cwd=str(ROOT))
    return result.returncode


def _read_csv(path: Path):
    if not path.exists():
        return None
    try:
        import pandas as pd

        return pd.read_csv(path)
    except Exception:
        return None


def _read_csv_as_md_table(path: Path, max_rows: int = 20) -> str:
    df = _read_csv(path)
    if df is None or df.empty:
        return ""
    if len(df) > max_rows:
        df = df.head(max_rows)
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _best_models_from_metrics(metrics_df) -> dict:
    if metrics_df is None or metrics_df.empty:
        return {}
    ok_df = metrics_df[metrics_df["status"] == "ok"].copy()
    if ok_df.empty:
        return {}
    best = {}
    for stage, group in ok_df.groupby("stage"):
        best_row = group.sort_values("rmse").iloc[0]
        best[stage] = best_row["model"]
    return best


def _write_report(reports_dir: Path, dataset_path: Path) -> None:
    report_path = reports_dir / "baseline_ml_report.md"
    stage_metrics_path = reports_dir / "stagewise_regression_metrics.csv"
    combined_metrics_path = reports_dir / "combined_regression_metrics.csv"
    quintile_metrics_path = reports_dir / "quintile_classification_metrics.csv"

    stage_metrics = _read_csv(stage_metrics_path)
    combined_metrics = _read_csv(combined_metrics_path)
    quintile_metrics = _read_csv(quintile_metrics_path)

    dataset_df = _read_csv(dataset_path)
    train_rows = 0
    val_rows = 0
    stages = []
    if dataset_df is not None and "split" in dataset_df.columns:
        train_rows = int((dataset_df["split"] == "train").sum())
        val_rows = int((dataset_df["split"] == "validation").sum())
        if "stage" in dataset_df.columns:
            stages = sorted(dataset_df["stage"].dropna().unique().tolist())

    models_tried = []
    if stage_metrics is not None and "model" in stage_metrics.columns:
        models_tried = sorted(stage_metrics["model"].dropna().unique().tolist())

    best_models = _best_models_from_metrics(stage_metrics)

    stage_table = _read_csv_as_md_table(stage_metrics_path)
    combined_table = _read_csv_as_md_table(combined_metrics_path)
    quintile_table = _read_csv_as_md_table(quintile_metrics_path)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Baseline ML Report\n\n")
        f.write(f"- Dataset file: {dataset_path}\n")
        f.write("- Train/validation split: train=2020-2022, validation=2023, prediction_2025=2024\n")
        f.write(f"- Training rows: {train_rows}\n")
        f.write(f"- Validation rows: {val_rows}\n")
        f.write(f"- Stages trained: {', '.join(stages) if stages else 'none'}\n")
        f.write(f"- Models tried: {', '.join(models_tried) if models_tried else 'none'}\n\n")
        f.write("## Best model per stage\n")
        if best_models:
            for stage, model in best_models.items():
                f.write(f"- {stage}: {model}\n")
        else:
            f.write("- No best models available.\n")
        f.write("\n## Stage-wise regression metrics\n")
        if stage_table:
            f.write(stage_table + "\n\n")
        else:
            f.write("- Metrics not available. Run the training scripts first.\n\n")
        f.write("## Combined regression metrics\n")
        if combined_table:
            f.write(combined_table + "\n\n")
        else:
            f.write("- Metrics not available. Run the training scripts first.\n\n")
        f.write("## Quintile classification metrics\n")
        if quintile_table:
            f.write(quintile_table + "\n\n")
        else:
            f.write("- Metrics not available. Run the training scripts first.\n\n")
        f.write("## Limitations\n")
        f.write("- Baselines do not model temporal sequences across stages.\n")
        f.write("- Some stages may be skipped if validation data is missing.\n")
        f.write("- Optional models (LightGBM/XGBoost) run only if installed.\n\n")
        f.write("## Next step\n")
        f.write("- Train LSTM temporal sequence model.\n")


def main() -> None:
    regression_script = ROOT / "src" / "models" / "train_stagewise_regression.py"
    classifier_script = ROOT / "src" / "models" / "train_quintile_classifier.py"

    exit_codes = []
    exit_codes.append(_run_script(regression_script))
    exit_codes.append(_run_script(classifier_script))

    reports_dir = ROOT / "reports" / "model_results"
    dataset_path = ROOT / "data" / "processed" / "stage_features_with_split.csv"
    _write_report(reports_dir, dataset_path)

    if any(code != 0 for code in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
