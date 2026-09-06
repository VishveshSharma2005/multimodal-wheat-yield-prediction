from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)


def _load_meta(reports_dir: Path, model_name: str) -> dict:
    meta_path = reports_dir / f"{model_name}_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _safe_count_csv(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(len(pd.read_csv(path)))


def _write_validation_report(
    reports_dir: Path,
    config: dict,
    comparison: pd.DataFrame,
) -> Path:
    split_path = Path(config["paths"]["interim_dir"]) / "cnn_dataset_split.csv"
    processed_dir = Path(config["paths"]["processed_dir"])
    feature_dir = processed_dir / "cnn_features"
    models_dir = ROOT / "models" / "cnn"

    split_df = pd.read_csv(split_path) if split_path.exists() else pd.DataFrame()
    image_count = int(len(split_df)) if not split_df.empty else 0
    stage_count = int(split_df["stage"].nunique()) if "stage" in split_df else 0
    stage_distribution = (
        split_df["stage"].value_counts().sort_index().to_dict() if "stage" in split_df else {}
    )

    best_row = comparison.iloc[0].to_dict()
    best_model = str(best_row.get("model", "unknown"))
    validation_accuracy = None
    best_meta = _load_meta(reports_dir, best_model)
    if best_meta:
        validation_accuracy = best_meta.get("best_validation_accuracy")

    feature_counts = {}
    if feature_dir.exists():
        for feature_path in sorted(feature_dir.glob("*_features.csv")):
            feature_counts[feature_path.name] = _safe_count_csv(feature_path)

    combined_feature_count = _safe_count_csv(processed_dir / "cnn_features.csv")
    stage_feature_count = _safe_count_csv(processed_dir / "cnn_stage_features.csv")

    generated_files = [
        models_dir / f"{model_name}_best.keras"
        for model_name in comparison["model"].astype(str).tolist()
    ]
    generated_files.extend(
        [
            reports_dir / "cnn_evaluation_metrics.csv",
            reports_dir / "cnn_model_comparison.csv",
            processed_dir / "cnn_features.csv",
            processed_dir / "cnn_stage_features.csv",
        ]
    )

    missing_files = [str(path) for path in generated_files if not path.exists()]
    pipeline_status = "PASS" if not missing_files else "PARTIAL"

    lines = [
        "# CNN Pipeline Validation Report",
        "",
        f"Pipeline status: {pipeline_status}",
        "",
        "## Models trained",
    ]
    for model_name in comparison["model"].astype(str).tolist():
        model_meta = _load_meta(reports_dir, model_name)
        model_path = models_dir / f"{model_name}_best.keras"
        lines.append(
            "- "
            f"{model_name}: model_path={model_path}, "
            f"best_validation_accuracy={model_meta.get('best_validation_accuracy')}, "
            f"best_validation_loss={model_meta.get('best_validation_loss')}"
        )

    lines.extend(
        [
            "",
            f"Image count: {image_count}",
            f"Stage count: {stage_count}",
            "",
            "## Stage distribution",
        ]
    )
    for stage, count in stage_distribution.items():
        lines.append(f"- {stage}: {int(count)}")

    lines.extend(
        [
            "",
            f"Best model: {best_model}",
            f"Validation accuracy: {validation_accuracy}",
            f"Extracted feature count: {combined_feature_count}",
            f"Stage feature rows: {stage_feature_count}",
            "",
            "## Per-model feature files",
        ]
    )
    if feature_counts:
        for name, count in feature_counts.items():
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- None found")

    lines.extend(["", "## Generated files"])
    for path in generated_files:
        status = "exists" if path.exists() else "missing"
        lines.append(f"- {path}: {status}")

    if missing_files:
        lines.extend(["", "## Missing files"])
        for path in missing_files:
            lines.append(f"- {path}")

    report_path = reports_dir / "cnn_pipeline_validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    ensure_dir(reports_dir)

    metrics_path = reports_dir / "cnn_evaluation_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Missing cnn_evaluation_metrics.csv. Run evaluate_cnn_models.py first.")

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        logger.warning("cnn_evaluation_metrics.csv is empty.")
        return

    training_times = []
    for _, row in metrics.iterrows():
        model_name = row["model"]
        meta = _load_meta(reports_dir, model_name)
        training_times.append(meta.get("training_time_seconds", float("nan")))

    metrics = metrics.copy()
    metrics["training_time"] = training_times
    metrics = metrics.sort_values("f1_score", ascending=False).reset_index(drop=True)

    out_path = reports_dir / "cnn_model_comparison.csv"
    metrics.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    report_path = _write_validation_report(reports_dir, config, metrics)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
