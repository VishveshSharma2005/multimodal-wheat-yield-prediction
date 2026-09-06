from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


MODEL_NAMES = ["densenet121", "mobilenetv2", "inceptionv3", "vgg16", "xception"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_cnn(reports_dir: Path) -> tuple[str, float | None]:
    comparison_path = reports_dir / "cnn_model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        if not comparison.empty:
            metric_col = "accuracy" if "accuracy" in comparison.columns else "f1_score"
            row = comparison.sort_values(metric_col, ascending=False).iloc[0]
            return str(row["model"]), float(row[metric_col])

    best_model = ""
    best_acc = None
    for model_name in MODEL_NAMES:
        meta = _load_json(reports_dir / f"{model_name}_meta.json")
        acc = meta.get("best_validation_accuracy")
        if acc is None:
            continue
        if best_acc is None or float(acc) > best_acc:
            best_model = model_name
            best_acc = float(acc)
    return best_model or "pending", best_acc


def _metric_delta(ablation: pd.DataFrame, metric: str) -> float | None:
    if ablation.empty or metric not in ablation.columns:
        return None
    baseline = ablation[ablation["ablation"] == "Weather + Satellite + Handcrafted Images"]
    cnn = ablation[ablation["ablation"] == "Weather + Satellite + Handcrafted Images + CNN"]
    if baseline.empty or cnn.empty:
        return None
    base_val = baseline.iloc[0][metric]
    cnn_val = cnn.iloc[0][metric]
    if pd.isna(base_val) or pd.isna(cnn_val):
        return None
    if metric in {"RMSE", "MAPE"}:
        return float(base_val - cnn_val)
    return float(cnn_val - base_val)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    ensure_dir(reports_dir)

    best_model, best_validation_accuracy = _best_cnn(reports_dir)

    stage_path = processed_dir / "stage_features_with_split.csv"
    cnn_stage_path = processed_dir / "stage_features_with_cnn.csv"
    base_feature_count = len(pd.read_csv(stage_path).columns) if stage_path.exists() else 0
    cnn_feature_count = len(pd.read_csv(cnn_stage_path).columns) if cnn_stage_path.exists() else 0
    feature_count_increase = cnn_feature_count - base_feature_count if cnn_feature_count else 0

    ablation_path = reports_dir / "cnn_ablation_study.csv"
    ablation = pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()

    report_path = reports_dir / "cnn_improvement_report.md"
    lines = [
        "# CNN Improvement Report",
        "",
        f"Best CNN architecture: {best_model}",
        f"Best validation accuracy: {best_validation_accuracy}",
        "",
        "## Architectures compared",
    ]
    for model_name in MODEL_NAMES:
        meta = _load_json(reports_dir / f"{model_name}_meta.json")
        lines.append(f"- {model_name}: best_validation_accuracy={meta.get('best_validation_accuracy')}")

    lines.extend(
        [
            "",
            f"Feature count increase: {feature_count_increase}",
            f"RMSE improvement: {_metric_delta(ablation, 'RMSE')}",
            f"MAPE improvement: {_metric_delta(ablation, 'MAPE')}",
            f"R2 improvement: {_metric_delta(ablation, 'R2')}",
            "",
            f"Ablation source: {ablation_path}",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
