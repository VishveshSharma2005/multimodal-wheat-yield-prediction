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


def _safe_mape(y_true, y_pred):
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    figures_dir = Path(config["paths"]["reports_dir"]) / "figures"
    ensure_dir(reports_dir)
    ensure_dir(figures_dir)

    val_path = processed_dir / "validation_stagewise_predictions.csv"
    pred_path = processed_dir / "mehsana_2025_stagewise_predictions.csv"

    if not val_path.exists():
        logger.warning("Missing %s", val_path)
        return

    val_df = pd.read_csv(val_path)
    if val_df.empty:
        logger.warning("Validation predictions are empty.")
        return

    metrics_rows = []
    for stage, group in val_df.groupby("stage"):
        actual = group["actual_yield_kg_ha"].to_numpy(dtype=float)
        pred = group["predicted_yield_kg_ha"].to_numpy(dtype=float)
        mask = np.isfinite(actual) & np.isfinite(pred)
        if mask.sum() == 0:
            continue
        actual = actual[mask]
        pred = pred[mask]
        mae = float(np.mean(np.abs(actual - pred)))
        mse = float(np.mean((actual - pred) ** 2))
        rmse = float(np.sqrt(mse))
        mape = _safe_mape(actual, pred)
        ss_res = float(np.sum((actual - pred) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan
        stage_end = group["stage_end_das"].dropna().median() if "stage_end_das" in group.columns else np.nan
        metrics_rows.append(
            {
                "stage": stage,
                "stage_end_das": stage_end,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "r2": r2,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values("stage_end_das")

    report_path = reports_dir / "stagewise_improvement_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Stage-wise Improvement Report\n\n")
        f.write("## Validation metrics by stage (2023-24)\n")
        if not metrics_df.empty:
            try:
                f.write(metrics_df.to_markdown(index=False) + "\n\n")
            except Exception:
                f.write(metrics_df.to_string(index=False) + "\n\n")
            if len(metrics_df) >= 2:
                rmse_start = metrics_df.iloc[0]["rmse"]
                rmse_end = metrics_df.iloc[-1]["rmse"]
                delta = rmse_end - rmse_start
                if abs(delta) < 1e-3:
                    trend = "flat"
                elif delta < 0:
                    trend = "improves"
                else:
                    trend = "worsens"
                f.write(f"- RMSE trend from early to late stages: {trend}.\n")
                best_row = metrics_df.loc[metrics_df["rmse"].idxmin()]
                worst_row = metrics_df.loc[metrics_df["rmse"].idxmax()]
                f.write(f"- Best stage by RMSE: {best_row['stage']} ({best_row['rmse']:.2f}).\n")
                f.write(f"- Worst stage by RMSE: {worst_row['stage']} ({worst_row['rmse']:.2f}).\n")
                f.write("- Stage-wise improvement is expected but not guaranteed if late-stage satellite/drone features are missing.\n")
        else:
            f.write("- No valid metrics available.\n")

    try:
        import matplotlib.pyplot as plt

        if not metrics_df.empty:
            plt.figure(figsize=(8, 4))
            plt.plot(metrics_df["stage_end_das"], metrics_df["rmse"], marker="o")
            plt.xlabel("Stage end DAS")
            plt.ylabel("RMSE")
            plt.title("Stage-wise RMSE trend")
            plt.tight_layout()
            plt.savefig(figures_dir / "stagewise_rmse_trend.png")
            plt.close()

            plt.figure(figsize=(8, 4))
            plt.plot(metrics_df["stage_end_das"], metrics_df["mae"], marker="o")
            plt.xlabel("Stage end DAS")
            plt.ylabel("MAE")
            plt.title("Stage-wise MAE trend")
            plt.tight_layout()
            plt.savefig(figures_dir / "stagewise_mae_trend.png")
            plt.close()
    except Exception as exc:
        logger.warning("Plotting failed: %s", exc)

    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        if not pred_df.empty and "stage_end_das" in pred_df.columns:
            if pred_df["predicted_yield_kg_ha"].nunique(dropna=True) <= 1:
                with report_path.open("a", encoding="utf-8") as f:
                    f.write("\n- Flat Mehsana prediction detected. Causes may be missing satellite/features or model fallback.\n")
            try:
                import matplotlib.pyplot as plt

                pred_df = pred_df.sort_values("stage_end_das")
                plt.figure(figsize=(8, 4))
                plt.plot(pred_df["stage_end_das"], pred_df["predicted_yield_kg_ha"], marker="o", label="predicted")
                actual = pred_df["actual_yield_kg_ha"].dropna()
                if not actual.empty:
                    plt.plot(pred_df["stage_end_das"], pred_df["actual_yield_kg_ha"], marker="o", label="actual")
                plt.xlabel("Stage end DAS")
                plt.ylabel("Yield (kg/ha)")
                plt.title("Mehsana 2025 stage-wise prediction")
                plt.legend()
                plt.tight_layout()
                plt.savefig(figures_dir / "mehsana_2025_stagewise_prediction.png")
                plt.close()
            except Exception as exc:
                logger.warning("Prediction plot failed: %s", exc)

    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
