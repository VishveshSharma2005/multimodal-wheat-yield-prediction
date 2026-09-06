from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _sanitize_stage(stage: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stage)


def _feature_flags(row: pd.Series) -> tuple[int, int, int, int, int]:
    weather_cols = ["rain_sum", "cumulative_rain_mm", "tmin_mean", "tmax_mean", "mean_tmin_c", "mean_tmax_c", "gdd_cum"]
    satellite_cols = ["ndvi_last", "ndvi_mean", "ndvi_max", "ndvi_slope", "ndre_last", "ndre_mean", "evi_last", "evi_mean"]
    image_cols = [
        "image_green_pixel_ratio_stage_avg",
        "image_canopy_density_proxy_stage_avg",
        "image_vegetation_pixel_ratio_stage_avg",
        "image_brightness_mean_stage_avg",
        "image_saturation_mean_stage_avg",
        "image_texture_proxy_stage_avg",
        "image_count_stage",
    ]
    cnn_cols = [col for col in row.index if str(col).startswith("cnn_")]
    available_count = int(row.notna().sum())
    weather_flag = int(any(pd.notna(row.get(col)) for col in weather_cols if col in row.index))
    satellite_flag = int(any(pd.notna(row.get(col)) for col in satellite_cols if col in row.index))
    image_flag = int(any(pd.notna(row.get(col)) for col in image_cols if col in row.index))
    cnn_flag = int(any(pd.notna(row.get(col)) for col in cnn_cols))
    return available_count, weather_flag, satellite_flag, image_flag, cnn_flag


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    if y_true.empty:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
    y_true_arr = y_true.to_numpy(dtype=float)
    y_pred_arr = y_pred.to_numpy(dtype=float)
    denom = np.where(y_true_arr == 0, np.nan, y_true_arr)
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mape": float(np.nanmean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100.0),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }


def _prepare_split_df(df: pd.DataFrame, split: str, mehsana_only: bool) -> pd.DataFrame:
    split_df = df[df["split"] == split].copy()
    if mehsana_only and "district" in split_df.columns:
        split_df = split_df[split_df["district"].astype(str).str.contains("MEHSANA", case=False, na=False)]
    return split_df


def _load_model_for_stage(stage: str, models_dir: Path, combined_model):
    stage_path = models_dir / f"best_model_{_sanitize_stage(stage)}.joblib"
    if stage_path.exists():
        return load(stage_path), stage_path
    if combined_model is not None:
        return combined_model, models_dir / "best_combined_model.joblib"
    return None, None


def _predict_for_split(df: pd.DataFrame, split: str, models_dir: Path, out_path: Path, mehsana_only: bool) -> pd.DataFrame:
    rows = []
    split_df = _prepare_split_df(df, split, mehsana_only)

    if split_df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "case_id",
                "district",
                "season_year_start",
                "season_year_end",
                "stage",
                "stage_end_das",
                "actual_yield_kg_ha",
                "predicted_yield_kg_ha",
                "absolute_error",
                "percentage_error",
                "model_name",
                "evaluation_status",
            ]
        )
        empty_df.to_csv(out_path, index=False)
        logger.warning("No rows for split=%s. Wrote empty %s", split, out_path)
        return empty_df

    combined_model_path = models_dir / "best_combined_model.joblib"
    combined_model = load(combined_model_path) if combined_model_path.exists() else None

    for stage in sorted(split_df["stage"].dropna().unique().tolist()):
        stage_df = split_df[split_df["stage"] == stage].copy()
        model, used_model_path = _load_model_for_stage(stage, models_dir, combined_model)
        if model is None:
            logger.warning("Missing model for stage %s", stage)
            continue

        try:
            preds = model.predict(stage_df)
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", stage, exc)
            continue

        model_name = model.named_steps.get("model").__class__.__name__ if hasattr(model, "named_steps") else type(model).__name__
        for idx, pred in zip(stage_df.index, preds):
            row = stage_df.loc[idx]
            actual = row.get("yield_kg_ha")
            actual_available = int(pd.notna(actual))
            abs_err = np.nan
            pct_err = np.nan
            if actual_available:
                abs_err = float(abs(actual - pred))
                pct_err = float(abs_err / actual * 100.0) if actual != 0 else np.nan

            available_count, weather_flag, satellite_flag, image_flag, cnn_flag = _feature_flags(row)
            rows.append(
                {
                    "case_id": row.get("case_id"),
                    "district": row.get("district"),
                    "season_year_start": row.get("season_year_start"),
                    "season_year_end": row.get("season_year_end"),
                    "stage": row.get("stage"),
                    "stage_end_das": row.get("stage_end_das"),
                    "actual_yield_kg_ha": actual,
                    "predicted_yield_kg_ha": float(pred),
                    "absolute_error": abs_err,
                    "percentage_error": pct_err,
                    "model_name": model_name,
                    "model_path": str(used_model_path) if used_model_path else "",
                    "features_available_count": available_count,
                    "weather_available_flag": weather_flag,
                    "satellite_available_flag": satellite_flag,
                    "image_available_flag": image_flag,
                    "cnn_available_flag": cnn_flag,
                    "actual_available": actual_available,
                    "evaluation_status": "available" if actual_available else "pending_actual_apy",
                }
            )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df["prediction_changed_from_previous_stage"] = 0
        for case_id, group in out_df.groupby("case_id"):
            group = group.sort_values("stage_end_das")
            prev = None
            for idx, pred in zip(group.index, group["predicted_yield_kg_ha"]):
                if prev is None:
                    out_df.loc[idx, "prediction_changed_from_previous_stage"] = 0
                else:
                    out_df.loc[idx, "prediction_changed_from_previous_stage"] = int(pred != prev)
                prev = pred

    if split == "prediction_2025" and not out_df.empty:
        for case_id, group in out_df.groupby("case_id"):
            if group["predicted_yield_kg_ha"].nunique(dropna=True) <= 1:
                logger.warning("Flat prediction likely due to weak/missing stage-dependent features for %s", case_id)

    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(out_df))
    return out_df


def _write_prediction_2025_report(prediction_df: pd.DataFrame, mehsana_df: pd.DataFrame, report_path: Path) -> None:
    available_df = prediction_df[prediction_df["actual_available"] == 1].copy() if not prediction_df.empty else pd.DataFrame()
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Prediction 2025 Evaluation\n\n")
        f.write(f"- Prediction rows: {len(prediction_df)}\n")
        f.write(f"- Rows with actual APY: {len(available_df)}\n")
        if available_df.empty:
            f.write("- Actual APY is unavailable. Evaluation status: pending_actual_apy\n")
            f.write("- MAPE: N/A\n")
            f.write("- RMSE: N/A\n")
            f.write("- MAE: N/A\n")
            f.write("- R2: N/A\n")
        else:
            metrics = _compute_metrics(available_df["actual_yield_kg_ha"], available_df["predicted_yield_kg_ha"])
            f.write(f"- MAPE: {metrics['mape']:.4f}\n")
            f.write(f"- RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"- MAE: {metrics['mae']:.4f}\n")
            f.write(f"- R2: {metrics['r2']:.4f}\n")
        f.write("\n## Actual availability by district\n")
        if available_df.empty or "district" not in prediction_df.columns:
            f.write("- Pending actual APY\n")
        else:
            district_rows = []
            for district, group in prediction_df.groupby("district"):
                actual_rows = group[group["actual_available"] == 1]
                if actual_rows.empty:
                    district_rows.append({"district": district, "rows": len(group), "actual_rows": 0, "mape": ""})
                    continue
                metrics = _compute_metrics(actual_rows["actual_yield_kg_ha"], actual_rows["predicted_yield_kg_ha"])
                district_rows.append(
                    {
                        "district": district,
                        "rows": len(group),
                        "actual_rows": len(actual_rows),
                        "mape": round(metrics["mape"], 4) if pd.notna(metrics["mape"]) else "",
                        "rmse": round(metrics["rmse"], 4) if pd.notna(metrics["rmse"]) else "",
                        "mae": round(metrics["mae"], 4) if pd.notna(metrics["mae"]) else "",
                        "r2": round(metrics["r2"], 4) if pd.notna(metrics["r2"]) else "",
                    }
                )
            district_df = pd.DataFrame(district_rows)
            try:
                f.write(district_df.to_markdown(index=False))
            except Exception:
                f.write(district_df.to_string(index=False))
            f.write("\n")

        f.write("\n## Mehsana 2025 prediction\n")
        if mehsana_df.empty:
            f.write("- No Mehsana 2025 rows found.\n")
        elif mehsana_df["actual_available"].sum() == 0:
            f.write("- Mehsana 2025 prediction generated, but actual APY is pending.\n")
        else:
            mehsana_metrics = _compute_metrics(mehsana_df.loc[mehsana_df["actual_available"] == 1, "actual_yield_kg_ha"], mehsana_df.loc[mehsana_df["actual_available"] == 1, "predicted_yield_kg_ha"])
            f.write(f"- Mehsana actual rows: {int(mehsana_df['actual_available'].sum())}\n")
            f.write(f"- Mehsana MAPE: {mehsana_metrics['mape']:.4f}\n")
            f.write(f"- Mehsana RMSE: {mehsana_metrics['rmse']:.4f}\n")
            f.write(f"- Mehsana MAE: {mehsana_metrics['mae']:.4f}\n")
            f.write(f"- Mehsana R2: {mehsana_metrics['r2']:.4f}\n")


def main() -> None:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    ensure_dir(processed_dir)
    ensure_dir(reports_dir)

    stage_path = processed_dir / "stage_features_with_cnn.csv"
    if not stage_path.exists():
        stage_path = processed_dir / "stage_features_with_split.csv"
    if not stage_path.exists():
        raise FileNotFoundError("Missing stage_features_with_cnn.csv or stage_features_with_split.csv")

    df = pd.read_csv(stage_path)

    cases_path = interim_dir / "cases.csv"
    if cases_path.exists():
        cases = pd.read_csv(cases_path)
        for col in ["district", "season_year_start", "season_year_end", "yield_kg_ha", "actual_available", "evaluation_status"]:
            if col not in df.columns and col in cases.columns:
                df = df.merge(cases[["case_id", col]], on="case_id", how="left")

    if "yield_kg_ha" not in df.columns:
        targets_path = processed_dir / "targets.csv"
        if targets_path.exists():
            targets = pd.read_csv(targets_path)
            df = df.merge(targets[["case_id", "yield_kg_ha"]], on="case_id", how="left")

    if "actual_available" not in df.columns:
        df["actual_available"] = df["yield_kg_ha"].notna().astype(int)
    if "evaluation_status" not in df.columns:
        df["evaluation_status"] = df["actual_available"].map(lambda flag: "available" if int(flag) == 1 else "pending_actual_apy")

    models_dir = ROOT / "models" / "regression"
    val_out = processed_dir / "validation_stagewise_predictions.csv"
    prediction_out = processed_dir / "prediction_2025_stagewise_predictions.csv"
    mehsana_out = processed_dir / "mehsana_2025_stagewise_predictions.csv"

    _predict_for_split(df, "validation", models_dir, val_out, mehsana_only=False)
    prediction_df = _predict_for_split(df, "prediction_2025", models_dir, prediction_out, mehsana_only=False)
    mehsana_df = _predict_for_split(df, "prediction_2025", models_dir, mehsana_out, mehsana_only=True)

    report_path = reports_dir / "prediction_2025_evaluation.md"
    _write_prediction_2025_report(prediction_df, mehsana_df, report_path)

    if prediction_df.empty or prediction_df["actual_available"].sum() == 0:
        print("Prediction 2025 actual APY: pending_actual_apy")
    else:
        print("Prediction 2025 actual APY detected successfully.")
        print(f"Prediction 2025 actual APY rows: {int(prediction_df['actual_available'].sum())}")
        print(f"Prediction 2025 MAPE: {_compute_metrics(prediction_df.loc[prediction_df['actual_available'] == 1, 'actual_yield_kg_ha'], prediction_df.loc[prediction_df['actual_available'] == 1, 'predicted_yield_kg_ha'])['mape']:.4f}")


if __name__ == "__main__":
    main()
