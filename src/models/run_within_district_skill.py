"""Within-district (anomaly) skill of the stage-wise validation predictions.

Why
---
The target is a district-level yield and districts differ systematically and
persistently. A model that only learns "which district is this" can therefore
post a very high R2 without predicting anything about the *season*. Because
every district appears in every year, that district effect is present in both
the training and the validation split, so ordinary R2 cannot separate the two.

What this does
--------------
Removes the district mean, computed from the **training seasons only** (2020-21
to 2022-23), from both the observed and the predicted 2023-24 yield, and scores
the remainder:

    anomaly_actual = actual_2023_24  - district_mean(training seasons)
    anomaly_pred   = predicted       - district_mean(training seasons)

Scoring the anomalies answers: *given that we already know the district's
typical yield, does the model add information about this particular season?*

The reference point is the trivial anomaly predictor - predict zero anomaly,
i.e. predict the district's historical mean. Its anomaly RMSE is the standard
deviation of the observed anomaly. A model with anomaly R2 <= 0 has no
within-district skill.

Using training-season means keeps the analysis leakage-free: no validation-year
information enters the centring.

Outputs
-------
    reports/model_results/within_district_skill.csv
    reports/summaries/within_district_skill.md
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)

TRAIN_YEARS = [2020, 2021, 2022]
VALIDATION_YEAR = 2023

STAGE_ORDER = [
    "sowing", "early_vegetative", "tillering", "stem_elongation", "booting_heading",
    "flowering", "grain_filling_initial", "grain_filling", "maturity_preharvest",
]


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    config = load_config()
    processed = Path(config["paths"]["processed_dir"])
    reports = Path(config["paths"]["reports_dir"]) / "model_results"
    summaries = Path(config["paths"]["reports_dir"]) / "summaries"
    ensure_dir(reports)
    ensure_dir(summaries)

    pred_path = processed / "validation_stagewise_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}. Run src/models/predict_stagewise.py first.")
    preds = pd.read_csv(pred_path)

    stage_path = processed / "stage_features.csv"
    if not stage_path.exists():
        raise FileNotFoundError(f"Missing {stage_path}.")
    stage = pd.read_csv(stage_path)

    # District climatology from TRAINING seasons only - no validation information.
    train_cases = (stage[stage["season_year_start"].isin(TRAIN_YEARS)]
                   .drop_duplicates(["district", "season_year_start"]))
    clim = train_cases.groupby("district")["yield_kg_ha"].mean()
    global_mean = float(train_cases["yield_kg_ha"].mean())
    logger.info("District climatology from %s: %d districts, global mean %.2f",
                TRAIN_YEARS, clim.size, global_mean)

    d = preds.copy()
    d["district_train_mean"] = d["district"].map(clim).fillna(global_mean)
    d["anomaly_actual"] = d["actual_yield_kg_ha"] - d["district_train_mean"]
    d["anomaly_pred"] = d["predicted_yield_kg_ha"] - d["district_train_mean"]

    rows = []

    raw = _scores(d["actual_yield_kg_ha"], d["predicted_yield_kg_ha"])
    rows.append({"scope": "all stages", "stage": "", "space": "raw yield",
                 "predictor": "stage-wise model", **raw})

    clim_raw = _scores(d["actual_yield_kg_ha"], d["district_train_mean"])
    rows.append({"scope": "all stages", "stage": "", "space": "raw yield",
                 "predictor": "district historical mean (reference)", **clim_raw})

    anom = _scores(d["anomaly_actual"], d["anomaly_pred"])
    rows.append({"scope": "all stages", "stage": "", "space": "within-district anomaly",
                 "predictor": "stage-wise model", **anom})

    zero = _scores(d["anomaly_actual"], np.zeros(len(d)))
    rows.append({"scope": "all stages", "stage": "", "space": "within-district anomaly",
                 "predictor": "zero anomaly (= district mean)", **zero})

    for stg in STAGE_ORDER:
        sub = d[d["stage"] == stg]
        if sub.empty:
            continue
        rows.append({"scope": "per stage", "stage": stg, "space": "within-district anomaly",
                     "predictor": "stage-wise model", **_scores(sub["anomaly_actual"], sub["anomaly_pred"])})

    out = pd.DataFrame(rows)
    for c in ("mae", "rmse", "r2"):
        out[c] = out[c].round(4)
    out_path = reports / "within_district_skill.csv"
    out.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    _write_summary(summaries / "within_district_skill.md", d, raw, clim_raw, anom, zero, out)


def _f(v) -> str:
    return "n/a" if pd.isna(v) else f"{v:.2f}"


def _write_summary(path: Path, d: pd.DataFrame, raw: dict, clim_raw: dict,
                   anom: dict, zero: dict, table: pd.DataFrame) -> None:
    sd_anom = float(np.nanstd(d["anomaly_actual"]))
    with path.open("w", encoding="utf-8") as f:
        f.write("# Within-district (anomaly) skill\n\n")
        f.write("Season: **2023-24 validation**, 297 stage-level predictions "
                "(33 districts x 9 stages), stage-wise GradientBoosting models.\n\n")
        f.write("## What this measures\n\n")
        f.write("District yields differ systematically and persistently, and every district\n"
                "appears in both the training and the validation split. Ordinary R2 therefore\n"
                "cannot tell \"knows the agronomy of this season\" apart from \"knows which\n"
                "district this is\". Removing each district's **training-season** mean from both\n"
                "the observed and the predicted value isolates the season-to-season signal.\n\n")
        f.write("The centring uses training seasons only (2020-21..2022-23), so no\n"
                "validation-year information enters it.\n\n")

        f.write("## Results\n\n")
        f.write("### Raw yield space\n\n")
        f.write("| Predictor | n | MAE | RMSE | R2 |\n|---|---|---|---|---|\n")
        f.write(f"| Stage-wise model | {raw['n']} | {_f(raw['mae'])} | {_f(raw['rmse'])} | {raw['r2']:.4f} |\n")
        f.write(f"| District historical mean (reference) | {clim_raw['n']} | {_f(clim_raw['mae'])} | "
                f"{_f(clim_raw['rmse'])} | {clim_raw['r2']:.4f} |\n\n")

        f.write("### Within-district anomaly space\n\n")
        f.write("| Predictor | n | MAE | RMSE | R2 |\n|---|---|---|---|---|\n")
        f.write(f"| Stage-wise model | {anom['n']} | {_f(anom['mae'])} | {_f(anom['rmse'])} | {anom['r2']:.4f} |\n")
        f.write(f"| Zero anomaly (= district mean) | {zero['n']} | {_f(zero['mae'])} | "
                f"{_f(zero['rmse'])} | {zero['r2']:.4f} |\n\n")
        f.write(f"Observed anomaly standard deviation: **{sd_anom:.2f} kg/ha**.\n\n")

        f.write("### Per stage (anomaly space)\n\n")
        f.write("| Stage | n | MAE | RMSE | R2 |\n|---|---|---|---|---|\n")
        for _, r in table[(table.scope == "per stage")].iterrows():
            f.write(f"| {r['stage']} | {int(r['n'])} | {_f(r['mae'])} | {_f(r['rmse'])} | "
                    f"{'n/a' if pd.isna(r['r2']) else f'{r.r2:.4f}'} |\n")

        f.write("\n## Interpretation\n\n")
        if pd.notna(anom["r2"]) and anom["r2"] > 0:
            f.write(f"- Anomaly R2 = **{anom['r2']:.4f} > 0**: after removing the district effect the\n"
                    "  model retains some skill at predicting the season-specific deviation.\n")
        else:
            f.write(f"- Anomaly R2 = **{anom['r2']:.4f} <= 0**: once each district's historical mean is\n"
                    "  removed, the model explains **none** of the remaining season-to-season\n"
                    "  variation. Its headline R2 in raw-yield space is carried by district identity,\n"
                    "  not by the weather / satellite / calendar features.\n")
        f.write(f"- For comparison, simply predicting each district's historical mean scores\n"
                f"  RMSE {_f(clim_raw['rmse'])} in raw space against the model's {_f(raw['rmse'])}.\n")
        f.write("\n## Limits of this analysis\n\n")
        f.write("- 33 districts x 1 validation season = 33 independent case-level anomalies; the 297\n"
                "  stage rows are not independent (9 rows share one target per case).\n"
                "- This is a predictive-skill decomposition, **not** a causal analysis. It does not\n"
                "  say the environmental variables are irrelevant to yield, only that this model on\n"
                "  this data does not extract usable season-level signal from them.\n")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
