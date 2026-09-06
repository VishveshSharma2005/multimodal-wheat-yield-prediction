from __future__ import annotations

from pathlib import Path
import copy
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.features.feature_policy import is_excluded
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

RANDOM_SEED = 42


def _set_seeds(torch, seed: int = RANDOM_SEED) -> None:
    """Make a training run reproducible.

    Seeds Python, NumPy and PyTorch, and forces cuDNN into its deterministic
    kernels. Without this the LSTM produced a different result on every run
    (an earlier run recorded RMSE 202.51, a later one 166.69).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def _safe_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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


def _select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    numeric_cols = [
        col
        for col in df.columns
        if (not _is_excluded(col)) and pd.api.types.is_numeric_dtype(df[col])
    ]
    dropped = []
    selected = []
    for col in numeric_cols:
        pct = _missing_pct(df[col])
        if pct > 90.0:
            dropped.append({"feature": col, "reason": f"missing {pct:.2f}%"})
        else:
            selected.append(col)
    return selected, dropped


def _load_dataset() -> tuple[pd.DataFrame, Path, Path, Path, Path, Path]:
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    figures_dir = Path(config["paths"]["reports_dir"]) / "figures"
    models_dir = ROOT / "models" / "deep_learning"

    ensure_dir(reports_dir)
    ensure_dir(figures_dir)
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
        df = df.merge(
            cases[["case_id", "district", "season_year_start", "season_year_end"]],
            on="case_id",
            how="left",
        )

    targets_path = processed_dir / "targets.csv"
    if targets_path.exists() and "yield_kg_ha" not in df.columns:
        targets = pd.read_csv(targets_path)
        df = df.merge(targets[["case_id", "yield_kg_ha"]], on="case_id", how="left")

    if "split" not in df.columns or df["split"].isna().all():
        if "season_year_start" not in df.columns:
            raise ValueError("Missing split and season_year_start columns.")
        df["split"] = _compute_split(df["season_year_start"])

    return df, stage_path, processed_dir, reports_dir, figures_dir, models_dir


def _sort_group(group: pd.DataFrame) -> pd.DataFrame:
    if "stage_end_das" in group.columns:
        return group.sort_values(["stage_end_das", "stage_start_das"], kind="mergesort")
    if "stage_start_das" in group.columns:
        return group.sort_values(["stage_start_das"], kind="mergesort")
    return group


def _build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    train_rows = []
    val_rows = []
    pred_rows = []

    for case_id, group in df.groupby("case_id"):
        group = _sort_group(group)
        split_mode = group["split"].mode(dropna=True)
        split = split_mode.iloc[0] if not split_mode.empty else group["split"].dropna().iloc[0]
        if split not in {"train", "validation", "prediction_2025"}:
            continue
        features = group[feature_cols].to_numpy(dtype=float, copy=True)
        yield_series = group["yield_kg_ha"].dropna() if "yield_kg_ha" in group.columns else pd.Series([], dtype=float)
        yield_value = yield_series.iloc[0] if not yield_series.empty else np.nan

        if split in {"train", "validation"} and np.isnan(yield_value):
            continue

        row = {
            "case_id": case_id,
            "features": features,
            "length": len(features),
            "yield_kg_ha": yield_value,
            "district": group["district"].dropna().iloc[0] if "district" in group.columns else "",
            "season_year_start": group["season_year_start"].dropna().iloc[0] if "season_year_start" in group.columns else np.nan,
            "season_year_end": group["season_year_end"].dropna().iloc[0] if "season_year_end" in group.columns else np.nan,
        }
        if split == "train":
            train_rows.append(row)
        elif split == "validation":
            val_rows.append(row)
        elif split == "prediction_2025":
            pred_rows.append(row)

    return train_rows, val_rows, pred_rows


def _pad_sequences(rows: list[dict], max_len: int, n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0, max_len, n_features), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    X = np.zeros((len(rows), max_len, n_features), dtype=np.float32)
    lengths = np.zeros((len(rows),), dtype=np.int64)
    y = np.zeros((len(rows),), dtype=np.float32)
    for i, row in enumerate(rows):
        seq = row["features"].astype(np.float32, copy=False)
        seq_len = min(len(seq), max_len)
        X[i, :seq_len, :] = seq[:seq_len]
        lengths[i] = seq_len
        y[i] = row.get("yield_kg_ha", np.nan)
    return X, lengths, y


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _write_summary(
    summary_path: Path,
    stage_path: Path,
    n_train: int,
    n_val: int,
    n_pred: int,
    n_steps: int,
    n_features: int,
    candidate_features: int,
    dropped_features: list[dict],
    final_features: list[str],
    metrics: dict | None,
    pred_path: Path | None,
    note: str | None,
) -> None:
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# LSTM Sequence Model Summary\n\n")
        f.write(f"- Input file: {stage_path}\n")
        f.write(f"- Train sequences: {n_train}\n")
        f.write(f"- Validation sequences: {n_val}\n")
        f.write(f"- Prediction sequences: {n_pred}\n")
        f.write(f"- Timesteps per sequence (max): {n_steps}\n")
        f.write(f"- Candidate features: {candidate_features}\n")
        f.write(f"- Feature count (final): {n_features}\n")
        f.write("- Architecture: LSTM input -> hidden_size=64, num_layers=2, dropout=0.2, output=1\n")
        f.write("- Training settings: epochs=100, batch_size=16, lr=0.001, optimizer=Adam, loss=MSE, early_stopping=15\n\n")
        if dropped_features:
            f.write("## Dropped features\n")
            for item in dropped_features:
                f.write(f"- {item['feature']}: {item['reason']}\n")
            f.write("\n")
        if final_features:
            f.write("## Final feature list\n")
            f.write("- " + ", ".join(final_features) + "\n\n")
            groups = sorted({_feature_group(col) for col in final_features})
            f.write("## Feature groups used\n")
            f.write("- " + ", ".join(groups) + "\n\n")
            if not any(group in {"weather", "satellite", "image", "cnn"} for group in groups):
                f.write("- Warning: no weather/satellite/image/cnn features in final LSTM inputs.\n\n")
        if metrics:
            f.write("## Validation metrics\n")
            f.write(f"- MAE: {metrics['mae']:.4f}\n")
            f.write(f"- MSE: {metrics['mse']:.4f}\n")
            f.write(f"- RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"- MAPE: {metrics['mape']:.4f}\n")
            f.write(f"- R2: {metrics['r2']:.4f}\n\n")
        else:
            f.write("## Validation metrics\n- Not available.\n\n")
        if pred_path:
            f.write(f"- 2025 predictions: {pred_path}\n")
        else:
            f.write("- 2025 predictions: none\n")
        if note:
            f.write(f"\n## Notes\n- {note}\n")
        f.write("\n## Limitations\n")
        f.write("- LSTM learns temporal sequences but still depends on feature quality.\n")
        f.write("- Baseline ML models are tabular and do not model temporal progression.\n")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except Exception:
        reports_dir = Path(load_config()["paths"]["reports_dir"]) / "model_results"
        ensure_dir(reports_dir)
        summary_path = reports_dir / "lstm_summary.md"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("# LSTM Sequence Model Summary\n\n")
            f.write("- PyTorch is not installed.\n")
            f.write("- Install with: pip install torch\n")
        return

    _set_seeds(torch)
    logger.info("Seeded Python/NumPy/PyTorch with %d for a reproducible run.", RANDOM_SEED)

    df, stage_path, processed_dir, reports_dir, figures_dir, models_dir = _load_dataset()
    candidate_features = 0
    dropped_features: list[dict] = []
    feature_cols: list[str] = []

    if "yield_kg_ha" not in df.columns:
        _write_summary(
            reports_dir / "lstm_summary.md",
            stage_path,
            0,
            0,
            0,
            0,
            0,
            candidate_features,
            dropped_features,
            feature_cols,
            None,
            None,
            "Missing yield_kg_ha after merging targets.",
        )
        return

    df = df[df["split"].isin(["train", "validation", "prediction_2025"])].copy()

    numeric_cols, dropped_features = _select_feature_columns(df)
    candidate_features = len(numeric_cols) + len(dropped_features)

    categorical_columns = [col for col in ["district", "crop_name", "season_type", "stage_name"] if col in df.columns]
    if not numeric_cols and not categorical_columns:
        _write_summary(
            reports_dir / "lstm_summary.md",
            stage_path,
            0,
            0,
            0,
            0,
            0,
            candidate_features,
            dropped_features,
            [],
            None,
            None,
            "No numeric feature columns available for LSTM training.",
        )
        return

    district_features = []
    if categorical_columns:
        encoder = _safe_onehot_encoder()
        train_mask = df["split"] == "train"
        train_categorical = df.loc[train_mask, categorical_columns].fillna("Unknown").astype(str).to_numpy()
        encoder.fit(train_categorical)
        all_categorical = df[categorical_columns].fillna("Unknown").astype(str).to_numpy()
        encoded = encoder.transform(all_categorical)
        def _sanitize(val: str) -> str:
            return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(val))

        encoded_cols = []
        for col_name, categories in zip(categorical_columns, encoder.categories_):
            encoded_cols.extend([f"{col_name}_{_sanitize(val)}" for val in categories])
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        df = pd.concat([df, encoded_df], axis=1)
        district_features = encoded_cols

    feature_cols = numeric_cols + district_features
    selected_path = reports_dir / "lstm_selected_features.csv"
    selected_rows = []
    for col in numeric_cols:
        selected_rows.append(
            {
                "feature": col,
                "status": "selected",
                "reason": "numeric",
                "feature_group": _feature_group(col),
            }
        )
    for item in dropped_features:
        selected_rows.append(
            {
                "feature": item["feature"],
                "status": "dropped",
                "reason": item["reason"],
                "feature_group": _feature_group(item["feature"]),
            }
        )
    for col in district_features:
        selected_rows.append(
            {
                "feature": col,
                "status": "selected",
                "reason": "categorical one-hot",
                "feature_group": _feature_group(col),
            }
        )
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    train_rows, val_rows, pred_rows = _build_sequences(df, feature_cols)

    n_train = len(train_rows)
    n_val = len(val_rows)
    n_pred = len(pred_rows)

    if n_train == 0:
        _write_summary(
            reports_dir / "lstm_summary.md",
            stage_path,
            n_train,
            n_val,
            n_pred,
            0,
            len(feature_cols),
            candidate_features,
            dropped_features,
            feature_cols,
            None,
            None,
            "No training sequences available.",
        )
        return

    # Fit imputer and scaler on training data only
    train_stack = np.vstack([row["features"] for row in train_rows])
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train_stack)
    train_scaled = scaler.fit_transform(train_imputed)

    def _transform_rows(rows: list[dict]) -> list[dict]:
        if not rows:
            return rows
        stacked = np.vstack([row["features"] for row in rows])
        stacked = imputer.transform(stacked)
        stacked = scaler.transform(stacked)
        idx = 0
        for row in rows:
            length = row["length"]
            row["features"] = stacked[idx : idx + length]
            idx += length
        return rows

    train_rows = _transform_rows(train_rows)
    val_rows = _transform_rows(val_rows)
    pred_rows = _transform_rows(pred_rows)

    max_len = max([row["length"] for row in train_rows + val_rows + pred_rows])
    n_features = len(feature_cols)

    X_train, len_train, y_train = _pad_sequences(train_rows, max_len, n_features)
    X_val, len_val, y_val = _pad_sequences(val_rows, max_len, n_features)
    X_pred, len_pred, _ = _pad_sequences(pred_rows, max_len, n_features)

    y_mean = float(np.nanmean(y_train))
    y_std = float(np.nanstd(y_train))
    if y_std == 0 or np.isnan(y_std):
        y_std = 1.0
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std if n_val > 0 else y_val

    class SequenceDataset(Dataset):
        def __init__(self, X, lengths, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.lengths = torch.tensor(lengths, dtype=torch.int64)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.lengths[idx], self.y[idx]

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                batch_first=True,
            )
            self.fc = nn.Linear(64, 1)

        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            last_hidden = h_n[-1]
            return self.fc(last_hidden).squeeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # A seeded generator makes the shuffle order reproducible run to run.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        SequenceDataset(X_train, len_train, y_train_norm),
        batch_size=16,
        shuffle=True,
        generator=loader_generator,
    )

    val_loader = None
    if n_val > 0:
        val_loader = DataLoader(
            SequenceDataset(X_val, len_val, y_val_norm),
            batch_size=16,
            shuffle=False,
        )

    best_val_rmse = np.inf
    best_state = None
    best_epoch = None
    patience = 15
    patience_left = patience
    logs = []

    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_len, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_len = batch_len.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x, batch_len)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = np.nan
        val_rmse = np.nan

        if val_loader is not None:
            model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for batch_x, batch_len, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_len = batch_len.to(device)
                    preds = model(batch_x, batch_len)
                    val_preds.append(preds.cpu().numpy())
                    val_true.append(batch_y.numpy())
            if val_preds:
                val_preds = np.concatenate(val_preds)
                val_true = np.concatenate(val_true)
                val_loss = float(np.mean((val_preds - val_true) ** 2))
                val_rmse = float(np.sqrt(val_loss))
                val_preds_denorm = val_preds * y_std + y_mean
                val_true_denorm = val_true * y_std + y_mean
                val_rmse = float(np.sqrt(np.mean((val_preds_denorm - val_true_denorm) ** 2)))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    # deepcopy is essential: state_dict() returns tensors that
                    # alias the live parameters, so without it later epochs
                    # mutate this snapshot in place and the "best" checkpoint
                    # silently becomes the final epoch.
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    patience_left = patience
                else:
                    patience_left -= 1
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
            }
        )
        if val_loader is not None and patience_left <= 0:
            break

    log_path = reports_dir / "lstm_training_log.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = logs[-1]["epoch"] if logs else None

    model.load_state_dict(best_state)
    logger.info(
        "Restored best checkpoint: epoch %s (validation RMSE %.4f) out of %d epochs run.",
        best_epoch,
        best_val_rmse,
        len(logs),
    )
    model_path = models_dir / "lstm_yield_model.pt"
    torch.save(
        {
            "model_state": best_state,
            "input_size": n_features,
            "max_len": max_len,
            "feature_columns": feature_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "imputer_statistics": imputer.statistics_.tolist(),
            "target_mean": y_mean,
            "target_std": y_std,
            "best_epoch": best_epoch,
            "best_val_rmse": float(best_val_rmse) if np.isfinite(best_val_rmse) else None,
            "epochs_run": len(logs),
            "random_seed": RANDOM_SEED,
        },
        model_path,
    )

    metrics = None
    if n_val > 0:
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(len_val).to(device))
            preds = preds.cpu().numpy()
        preds = preds * y_std + y_mean
        y_true = y_val
        mae = float(np.mean(np.abs(y_true - preds)))
        mse = float(np.mean((y_true - preds) ** 2))
        rmse = float(np.sqrt(mse))
        mape = _safe_mape(y_true, preds)
        ss_res = float(np.sum((y_true - preds) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan
        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2}

        pd.DataFrame([metrics]).to_csv(reports_dir / "lstm_metrics.csv", index=False)
    else:
        pd.DataFrame([]).to_csv(reports_dir / "lstm_metrics.csv", index=False)

    # Plot training loss
    try:
        import matplotlib.pyplot as plt

        fig_path = figures_dir / "lstm_training_loss.png"
        epochs_logged = [row["epoch"] for row in logs]
        train_loss_vals = [row["train_loss"] for row in logs]
        val_loss_vals = [row["val_loss"] for row in logs]
        plt.figure(figsize=(8, 4))
        plt.plot(epochs_logged, train_loss_vals, label="train")
        plt.plot(epochs_logged, val_loss_vals, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LSTM Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    except Exception:
        pass

    pred_path = None
    if n_pred > 0:
        model.eval()
        with torch.no_grad():
            pred_vals = model(
                torch.tensor(X_pred, dtype=torch.float32).to(device),
                torch.tensor(len_pred).to(device),
            )
            pred_vals = pred_vals.cpu().numpy()
            pred_vals = pred_vals * y_std + y_mean

        rows = []
        for row, pred in zip(pred_rows, pred_vals):
            actual = row.get("yield_kg_ha", np.nan)
            abs_err = np.nan
            pct_err = np.nan
            if not np.isnan(actual):
                abs_err = float(abs(actual - pred))
                pct_err = float(abs_err / actual * 100.0) if actual != 0 else np.nan
            rows.append(
                {
                    "case_id": row.get("case_id"),
                    "district": row.get("district"),
                    "season_year_start": row.get("season_year_start"),
                    "season_year_end": row.get("season_year_end"),
                    "actual_yield_kg_ha": actual if not np.isnan(actual) else np.nan,
                    "lstm_predicted_yield_kg_ha": float(pred),
                    "absolute_error": abs_err,
                    "percentage_error": pct_err,
                }
            )

        pred_path = processed_dir / "lstm_predictions_2025.csv"
        pd.DataFrame(rows).to_csv(pred_path, index=False)

    _write_summary(
        reports_dir / "lstm_summary.md",
        stage_path,
        n_train,
        n_val,
        n_pred,
        max_len,
        n_features,
        candidate_features,
        dropped_features,
        feature_cols,
        metrics,
        pred_path,
        None,
    )

    logger.info("Wrote %s", reports_dir / "lstm_metrics.csv")
    logger.info("Wrote %s", reports_dir / "lstm_training_log.csv")
    if pred_path:
        logger.info("Wrote %s", pred_path)


if __name__ == "__main__":
    main()
