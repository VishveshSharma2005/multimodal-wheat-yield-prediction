from __future__ import annotations

from pathlib import Path
import json
import math

import pandas as pd
import tensorflow as tf

from src.utils.paths import project_root


def configure_tensorflow(logger) -> bool:
    gpus = tf.config.list_physical_devices("GPU")
    gpu_detected = bool(gpus)
    print(f"GPU detected: {gpu_detected}")

    if gpu_detected:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                logger.warning("Could not set GPU memory growth: %s", exc)
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Enabled mixed precision policy: mixed_float16")
        except Exception as exc:
            logger.warning("Could not enable mixed precision: %s", exc)
        logger.info("Detected %d GPU(s)", len(gpus))
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        logger.info("No GPU detected. Running on CPU.")

    return gpu_detected


def validate_image_file(image_path: str | Path) -> tuple[bool, str]:
    path = Path(image_path)
    try:
        data = tf.io.read_file(str(path))
        img = tf.image.decode_image(data, channels=3, expand_animations=False)
        _ = img.shape
        return True, ""
    except Exception as exc:
        return False, str(exc)


def portable_image_path(image_path: str | Path) -> Path:
    """Map saved image paths to the current project copy.

    Older split files may contain Windows absolute paths from the machine that
    created the split. Colab needs those paths rewritten under /content/drive.
    """
    root = project_root()
    raw = str(image_path).strip()
    path = Path(raw)
    if path.exists():
        return path

    normalized = raw.replace("\\", "/")
    anchors = (
        "data/raw/images/",
        "data/interim/",
        "data/processed/",
        "data/raw/",
        "reports/",
        "models/",
    )
    for anchor in anchors:
        if anchor in normalized:
            candidate = root / normalized.split(anchor, 1)[1]
            candidate = root / anchor.rstrip("/") / normalized.split(anchor, 1)[1]
            if candidate.exists():
                return candidate
            return candidate

    if not path.is_absolute():
        return root / path
    return path


def normalize_image_paths(df: pd.DataFrame, path_col: str = "image_path") -> pd.DataFrame:
    if path_col not in df.columns:
        return df
    df = df.copy()
    df[path_col] = df[path_col].map(lambda value: str(portable_image_path(value)))
    return df


def filter_valid_images(
    df: pd.DataFrame,
    reports_dir: Path,
    logger,
    path_col: str = "image_path",
) -> pd.DataFrame:
    if path_col not in df.columns:
        return df

    df = normalize_image_paths(df, path_col=path_col)
    valid_rows = []
    corrupted_rows = []
    for _, row in df.iterrows():
        image_path = row[path_col]
        is_valid, error_message = validate_image_file(image_path)
        if is_valid:
            valid_rows.append(row)
        else:
            logger.warning("Skipping unreadable image %s: %s", image_path, error_message)
            corrupted_rows.append(
                {
                    "image_path": image_path,
                    "error_message": error_message,
                }
            )

    reports_dir.mkdir(parents=True, exist_ok=True)
    corrupted_path = reports_dir / "corrupted_images.csv"
    pd.DataFrame(corrupted_rows, columns=["image_path", "error_message"]).to_csv(
        corrupted_path,
        index=False,
    )
    logger.info("Wrote %s", corrupted_path)

    if corrupted_rows:
        logger.warning("Excluded %d corrupted image(s) from CNN dataset.", len(corrupted_rows))
    return pd.DataFrame(valid_rows).reset_index(drop=True)


def log_dataset_diagnostics(split_df: pd.DataFrame, logger) -> dict:
    split_df = split_df.copy()
    total_images = int(len(split_df))
    stage_counts = split_df["stage"].value_counts().sort_index()
    split_counts = split_df["split"].value_counts()

    train_count = int(split_counts.get("train", 0))
    validation_count = int(split_counts.get("validation", 0))
    test_count = int(split_counts.get("test", 0))

    print(f"Total images: {total_images}")
    print("Stage distribution:")
    for stage, count in stage_counts.items():
        print(f"{stage}: {int(count)}")
    print(f"Train count: {train_count}")
    print(f"Validation count: {validation_count}")
    print(f"Test count: {test_count}")

    logger.info("Total images: %d", total_images)
    for stage, count in stage_counts.items():
        logger.info("Stage distribution - %s: %d", stage, int(count))
    logger.info("Train count: %d", train_count)
    logger.info("Validation count: %d", validation_count)
    logger.info("Test count: %d", test_count)

    return {
        "total_images": total_images,
        "stage_distribution": {str(stage): int(count) for stage, count in stage_counts.items()},
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
    }


def compute_steps(count: int, batch_size: int, split_name: str) -> int:
    if count <= 0:
        raise ValueError(f"{split_name} split is empty.")
    return max(1, math.ceil(count / batch_size))


def best_history_value(history: tf.keras.callbacks.History, key: str, mode: str) -> float | None:
    values = history.history.get(key, [])
    if not values:
        return None
    cleaned = [float(value) for value in values if pd.notna(value)]
    if not cleaned:
        return None
    if mode == "max":
        return max(cleaned)
    return min(cleaned)


def write_training_summary(
    reports_dir: Path,
    model_name: str,
    model_path: Path,
    training_time: float,
    best_val_accuracy: float | None,
    best_val_loss: float | None,
) -> Path:
    summary_path = reports_dir / f"{model_name}_training_summary.json"
    payload = {
        "model_name": model_name,
        "model_path": str(model_path),
        "training_time_seconds": float(training_time),
        "best_validation_accuracy": best_val_accuracy,
        "best_validation_loss": best_val_loss,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
