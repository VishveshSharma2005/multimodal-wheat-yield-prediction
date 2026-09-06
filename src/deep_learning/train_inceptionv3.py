from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir, to_project_relative
from src.utils.logging_utils import get_logger
from src.deep_learning.cnn_training_utils import (
    best_history_value,
    compute_steps,
    configure_tensorflow,
    filter_valid_images,
    log_dataset_diagnostics,
    normalize_image_paths,
    write_training_summary,
)

logger = get_logger(__name__)

MODEL_NAME = "inceptionv3"
DEFAULT_IMG_SIZE = 299


def _configure_gpu() -> None:
    configure_tensorflow(logger)


def _dataset_dir(config: dict) -> Path:
    images_dir = Path(config["paths"]["images_dir"])
    return images_dir / "wheat_stage_dataset"


def _split_path(config: dict) -> Path:
    interim_dir = Path(config["paths"]["interim_dir"])
    ensure_dir(interim_dir)
    return interim_dir / "cnn_dataset_split.csv"


def _scan_images(dataset_dir: Path) -> pd.DataFrame:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    records: list[dict] = []
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

    for stage_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        for pattern in extensions:
            for img_path in stage_dir.rglob(pattern):
                records.append(
                    {
                        "image_path": to_project_relative(img_path),
                        "stage": stage_dir.name,
                    }
                )

    if not records:
        raise ValueError("No images found in dataset directory.")

    return pd.DataFrame(records)


def _create_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["stage"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["stage"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def _load_or_create_split(config: dict, refresh: bool = False) -> pd.DataFrame:
    split_path = _split_path(config)
    if split_path.exists() and not refresh:
        df = pd.read_csv(split_path)
        return normalize_image_paths(df)

    dataset_dir = _dataset_dir(config)
    df = _scan_images(dataset_dir)
    df = _create_split(df, seed=42)
    df.to_csv(split_path, index=False)
    logger.info("Wrote %s", split_path)
    return normalize_image_paths(df)


def _build_label_mapping(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    class_names = sorted(df["stage"].dropna().unique().tolist())
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    return class_names, class_to_index


def _load_image(path: tf.Tensor, img_size: int) -> tf.Tensor:
    data = tf.io.read_file(path)
    img = tf.image.decode_image(data, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32)


def _make_dataset(
    df: pd.DataFrame,
    class_to_index: dict[str, int],
    img_size: int,
    batch_size: int,
    training: bool,
    preprocess_fn,
) -> tf.data.Dataset:
    paths = df["image_path"].astype(str).tolist()
    labels = df["stage"].map(class_to_index).astype(int).tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)

    augmenter = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )

    def _map_fn(path, label):
        img = _load_image(path, img_size)
        if training:
            img = augmenter(img, training=True)
        img = preprocess_fn(img)
        return img, label

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _build_model(
    num_classes: int,
    img_size: int,
    fine_tune: bool,
    fine_tune_at: int | None,
    learning_rate: float,
) -> tf.keras.Model:
    from tensorflow.keras.applications import InceptionV3

    base_model = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="cnn_gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="cnn_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)
    model = tf.keras.Model(inputs, outputs, name=MODEL_NAME)

    if fine_tune:
        base_model.trainable = True
        if fine_tune_at is not None:
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def _write_history(history: tf.keras.callbacks.History, out_path: Path) -> None:
    hist = history.history
    df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(hist.get("loss", [])) + 1),
            "loss": hist.get("loss", []),
            "accuracy": hist.get("accuracy", []),
            "val_loss": hist.get("val_loss", []),
            "val_accuracy": hist.get("val_accuracy", []),
        }
    )
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InceptionV3 for wheat stages.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--refresh-split", action="store_true")
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--fine-tune-at", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    _configure_gpu()

    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    split_df = filter_valid_images(_load_or_create_split(config, refresh=args.refresh_split), reports_dir, logger)
    dataset_diagnostics = log_dataset_diagnostics(split_df, logger)
    class_names, class_to_index = _build_label_mapping(split_df)

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "validation"]

    from tensorflow.keras.applications.inception_v3 import preprocess_input

    train_ds = _make_dataset(
        train_df,
        class_to_index,
        args.img_size,
        args.batch_size,
        training=True,
        preprocess_fn=preprocess_input,
    )
    val_ds = _make_dataset(
        val_df,
        class_to_index,
        args.img_size,
        args.batch_size,
        training=False,
        preprocess_fn=preprocess_input,
    )

    model = _build_model(
        num_classes=len(class_names),
        img_size=args.img_size,
        fine_tune=args.fine_tune,
        fine_tune_at=args.fine_tune_at,
        learning_rate=args.learning_rate,
    )

    models_dir = ROOT / "models" / "cnn"
    ensure_dir(models_dir)
    ensure_dir(reports_dir)

    model_path = models_dir / f"{MODEL_NAME}_best.keras"
    history_path = reports_dir / f"{MODEL_NAME}_history.csv"
    meta_path = reports_dir / f"{MODEL_NAME}_meta.json"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    start = time.time()
    steps_per_epoch = compute_steps(len(train_df), args.batch_size, "train")
    validation_steps = compute_steps(len(val_df), args.batch_size, "validation")
    train_ds = train_ds.repeat()
    val_ds = val_ds.repeat()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.time() - start

    _write_history(history, history_path)
    best_val_accuracy = best_history_value(history, "val_accuracy", "max")
    best_val_loss = best_history_value(history, "val_loss", "min")

    meta = {
        "model_name": MODEL_NAME,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "training_time_seconds": training_time,
        "best_validation_accuracy": best_val_accuracy,
        "best_validation_loss": best_val_loss,
        "class_names": class_names,
        "dataset_diagnostics": dataset_diagnostics,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary_path = write_training_summary(
        reports_dir,
        MODEL_NAME,
        model_path,
        training_time,
        best_val_accuracy,
        best_val_loss,
    )

    logger.info("Saved model: %s", model_path)
    logger.info("Saved history: %s", history_path)
    logger.info("Saved meta: %s", meta_path)
    logger.info("Saved training summary: %s", summary_path)
    logger.info("Training time: %.2f seconds", training_time)
    logger.info("Best validation accuracy: %s", best_val_accuracy)
    logger.info("Best validation loss: %s", best_val_loss)


if __name__ == "__main__":
    main()
