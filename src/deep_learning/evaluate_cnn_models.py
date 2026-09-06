from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.deep_learning.cnn_training_utils import configure_tensorflow

logger = get_logger(__name__)

MODEL_NAMES = [
    "densenet121",
    "mobilenetv2",
    "inceptionv3",
    "vgg16",
    "xception",
]


def _configure_gpu() -> None:
    configure_tensorflow(logger)


def _split_path(config: dict) -> Path:
    interim_dir = Path(config["paths"]["interim_dir"])
    return interim_dir / "cnn_dataset_split.csv"


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
    preprocess_fn,
) -> tf.data.Dataset:
    paths = df["image_path"].astype(str).tolist()
    labels = df["stage"].map(class_to_index).astype(int).tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, label):
        img = _load_image(path, img_size)
        img = preprocess_fn(img)
        return img, label

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _preprocess_for_model(model_name: str):
    if model_name == "densenet121":
        from tensorflow.keras.applications.densenet import preprocess_input

        return preprocess_input
    if model_name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        return preprocess_input
    if model_name == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input

        return preprocess_input
    if model_name == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input

        return preprocess_input
    if model_name == "xception":
        from tensorflow.keras.applications.xception import preprocess_input

        return preprocess_input
    raise ValueError(f"Unknown model: {model_name}")


def _load_meta(reports_dir: Path, model_name: str) -> dict:
    meta_path = reports_dir / f"{model_name}_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> None:
    _configure_gpu()

    config = load_config()
    split_path = _split_path(config)
    if not split_path.exists():
        raise FileNotFoundError("Missing cnn_dataset_split.csv. Run a training script first.")

    split_df = pd.read_csv(split_path)
    test_df = split_df[split_df["split"] == "test"]
    if test_df.empty:
        raise ValueError("Test split is empty. Check cnn_dataset_split.csv.")

    models_dir = ROOT / "models" / "cnn"
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"

    rows = []
    for model_name in MODEL_NAMES:
        model_path = models_dir / f"{model_name}_best.keras"
        if not model_path.exists():
            logger.warning("Missing model: %s", model_path)
            continue

        meta = _load_meta(reports_dir, model_name)
        img_size = int(meta.get("img_size", 224))
        batch_size = int(meta.get("batch_size", 16))
        class_names = meta.get("class_names")
        if not class_names:
            class_names = sorted(split_df["stage"].dropna().unique().tolist())

        class_to_index = {name: idx for idx, name in enumerate(class_names)}
        preprocess_fn = _preprocess_for_model(model_name)
        test_ds = _make_dataset(test_df, class_to_index, img_size, batch_size, preprocess_fn)

        y_true = test_df["stage"].map(class_to_index).astype(int).to_numpy()

        model = tf.keras.models.load_model(model_path)
        logits = model.predict(test_ds, verbose=1)
        y_pred = np.argmax(logits, axis=1)

        min_len = min(len(y_true), len(y_pred))
        if len(y_true) != len(y_pred):
            logger.warning(
                "%s prediction/label count mismatch: len(y_true)=%d len(y_pred)=%d. "
                "Truncating both arrays to %d before metrics.",
                model_name,
                len(y_true),
                len(y_pred),
                min_len,
            )
        if min_len == 0:
            logger.warning("No aligned predictions for %s. Skipping metrics.", model_name)
            continue
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        rows.append(
            {
                "model": model_name,
                "accuracy": acc,
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }
        )

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_path = reports_dir / f"{model_name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        logger.info("Wrote %s", cm_path)

    if not rows:
        logger.warning("No evaluation results produced.")
        return

    out_path = reports_dir / "cnn_evaluation_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
