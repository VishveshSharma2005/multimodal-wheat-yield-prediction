from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
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
    class_names = sorted(split_df["stage"].dropna().unique().tolist())
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    models_dir = ROOT / "models" / "cnn"
    reports_dir = Path(config["paths"]["reports_dir"]) / "model_results"
    processed_dir = Path(config["paths"]["processed_dir"])
    feature_dir = processed_dir / "cnn_features"
    ensure_dir(feature_dir)

    combined_vectors = {}

    for model_name in MODEL_NAMES:
        model_path = models_dir / f"{model_name}_best.keras"
        if not model_path.exists():
            logger.warning("Missing model: %s", model_path)
            continue

        meta = _load_meta(reports_dir, model_name)
        img_size = int(meta.get("img_size", 224))
        batch_size = int(meta.get("batch_size", 16))

        preprocess_fn = _preprocess_for_model(model_name)
        ds = _make_dataset(split_df, class_to_index, img_size, batch_size, preprocess_fn)

        model = tf.keras.models.load_model(model_path)
        if "cnn_gap" in [layer.name for layer in model.layers]:
            feature_output = model.get_layer("cnn_gap").output
        else:
            feature_output = model.layers[-2].output

        feature_model = tf.keras.Model(inputs=model.input, outputs=feature_output)
        logger.info(
            "Extracting %s features from %d image records with batch size %d",
            model_name,
            len(split_df),
            batch_size,
        )
        features = feature_model.predict(ds, verbose=1)

        image_records = split_df.reset_index(drop=True)
        feature_count = len(features)
        record_count = len(image_records)
        logger.info(
            "%s extraction produced %d feature vectors for %d image records",
            model_name,
            feature_count,
            record_count,
        )
        if feature_count != record_count:
            logger.warning(
                "%s feature/image count mismatch: len(features)=%d len(image_records)=%d. "
                "Truncating to the minimum length.",
                model_name,
                feature_count,
                record_count,
            )
        total = min(feature_count, record_count)
        if total == 0:
            logger.warning("No aligned feature records for %s. Skipping output.", model_name)
            continue

        records = []
        for idx in range(total):
            row = image_records.iloc[idx]
            vector = features[idx].astype(float).tolist()
            records.append(
                {
                    "image_path": row["image_path"],
                    "stage": row["stage"],
                    "feature_vector": json.dumps(vector),
                }
            )

        out_path = feature_dir / f"{model_name}_features.csv"
        pd.DataFrame(records).to_csv(out_path, index=False)
        logger.info("Wrote %s", out_path)

        combined_vectors[model_name] = {
            rec["image_path"]: json.loads(rec["feature_vector"]) for rec in records
        }

    if not combined_vectors:
        logger.warning("No feature files produced.")
        return

    ordered_models = [name for name in MODEL_NAMES if name in combined_vectors]
    combined_records = []
    for _, row in split_df.iterrows():
        image_path = row["image_path"]
        stage = row["stage"]
        vectors = []
        missing = False
        for model_name in ordered_models:
            vec = combined_vectors[model_name].get(image_path)
            if vec is None:
                missing = True
                break
            vectors.extend(vec)
        if missing:
            continue
        combined_records.append(
            {
                "image_path": image_path,
                "stage": stage,
                "feature_vector": json.dumps(vectors),
            }
        )

    combined_path = processed_dir / "cnn_features.csv"
    pd.DataFrame(combined_records).to_csv(combined_path, index=False)
    logger.info("Wrote %s", combined_path)


if __name__ == "__main__":
    main()
