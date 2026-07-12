"""Train baseline MLP model and export artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.preprocessing import fit_training_stats, normalize_features, save_training_stats


def _log(message: str) -> None:
    print(f"[TRAIN] {time.strftime('%H:%M:%S')} | {message}")


def build_model(input_dim: int = 6) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LogiBridge baseline model")
    parser.add_argument("--data-dir", default="training")
    parser.add_argument("--out-dir", default="training/models")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log(f"Starting training with seed={args.seed}")
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    _log(f"Loaded dataset: X={x.shape}, y={y.shape}")

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    _log(f"Split data: train={x_train.shape[0]} val={x_val.shape[0]}")

    stats = fit_training_stats(x_train)
    save_training_stats("data_pipeline/training_stats.npy", stats)
    _log("Saved normalization stats to data_pipeline/training_stats.npy")

    x_train_n = normalize_features(x_train, stats)
    x_val_n = normalize_features(x_val, stats)

    model = build_model(input_dim=x_train_n.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train_n,
        y_train,
        validation_data=(x_val_n, y_val),
        epochs=60,
        batch_size=16,
        verbose=0,
    )
    _log("Training complete; evaluating model")

    probs = model.predict(x_val_n, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_val, preds)

    _log(f"Validation accuracy: {acc * 100:.2f}%")
    print("Confusion matrix:")
    print(confusion_matrix(y_val, preds))
    print("Classification report:")
    print(classification_report(y_val, preds, digits=3))

    if acc < 0.88:
        raise RuntimeError("Validation accuracy is below 88%. Improve pipeline before proceeding.")

    model.save(out_dir / "model_fp32.keras")
    _log(f"Saved Keras model to {out_dir / 'model_fp32.keras'}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32_tflite = converter.convert()
    (out_dir / "model_fp32.tflite").write_bytes(fp32_tflite)
    _log(f"Saved FP32 TFLite to {out_dir / 'model_fp32.tflite'}")

    np.save(out_dir / "val_X.npy", x_val)
    np.save(out_dir / "val_y.npy", y_val)

    _log(f"Saved validation split to {out_dir}")


if __name__ == "__main__":
    main()
