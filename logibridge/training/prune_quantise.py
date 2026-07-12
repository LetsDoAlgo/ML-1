"""Structured pruning + PTQ INT8 conversion pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.preprocessing import load_training_stats, normalize_features


def _log(message: str) -> None:
    print(f"[PRUNE] {time.strftime('%H:%M:%S')} | {message}")


def _build_baseline_model(input_dim: int = 6) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )


def _load_baseline_model(model_path: str, input_dim: int = 6) -> tf.keras.Model | None:
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as exc:
        # Keras/tf_keras deserialization can break across versions.
        _log(f"Direct load failed ({exc}); will fallback to local baseline pretraining")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune and quantize LogiBridge model")
    parser.add_argument("--data-path", default="training/X.npy")
    parser.add_argument("--labels-path", default="training/y.npy")
    parser.add_argument("--baseline-model", default="training/models/model_fp32.keras")
    parser.add_argument("--out-path", default="training/models/model_pruned_int8.tflite")
    return parser.parse_args()


def main() -> None:
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow_model_optimization is required. Install it with pip install tensorflow-model-optimization"
        ) from exc

    args = parse_args()
    _log("Starting structured pruning + PTQ flow")

    x = np.load(args.data_path)
    y = np.load(args.labels_path)
    _log(f"Loaded data: X={x.shape}, y={y.shape}")

    stats = load_training_stats("data_pipeline/training_stats.npy")
    x_n = normalize_features(x, stats)

    model = _load_baseline_model(args.baseline_model, input_dim=x_n.shape[1])
    if model is None:
        model = _build_baseline_model(input_dim=x_n.shape[1])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_n, y, epochs=12, batch_size=16, verbose=0)
        _log("Fallback baseline pretraining completed")
    else:
        _log(f"Loaded baseline model from {args.baseline_model}")

    batch_size = 16
    epochs = 8
    end_step = int(np.ceil(x_n.shape[0] / batch_size) * epochs)

    prune_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.35,
        begin_step=0,
        end_step=end_step,
    )

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model, pruning_schedule=prune_schedule
    )
    pruned_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    pruned_model.fit(x_n, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
    _log("Pruning fine-tune completed")

    stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    def rep_dataset():
        for i in range(min(200, x_n.shape[0])):
            yield [x_n[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    _log("INT8 conversion after pruning completed")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    _log(f"Saved pruned INT8 model to {out_path} ({len(tflite_model) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
