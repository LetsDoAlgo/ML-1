"""Convert baseline Keras model to full INT8 TFLite using representative data."""

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
    print(f"[PTQ] {time.strftime('%H:%M:%S')} | {message}")


def representative_dataset_gen(x_cal: np.ndarray):
    for i in range(x_cal.shape[0]):
        yield [x_cal[i : i + 1].astype(np.float32)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Keras model to PTQ INT8")
    parser.add_argument("--model-path", default="training/models/model_fp32.keras")
    parser.add_argument("--calibration-data", default="training/X.npy")
    parser.add_argument("--out-path", default="training/models/model_int8.tflite")
    parser.add_argument("--calibration-samples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log(f"Loading baseline model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    x = np.load(args.calibration_data)
    _log(f"Loaded calibration data: {x.shape}")
    stats = load_training_stats("data_pipeline/training_stats.npy")
    x_n = normalize_features(x, stats)

    calib_count = min(args.calibration_samples, x_n.shape[0])
    x_cal = x_n[:calib_count]
    _log(f"Using calibration samples: {calib_count}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(x_cal)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    _log("INT8 conversion complete")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    _log(f"Saved INT8 model to {out_path} ({len(tflite_model) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
