"""Benchmark TFLite model variants on latency, accuracy, size, and energy proxy."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, recall_score

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    try:
        import tensorflow as tf

        TFLiteInterpreter = tf.lite.Interpreter
    except ImportError as exc:
        raise RuntimeError(
            "No TFLite interpreter backend found. Install tensorflow or tflite-runtime."
        ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.preprocessing import load_training_stats, normalize_features


LAPTOP_TDP_WATTS = 15.0


def _log(message: str) -> None:
    print(f"[BENCH] {time.strftime('%H:%M:%S')} | {message}")


def run_inference(interpreter, x: np.ndarray) -> np.ndarray:
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]

    x_in = x.astype(np.float32)
    if in_d["dtype"] == np.int8:
        scale, zero = in_d["quantization"]
        x_in = np.round(x_in / scale + zero).astype(np.int8)

    interpreter.set_tensor(in_d["index"], x_in)
    interpreter.invoke()
    out = interpreter.get_tensor(out_d["index"])

    if out_d["dtype"] == np.int8:
        scale, zero = out_d["quantization"]
        out = (out.astype(np.float32) - zero) * scale

    return out


def benchmark_model(model_path: Path, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    _log(f"Benchmarking {model_path.name}")
    interpreter = TFLiteInterpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    latencies = []
    preds = []

    # Warm-up
    for i in range(10):
        _ = run_inference(interpreter, x[i : i + 1])

    cpu_before = psutil.cpu_percent(interval=None)

    for i in range(200):
        sample = x[i % len(x) : (i % len(x)) + 1]
        start = time.perf_counter()
        out = run_inference(interpreter, sample)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        preds.append(int(np.argmax(out, axis=1)[0]))

        if i > 0 and i % 50 == 0:
            _log(f"{model_path.name} runs_completed={i}")

    cpu_after = psutil.cpu_percent(interval=None)
    cpu_avg = max((cpu_before + cpu_after) / 2.0, 1.0)

    mean_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    size_kb = model_path.stat().st_size / 1024.0

    # Build ground truth for cycled samples
    y_cycled = [y[i % len(y)] for i in range(200)]
    acc = accuracy_score(y_cycled, np.asarray(preds)) * 100.0
    rec2 = recall_score(y_cycled, np.asarray(preds), labels=[2], average=None)[0] * 100.0

    # Energy proxy: E = P * t, where P estimated from CPU utilization and TDP.
    power_w = (cpu_avg / 100.0) * LAPTOP_TDP_WATTS
    energy_mj = power_w * (mean_ms / 1000.0) * 1000.0

    return {
        "mean_latency_ms": mean_ms,
        "p95_latency_ms": p95_ms,
        "model_size_kb": size_kb,
        "accuracy_percent": acc,
        "energy_mj": energy_mj,
        "class2_recall_percent": rec2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LogiBridge models")
    parser.add_argument("--models-dir", default="training/models")
    parser.add_argument("--out-csv", default="optimisation/results/benchmark_results.csv")
    parser.add_argument("--val-x", default="training/models/val_X.npy")
    parser.add_argument("--val-y", default="training/models/val_y.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log("Starting benchmark process")

    stats = load_training_stats("data_pipeline/training_stats.npy")
    x = np.load(args.val_x)
    y = np.load(args.val_y)
    x_n = normalize_features(x, stats)
    _log(f"Loaded validation arrays: X={x.shape}, y={y.shape}")

    models = {
        "M1_FP32": Path(args.models_dir) / "model_fp32.tflite",
        "M2_PTQ_INT8": Path(args.models_dir) / "model_int8.tflite",
        "M3_PRUNED_PTQ_INT8": Path(args.models_dir) / "model_pruned_int8.tflite",
    }

    rows: List[Dict[str, float]] = []
    for name, path in models.items():
        if not path.exists():
            _log(f"Skipping {name}; missing {path}")
            continue
        metrics = benchmark_model(path, x_n, y)
        metrics["variant"] = name
        rows.append(metrics)
        _log(f"{name} metrics={metrics}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "variant",
                "mean_latency_ms",
                "p95_latency_ms",
                "model_size_kb",
                "accuracy_percent",
                "energy_mj",
                "class2_recall_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    _log(f"Saved benchmark results to {out_csv}")


if __name__ == "__main__":
    main()
