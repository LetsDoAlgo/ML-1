"""Generate labeled dataset for LogiBridge model training.

Creates feature windows for classes:
- 0 Normal (none)
- 1 Warning (temp_drift)
- 2 Critical (combined)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.preprocessing import FeatureExtractor
from data_pipeline.simulator import ColdChainSimulator


CLASS_TO_ANOMALY = {0: "none", 1: "temp_drift", 2: "combined"}
CLASS_TO_DURATION_MIN = {0: 20, 1: 15, 2: 15}


def _log(message: str) -> None:
    print(f"[DATASET] {time.strftime('%H:%M:%S')} | {message}", flush=True)


def generate_class_features(label: int, seed: int = 42) -> np.ndarray:
    anomaly = CLASS_TO_ANOMALY[label]
    duration_min = CLASS_TO_DURATION_MIN[label]

    sim = ColdChainSimulator(anomaly=anomaly, seed=seed + label)
    fe = FeatureExtractor(window_seconds=30, step_seconds=10)

    total_seconds = duration_min * 60
    features = []
    _log(f"Generating class={label} anomaly={anomaly} duration_min={duration_min}")
    for tick in range(total_seconds):
        now = float(tick)
        samples = sim.generate_tick(tick=tick, now=now)

        fe.add_temperature(now, samples["temperature"].temperature_c)
        if tick % 2 == 0:
            fe.add_vibration(now, samples["vibration_rms"].vibration_rms_g)

        feat = fe.maybe_extract(now)
        if feat is not None:
            features.append(feat)

        if tick > 0 and tick % 60 == 0:
            _log(f"class={label} tick={tick}/{total_seconds} windows={len(features)}")

    if not features:
        raise RuntimeError(f"No features generated for class {label}")
    _log(f"Completed class={label}; windows={len(features)}")
    return np.asarray(features, dtype=np.float32)


def build_dataset() -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for label in (0, 1, 2):
        feats = generate_class_features(label)
        xs.append(feats)
        ys.append(np.full((feats.shape[0],), label, dtype=np.int32))
    _log("All classes generated; stacking arrays")
    x = np.vstack(xs)
    y = np.concatenate(ys)
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LogiBridge training dataset")
    parser.add_argument("--out-dir", default="training")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        _log("Dataset generation started. Press Ctrl+C to stop.")
        x, y = build_dataset()
        np.save(out_dir / "X.npy", x)
        np.save(out_dir / "y.npy", y)

        _log(f"Saved dataset: X={x.shape}, y={y.shape} at {out_dir}")
    except KeyboardInterrupt:
        _log("Interrupted by user (Ctrl+C). Exiting safely.")
