"""Preprocessing pipeline for LogiBridge.

Sequence:
1) 5-sample moving average filter
2) 30-second sliding window features, 10-second step
3) Normalization using fixed training stats
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TrainingStats:
    mean: np.ndarray
    std: np.ndarray


class FeatureExtractor:
    def __init__(self, window_seconds: int = 30, step_seconds: int = 10) -> None:
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds

        self.temp_ma: Deque[float] = collections.deque(maxlen=5)
        self.vib_ma: Deque[float] = collections.deque(maxlen=5)

        self.temp_series: List[Tuple[float, float]] = []
        self.vib_series: List[Tuple[float, float]] = []

        self._last_feature_time: Optional[float] = None

    def add_temperature(self, ts: float, value: float) -> None:
        self.temp_ma.append(float(value))
        filt = float(np.mean(self.temp_ma))
        self.temp_series.append((ts, filt))
        self._trim_old(ts)

    def add_vibration(self, ts: float, value: float) -> None:
        self.vib_ma.append(float(value))
        filt = float(np.mean(self.vib_ma))
        self.vib_series.append((ts, filt))
        self._trim_old(ts)

    def _trim_old(self, now_ts: float) -> None:
        cutoff = now_ts - self.window_seconds - 5
        self.temp_series = [(t, v) for (t, v) in self.temp_series if t >= cutoff]
        self.vib_series = [(t, v) for (t, v) in self.vib_series if t >= cutoff]

    def maybe_extract(self, now_ts: float) -> Optional[np.ndarray]:
        if self._last_feature_time is not None and (now_ts - self._last_feature_time) < self.step_seconds:
            return None

        t_start = now_ts - self.window_seconds
        temp_vals = [v for (t, v) in self.temp_series if t >= t_start]
        vib_vals = [v for (t, v) in self.vib_series if t >= t_start]

        if len(temp_vals) < 5 or len(vib_vals) < 3:
            return None

        temp_arr = np.asarray(temp_vals, dtype=np.float32)
        vib_arr = np.asarray(vib_vals, dtype=np.float32)

        temp_mean = float(np.mean(temp_arr))
        temp_std = float(np.std(temp_arr))

        # Temperature rate of change in C/min over the window.
        t0 = self.temp_series[-len(temp_vals)][0]
        t1 = self.temp_series[-1][0]
        dt_min = max((t1 - t0) / 60.0, 1e-6)
        temp_roc = float((temp_arr[-1] - temp_arr[0]) / dt_min)

        vib_rms = float(np.sqrt(np.mean(np.square(vib_arr))))
        vib_peak = float(np.max(vib_arr))

        vib_std = float(np.std(vib_arr))
        if vib_std < 1e-9:
            vib_kurt = 0.0
        else:
            centered = (vib_arr - np.mean(vib_arr)) / vib_std
            vib_kurt = float(np.mean(centered**4))

        self._last_feature_time = now_ts
        return np.asarray([temp_mean, temp_std, temp_roc, vib_rms, vib_peak, vib_kurt], dtype=np.float32)


def fit_training_stats(features: np.ndarray) -> TrainingStats:
    if features.ndim != 2 or features.shape[1] != 6:
        raise ValueError("features must have shape [N, 6]")
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std < 1e-9] = 1.0
    return TrainingStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def normalize_features(features: np.ndarray, stats: TrainingStats) -> np.ndarray:
    return (features - stats.mean) / stats.std


def save_training_stats(path: str, stats: TrainingStats) -> None:
    data = np.stack([stats.mean, stats.std], axis=0)
    np.save(path, data)


def load_training_stats(path: str) -> TrainingStats:
    data = np.load(path)
    if data.shape != (2, 6):
        raise ValueError("training stats file has invalid shape")
    return TrainingStats(mean=data[0].astype(np.float32), std=data[1].astype(np.float32))


def shifted_stats(stats: TrainingStats, sigma: float = 3.0) -> TrainingStats:
    return TrainingStats(mean=stats.mean + sigma * stats.std, std=stats.std.copy())
