"""MQTT inference service for LogiBridge edge node."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
import sys
from typing import Deque, Dict

import numpy as np
import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.preprocessing import FeatureExtractor, load_training_stats, normalize_features


def _log(message: str) -> None:
    print(f"[INFER] {time.strftime('%H:%M:%S')} | {message}")


class TFLiteModel:
    def __init__(self, model_path: str) -> None:
        interpreter_cls = None
        try:
            from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter

            interpreter_cls = TFLiteInterpreter
        except ImportError:
            try:
                import tensorflow as tf

                interpreter_cls = tf.lite.Interpreter
            except ImportError as exc:
                raise RuntimeError(
                    "No TFLite interpreter backend found. Install tensorflow or tflite-runtime."
                ) from exc

        _log(f"Loading TFLite model: {model_path}")
        self.interpreter = interpreter_cls(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.in_details = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)

        # Handle INT8 models by quantizing the normalized input.
        if self.in_details["dtype"] == np.int8:
            scale, zero = self.in_details["quantization"]
            x = np.round(x / scale + zero).astype(np.int8)

        self.interpreter.set_tensor(self.in_details["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_details["index"])

        if self.out_details["dtype"] == np.int8:
            scale, zero = self.out_details["quantization"]
            out = (out.astype(np.float32) - zero) * scale

        # Ensure probabilities-like output.
        exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
        probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
        return probs


def _topic(truck_id: str, suffix: str) -> str:
    return f"logibridge/trucks/{truck_id}/{suffix}"


def run() -> None:
    truck_id = os.getenv("TRUCK_ID", "TRUCK_001")
    broker_host = os.getenv("MQTT_BROKER_HOST", "localhost")
    broker_port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    model_path = os.getenv("MODEL_PATH", str(ROOT / "training" / "models" / "model_int8.tflite"))

    stats_path = os.getenv("TRAINING_STATS", str(ROOT / "data_pipeline" / "training_stats.npy"))
    alert_log_path = Path(os.getenv("ALERT_LOG", str(ROOT / "inference" / "local_alert_log.jsonl")))

    try:
        model = TFLiteModel(model_path)
        stats = load_training_stats(stats_path)
    except Exception as exc:
        _log(f"Startup failed: {exc}")
        raise SystemExit(2) from exc

    extractor = FeatureExtractor(window_seconds=30, step_seconds=10)
    _log(f"Loaded training stats from {stats_path}")

    door_events: Deque[str] = deque(maxlen=32)

    client = mqtt.Client(client_id=f"logibridge-inference-{truck_id}")

    def on_connect(client: mqtt.Client, _userdata, _flags, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"MQTT connect failed with code {rc}")
        client.subscribe(_topic(truck_id, "sensors/temperature"), qos=1)
        client.subscribe(_topic(truck_id, "sensors/vibration_rms"), qos=1)
        client.subscribe(_topic(truck_id, "sensors/door_event"), qos=1)
        _log("Connected to MQTT and subscribed to sensor topics")

    def on_message(client: mqtt.Client, _userdata, msg: mqtt.MQTTMessage) -> None:
        payload = json.loads(msg.payload.decode("utf-8"))
        ts = float(payload["timestamp"])
        value = payload["value"]

        if msg.topic.endswith("temperature"):
            extractor.add_temperature(ts, float(value))
        elif msg.topic.endswith("vibration_rms"):
            extractor.add_vibration(ts, float(value))
        elif msg.topic.endswith("door_event"):
            door_events.append(str(value))

        feat = extractor.maybe_extract(ts)
        if feat is None:
            return

        x = normalize_features(feat.reshape(1, -1), stats)
        probs = model.predict(x)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))

        result = {
            "timestamp": ts,
            "truck_id": truck_id,
            "prediction": pred,
            "confidence": conf,
            "probabilities": probs.tolist(),
            "recent_door_events": list(door_events),
        }

        client.publish(_topic(truck_id, "inference"), json.dumps(result), qos=1)
        _log(f"inference class={pred} confidence={conf:.3f}")

        if pred == 2:
            alert = {"type": "CRITICAL", "timestamp": ts, "payload": result}
            client.publish(_topic(truck_id, "alerts"), json.dumps(alert), qos=1)
            with alert_log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(alert) + "\n")
            _log(f"CRITICAL alert stored at {alert_log_path}")

    client.on_connect = on_connect
    client.on_message = on_message
    _log(f"Connecting MQTT broker at {broker_host}:{broker_port}")
    try:
        client.connect(broker_host, broker_port, keepalive=60)
    except OSError as exc:
        _log(f"MQTT broker unavailable at {broker_host}:{broker_port} ({exc})")
        raise SystemExit(3) from exc

    client.loop_start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    run()
