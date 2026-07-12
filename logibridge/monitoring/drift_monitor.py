"""Population Stability Index monitoring for inference confidence."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import List

import paho.mqtt.client as mqtt


BINS = [0.0, 0.25, 0.50, 0.75, 1.0]


def _log(message: str) -> None:
    print(f"[DRIFT] {time.strftime('%H:%M:%S')} | {message}")


def bucketize(values: List[float]) -> List[float]:
    counts = [0, 0, 0, 0]
    for v in values:
        if v < 0.25:
            counts[0] += 1
        elif v < 0.50:
            counts[1] += 1
        elif v < 0.75:
            counts[2] += 1
        else:
            counts[3] += 1
    total = max(sum(counts), 1)
    return [c / total for c in counts]


def psi(expected: List[float], actual: List[float]) -> float:
    eps = 1e-6
    total = 0.0
    for e, a in zip(expected, actual):
        e = max(e, eps)
        a = max(a, eps)
        total += (a - e) * math.log(a / e)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LogiBridge PSI drift monitor")
    parser.add_argument("--reference", default="monitoring/reference_dist.json")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--truck-id", default="TRUCK_001")
    parser.add_argument(
        "--report-interval-sec",
        type=float,
        default=10.0,
        help="Seconds between PSI reports (default: 10).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum confidences before first PSI report (default: 5).",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=10.0,
        help="Seconds between status heartbeats (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ref_data = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    except Exception as exc:
        _log(f"Startup failed: {exc}")
        raise SystemExit(2) from exc

    reference = ref_data["distribution"]
    _log(f"Loaded reference distribution from {args.reference}: {reference}")

    confidences = deque(maxlen=100)
    msg_count = 0
    last_report = 0.0
    last_heartbeat = time.time()

    client = mqtt.Client(client_id=f"logibridge-psi-{args.truck_id}")

    def on_connect(client: mqtt.Client, _userdata, _flags, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"MQTT connection failed with code {rc}")
        topic = f"logibridge/trucks/{args.truck_id}/inference"
        client.subscribe(topic, qos=1)
        _log(f"Connected to MQTT; subscribed to {topic}")

    def on_message(_client: mqtt.Client, _userdata, msg: mqtt.MQTTMessage) -> None:
        nonlocal msg_count
        payload = json.loads(msg.payload.decode("utf-8"))
        confidences.append(float(payload["confidence"]))
        msg_count += 1
        _log(
            f"received inference #{msg_count} "
            f"class={payload.get('prediction')} confidence={payload.get('confidence'):.3f}"
        )

    client.on_connect = on_connect
    client.on_message = on_message
    _log(f"Connecting MQTT broker at {args.broker_host}:{args.broker_port}")
    try:
        client.connect(args.broker_host, args.broker_port, keepalive=60)
    except OSError as exc:
        _log(f"MQTT broker unavailable at {args.broker_host}:{args.broker_port} ({exc})")
        raise SystemExit(3) from exc

    client.loop_start()
    _log(
        f"Waiting for inference messages... "
        f"report_interval={args.report_interval_sec}s min_samples={args.min_samples}"
    )

    try:
        while True:
            now = time.time()
            if now - last_report >= args.report_interval_sec and len(confidences) >= args.min_samples:
                actual = bucketize(list(confidences))
                score = psi(reference, actual)
                _log(
                    f"Current PSI: {score:.3f} bins={actual} "
                    f"samples={len(confidences)} total_msgs={msg_count}"
                )
                if score > 0.25:
                    print(f"[LOGIBRIDGE DRIFT ALERT] PSI={score:.3f}")
                last_report = now
                last_heartbeat = now
            elif now - last_heartbeat >= args.heartbeat_sec:
                _log(
                    f"heartbeat: samples={len(confidences)}/{args.min_samples} "
                    f"total_msgs={msg_count} (waiting for enough data)"
                )
                last_heartbeat = now
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
