"""Cold-chain truck sensor simulator with MQTT publishing support.

Streams:
- temperature @ 1 Hz
- vibration_rms @ 0.5 Hz
- door_event as discrete events
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


ANOMALY_CHOICES = ("none", "temp_drift", "vibration", "combined")


def _log(message: str) -> None:
    print(f"[SIM] {time.strftime('%H:%M:%S')} | {message}")


@dataclass
class SensorSample:
    timestamp: float
    temperature_c: Optional[float] = None
    vibration_rms_g: Optional[float] = None
    door_event: Optional[str] = None


class ColdChainSimulator:
    def __init__(self, anomaly: str = "none", seed: int = 42) -> None:
        if anomaly not in ANOMALY_CHOICES:
            raise ValueError(f"Unsupported anomaly mode: {anomaly}")
        self.anomaly = anomaly
        self.rand = random.Random(seed)
        self.temp = 4.0

    def _temperature(self, tick: int) -> float:
        # Base normal distribution around setpoint 4.0 C.
        base = self.rand.gauss(4.0, 0.3)
        if self.anomaly in ("temp_drift", "combined"):
            drift = 0.08 * tick
            return base + drift
        return base

    def _vibration(self) -> float:
        if self.anomaly in ("vibration", "combined"):
            return max(0.0, self.rand.gauss(1.2, 0.15))
        return max(0.0, self.rand.gauss(0.45, 0.05))

    def _door_event(self) -> Optional[str]:
        # Generate sparse open/close events.
        event_prob = 0.015
        if self.rand.random() < event_prob:
            return self.rand.choice(["OPEN", "CLOSE"])
        return None

    def generate_tick(self, tick: int, now: Optional[float] = None) -> Dict[str, SensorSample]:
        now = time.time() if now is None else now
        temperature = SensorSample(timestamp=now, temperature_c=self._temperature(tick))
        vibration = SensorSample(timestamp=now, vibration_rms_g=self._vibration())
        door = SensorSample(timestamp=now, door_event=self._door_event())
        return {"temperature": temperature, "vibration_rms": vibration, "door_event": door}


def _publish_sample(client: mqtt.Client, topic: str, sample: SensorSample, truck_id: str) -> None:
    payload = {"truck_id": truck_id, "timestamp": sample.timestamp}
    if sample.temperature_c is not None:
        payload["value"] = sample.temperature_c
    if sample.vibration_rms_g is not None:
        payload["value"] = sample.vibration_rms_g
    if sample.door_event is not None:
        payload["value"] = sample.door_event
    client.publish(topic, json.dumps(payload), qos=1)


def run_simulator(
    broker_host: str,
    broker_port: int,
    truck_id: str,
    anomaly: str,
    duration_seconds: Optional[int],
) -> None:
    simulator = ColdChainSimulator(anomaly=anomaly)

    client = None
    if mqtt is not None:
        try:
            client = mqtt.Client(client_id=f"logibridge-sim-{truck_id}")
            client.connect(broker_host, broker_port, keepalive=60)
            client.loop_start()
        except Exception as exc:
            _log(f"MQTT unavailable ({exc}); falling back to stdout mode")
            client = None

    _log(
        f"Starting simulator truck_id={truck_id} anomaly={anomaly} "
        f"broker={broker_host}:{broker_port}"
    )

    start = time.time()
    tick = 0
    msg_count = 0
    while True:
        now = time.time()
        if duration_seconds is not None and now - start >= duration_seconds:
            break

        samples = simulator.generate_tick(tick=tick, now=now)

        # temperature at 1 Hz
        temp_topic = f"logibridge/trucks/{truck_id}/sensors/temperature"
        if client:
            _publish_sample(client, temp_topic, samples["temperature"], truck_id)
        else:
            print(temp_topic, samples["temperature"])
        msg_count += 1

        # vibration at 0.5 Hz, publish every 2 seconds
        if tick % 2 == 0:
            vib_topic = f"logibridge/trucks/{truck_id}/sensors/vibration_rms"
            if client:
                _publish_sample(client, vib_topic, samples["vibration_rms"], truck_id)
            else:
                print(vib_topic, samples["vibration_rms"])
            msg_count += 1

        # discrete door events only when present
        if samples["door_event"].door_event is not None:
            door_topic = f"logibridge/trucks/{truck_id}/sensors/door_event"
            if client:
                _publish_sample(client, door_topic, samples["door_event"], truck_id)
            else:
                print(door_topic, samples["door_event"])
            msg_count += 1

        if tick > 0 and tick % 10 == 0:
            _log(f"tick={tick} published_messages={msg_count}")

        tick += 1
        time.sleep(1.0)

    if client:
        client.loop_stop()
        client.disconnect()
    _log(f"Simulator stopped after {tick} ticks; total_messages={msg_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LogiBridge cold-chain sensor simulator")
    parser.add_argument("--anomaly", choices=ANOMALY_CHOICES, default="none")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--truck-id", default="TRUCK_001")
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=None,
        help="Optional finite run duration. If omitted, runs forever.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulator(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        truck_id=args.truck_id,
        anomaly=args.anomaly,
        duration_seconds=args.duration_seconds,
    )
