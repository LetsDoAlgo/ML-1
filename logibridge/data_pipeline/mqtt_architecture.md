# MQTT Architecture (LogiBridge)

## Quick Index

- [Topic Tree](#topic-tree)
- [QoS Strategy](#qos-strategy)
- [Offline-First Behavior](#offline-first-behavior)
- [Data Fusion Decision](#data-fusion-decision)

## Topic Tree

- logibridge/trucks/{truck_id}/sensors/temperature
- logibridge/trucks/{truck_id}/sensors/vibration_rms
- logibridge/trucks/{truck_id}/sensors/door_event
- logibridge/trucks/{truck_id}/inference
- logibridge/trucks/{truck_id}/alerts
- logibridge/trucks/{truck_id}/ops_sync

## QoS Strategy

- QoS 1 for sensor topics to ensure at-least-once delivery without excessive overhead.
- QoS 1 for inference and alerts because operational center visibility is safety-critical.
- QoS 0 is not used for alerts due to potential packet loss on rural links.

## Offline-First Behavior

- Inference runs fully on device and does not require cloud round trips.
- Alerts are written to local log storage first.
- During connectivity gaps, logs are buffered locally and replayed to ops_sync on reconnection.

## Data Fusion Decision

Feature-level fusion is used in this project:
- Temperature and vibration are independently filtered and summarized.
- Final 6-value feature vector is concatenated before inference.

Why not data-level fusion:
- Sampling rates differ (1 Hz vs 0.5 Hz), causing alignment complexity and noisy interpolation.

Why not decision-level fusion:
- Separate sub-model management increases deployment complexity and update risk for edge fleets.
