# 10-Stage Edge ML Pipeline Mapping (LogiBridge)

## Quick Index

- [Problem framing](#1-problem-framing-classify-refrigerated-cargo-state-into-normal-warning-critical-with-90-second-alert-sla)
- [Data acquisition](#2-data-acquisition-edge-node-ingests-temperature-vibration-rms-and-door-events-via-mqtt-topics)
- [Data preprocessing](#3-data-preprocessing-moving-average-filtering-30-second-windows-10-second-stride-and-fixed-normalization)
- [Feature engineering](#4-feature-engineering-6-value-vector-combines-thermal-stability-and-vibration-health-indicators)
- [Dataset construction](#5-dataset-construction-simulator-generates-labeled-windows-for-none-temp_drift-and-combined-modes)
- [Model training](#6-model-training-mlp-3216-hidden-units-trained-with-validation-split-and-88-gating)
- [Model optimization](#7-model-optimization-ptq-int8-and-structured-pruning--ptq-create-deployable-variants)
- [Packaging and deployment](#8-packaging-and-deployment-inference-service-containerized-with-model_path-runtime-switch-for-ota-updates)
- [Runtime inference and actions](#9-runtime-inference-and-actions-edge-model-predicts-class-publishes-inference-and-persists-critical-alerts-locally)
- [Monitoring and lifecycle](#10-monitoring-and-lifecycle-psi-drift-monitor-tracks-confidence-distribution-and-triggers-operational-drift-alerts)

1. Problem framing: classify refrigerated cargo state into Normal, Warning, Critical with 90-second alert SLA.
2. Data acquisition: edge node ingests temperature, vibration RMS, and door events via MQTT topics.
3. Data preprocessing: moving average filtering, 30-second windows, 10-second stride, and fixed normalization.
4. Feature engineering: 6-value vector combines thermal stability and vibration health indicators.
5. Dataset construction: simulator generates labeled windows for none, temp_drift, and combined modes.
6. Model training: MLP (32,16 hidden units) trained with validation split and >=88% gating.
7. Model optimization: PTQ INT8 and structured pruning + PTQ create deployable variants.
8. Packaging and deployment: inference service containerized with MODEL_PATH runtime switch for OTA updates.
9. Runtime inference and actions: edge model predicts class, publishes inference, and persists Critical alerts locally.
10. Monitoring and lifecycle: PSI drift monitor tracks confidence distribution and triggers operational drift alerts.
