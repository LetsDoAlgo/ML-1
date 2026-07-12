# LogiBridge - Edge AI for Cold-Chain Fleet Monitoring

## Quick Index

- [What is implemented](#what-is-implemented)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [End-to-end run order](#end-to-end-run-order)
- [Notes for assignment evidence](#notes-for-assignment-evidence)

This repository implements the LogiEdge mini-project for AIML ZG535 (Machine Learning on Edge).

## What is implemented

- Offline-first edge inference architecture for refrigerated trucks
- Sensor simulator with anomaly injection modes and MQTT publishing
- Full preprocessing pipeline with fixed training normalization stats
- MLP training, PTQ conversion, and pruning + PTQ pipeline
- Dockerized inference service with MODEL_PATH switch
- PSI-based drift monitor on model confidence
- Ansible deployment playbook with 7 required tasks
- Benchmark script for 5 metrics and deployment recommendation inputs

## Repository layout

- scenario_architecture: constraint analysis and architecture notes
- hardware: device tradeoff and Roofline analysis
- data_pipeline: simulator, preprocessing, MQTT design
- training: dataset generation, training, conversion, pruning
- inference: runtime service and Dockerfile
- monitoring: PSI monitor and reference distribution
- deployment: Ansible playbook
- optimisation: benchmarking and result artifacts
- reports: report placeholders for Phase 1, Phase 2, and final report

## Setup

1. Create and activate virtual environment.
   Recommended Python version: 3.11 (TensorFlow wheels are not available on Python 3.14 in this environment).
2. Install dependencies:
   pip install -r requirements.txt
3. Optional: start local Mosquitto broker on port 1883.
4. On Windows, use the one-file demo launcher instead of the `python` alias:
   .\run_inference.bat
5. Optional launcher modes:
   .\run_inference.bat demo combined TRUCK_001
   .\run_inference.bat inference placeholder TRUCK_001
   .\run_inference.bat monitor placeholder TRUCK_001
   .\run_inference.bat simulator combined TRUCK_001

## End-to-end run order

1. Generate dataset
   python training/generate_dataset.py --out-dir training
2. Train baseline model and stats
   python training/train_model.py --data-dir training --out-dir training/models
3. Convert full INT8 PTQ
   python training/convert_ptq.py
4. Prune + quantize
   python training/prune_quantise.py
5. Start simulator
   .\run_inference.bat demo combined TRUCK_001
6. Run inference service only
   .\run_inference.bat inference placeholder TRUCK_001
7. Run drift monitor only
   .\run_inference.bat monitor placeholder TRUCK_001
8. Benchmark variants
   python optimisation/benchmark.py

## Notes for assignment evidence

- Use --anomaly combined during live runs to trigger Warning/Critical behavior.
- For normalization sensitivity experiment, use shifted_stats in data_pipeline/preprocessing.py and compare validation accuracy.
- For Docker OTA layer-cache demo, rebuild after changing only inference/model.tflite and capture layer cache output.
- For Ansible idempotency, run deployment/logibridge_deploy.yml twice and capture changed=0 on second run.
