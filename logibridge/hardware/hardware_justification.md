# Hardware Selection and Roofline Analysis

## Quick Index

- [Constraint Triangle Comparison](#constraint-triangle-comparison)
- [Option 1: Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS)](#option-1-raspberry-pi-5-8-gb--ai-hat-13-tops)
- [Option 2: Jetson Orin Nano Super (67 TOPS)](#option-2-jetson-orin-nano-super-67-tops)
- [Option 3: STM32H7 custom MCU](#option-3-stm32h7-custom-mcu)
- [Quantitative Fleet Economics](#quantitative-fleet-economics)
- [Power and Thermal Envelope](#power-and-thermal-envelope)
- [SLA Fit (90-Second Alert Requirement)](#sla-fit-90-second-alert-requirement)
- [Arithmetic Intensity and Roofline](#arithmetic-intensity-and-roofline)
- [Operational Maintainability Considerations](#operational-maintainability-considerations)
- [Final Recommendation](#final-recommendation)

## Constraint Triangle Comparison

### Option 1: Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS)

- Cost: around Rs15000/truck (pilot 85 trucks around Rs12.75 lakh)
- Power: around 7.5W, within 10W AI budget
- Performance: sufficient for sub-second inference on compact MLP/TFLite
- Verdict: Best balance of performance, power, and cost for pilot scale

### Option 2: Jetson Orin Nano Super (67 TOPS)

- Cost: around Rs45000/truck (pilot around Rs38.25 lakh)
- Power: around 15W, exceeds stated 10W AI budget
- Performance: excellent but unnecessary for this low-FLOP classifier
- Verdict: Overprovisioned and cost-inefficient for 85 truck rollout

### Option 3: STM32H7 custom MCU

- Cost: around Rs3500/truck
- Power: around 0.4W
- Performance/memory: likely constrained for robust MLOps and OTA containerized workflow
- Verdict: attractive power/cost, but high implementation risk and reduced software flexibility

Dominant deployment vertex is reliability-under-constraints (latency + connectivity + maintainability), not raw TOPS. Option 1 is selected.

## Quantitative Fleet Economics

Pilot fleet (85 trucks):

- Option 1 (Pi 5 + AI HAT+): 85 x Rs15000 = Rs12.75 lakh
- Option 2 (Jetson Orin Nano): 85 x Rs45000 = Rs38.25 lakh
- Option 3 (STM32H7 custom): 85 x Rs3500 = Rs2.975 lakh

Full fleet (265 trucks):

- Option 1: 265 x Rs15000 = Rs39.75 lakh
- Option 2: 265 x Rs45000 = Rs119.25 lakh
- Option 3: 265 x Rs3500 = Rs9.275 lakh

Interpretation:

- Option 2 costs approximately Rs25.5 lakh more than Option 1 in pilot and approximately Rs79.5 lakh more at full scale, with limited additional value for a low-FLOP classifier.
- Option 3 has the best capex profile but shifts significant effort to embedded integration and lifecycle operations.

## Power and Thermal Envelope

Given a 10W AI power budget from 12V truck supply (via DC-DC converter):

- Option 1 at around 7.5W stays within budget with practical thermal headroom.
- Option 2 at around 15W exceeds the budget, increasing thermal and electrical integration complexity.
- Option 3 at around 0.4W is excellent on power, but software ecosystem limitations dominate risk.

## SLA Fit (90-Second Alert Requirement)

The SLA requires alerting within 90 seconds after a fault signature appears.

In this design, inference is produced every 10 seconds (30-second window, 10-second step). Therefore, per-inference compute must be comfortably below 10 seconds to leave margin for feature generation, alert publishing, and logging.

- Option 1 comfortably supports this for compact MLP/TFLite models.
- Option 2 also satisfies SLA but is overprovisioned for workload size.
- Option 3 can satisfy simple inference paths, but robust monitoring, OTA orchestration, and maintainability become the bottleneck.

## Arithmetic Intensity and Roofline

Given:
- Model compute per inference: 45 MFLOPs
- Data movement per inference: 18 MB
- Pi 5 peak compute: 16 GFLOP/s
- Pi 5 memory bandwidth: 12 GB/s

Arithmetic Intensity (AI):
AI = 45e6 / 18e6 = 2.5 FLOP/byte

Ridge point:
Ridge = 16 / 12 approx 1.33 FLOP/byte

Since AI (2.5) > ridge (1.33), model trends compute-bound relative to this simplified roofline estimate.

Optimization implication:
- Reduce arithmetic complexity (quantization/pruning) to lower latency
- Still optimize memory locality (operator fusion, cache-friendly tensor layout), but primary gain is from reducing compute per inference

## Operational Maintainability Considerations

Option 1 (Pi 5 + AI HAT+) provides the strongest delivery profile for this assignment because it supports:

- Standard Linux observability and debugging workflow
- Python/TFLite runtime without heavy custom firmware work
- Local broker, alert logging, and monitor agents in a familiar stack
- Practical OTA and deployment automation workflows

Option 3 can be production-viable in tightly controlled firmware ecosystems, but requires substantially higher engineering effort to match the same MLOps and update capabilities.

## Final Recommendation

Recommended edge platform: Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS).

Reason summary:

1. Meets latency and power constraints with margin.
2. Provides strong cost-to-capability balance for both pilot and scale-up.
3. Minimizes implementation and maintenance risk for edge MLOps requirements.

Arguments against alternatives:

- Against Jetson Orin Nano: exceeds power budget and is significantly costlier than needed.
- Against STM32H7 custom MCU: excellent efficiency, but higher software and lifecycle risk for this project scope.
