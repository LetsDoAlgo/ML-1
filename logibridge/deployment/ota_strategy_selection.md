# OTA Strategy Selection for FreightBridge

## Quick Index

- [Inputs](#inputs)
- [Cost per update cycle](#cost-per-update-cycle)
- [Full replacement (all 85 trucks at once)](#full-replacement-all-85-trucks-at-once)
- [Canary first (10 trucks, then 75 after validation)](#canary-first-10-trucks-then-75-after-validation)
- [Shadow mode (shadow image to all + active switch)](#shadow-mode-shadow-image-to-all--active-switch)
- [Recommendation](#recommendation)
- [Why not full replacement](#why-not-full-replacement)
- [Why not full shadow everywhere](#why-not-full-shadow-everywhere)

## Inputs

- Model size: 280 KB (INT8 TFLite)
- Cost: Rs0.10 per MB
- Fleet: 85 trucks
- Update frequency: every 6 weeks

## Cost per update cycle

Convert 280 KB to MB:
280 / 1024 = 0.2734 MB

Per truck update cost:
0.2734 x 0.10 = Rs0.02734

### Full replacement (all 85 trucks at once)

- Data: 0.2734 x 85 = 23.24 MB
- Cost: 23.24 x 0.10 = Rs2.32 per cycle

### Canary first (10 trucks, then 75 after validation)

- Stage 1 data: 0.2734 x 10 = 2.73 MB
- Stage 2 data: 0.2734 x 75 = 20.51 MB
- Total data: 23.24 MB
- Total cost: Rs2.32 per cycle

### Shadow mode (shadow image to all + active switch)

Assuming same model payload once per truck:
- Data: 23.24 MB
- Cost: Rs2.32 per cycle

If shadow sends additional telemetry for side-by-side monitoring, operational bandwidth can exceed full/canary.

## Recommendation

Choose canary deployment.

Reasoning:
- Same model payload cost as full replacement but materially lower safety risk.
- Cold-chain is safety critical: validating on 10 trucks first reduces probability of fleet-wide false negatives.
- Rural connectivity is unstable: phased rollout simplifies retry and rollback management in intermittent links.

## Why not full replacement

- Fast but high blast radius if model quality regresses, especially for Class 2 recall.

## Why not full shadow everywhere

- Operational complexity and added telemetry overhead are unnecessary for this model size and 6-week cadence.
