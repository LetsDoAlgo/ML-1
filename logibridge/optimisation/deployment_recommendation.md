# Deployment Recommendation for 85-Truck Pilot

## Quick Index

- [SLA Translation](#sla-translation)
- [Constraints Considered](#constraints-considered)
- [Variant Selection Logic](#variant-selection-logic)

## SLA Translation

The operational SLA is alert within 90 seconds of fault onset. With 10-second feature step, inference should be significantly below 10 seconds to preserve actuation and network margin. All optimized variants typically satisfy this on Pi-class hardware.

## Constraints Considered

- Device budget: 10W AI power envelope on 12V truck supply.
- Memory/storage limits: edge node should keep model compact for OTA and local storage reliability.
- Safety metric: Class 2 recall must exceed 95 percent.

## Variant Selection Logic

- M1 FP32: strongest baseline quality but larger model and higher compute/energy.
- M2 PTQ INT8: major latency and size reduction with minimal accuracy drop in most cases.
- M3 Pruned + PTQ INT8: best size/latency potential but must be checked for Class 2 recall erosion.

Recommend M2 PTQ INT8 if Class 2 recall remains above 95 percent and overall validation accuracy remains stable. If M3 also keeps Class 2 recall above 95 percent, prefer M3 for fleet OTA efficiency.

Final choice must be made from optimisation/results/benchmark_results.csv after actual run data is populated.
