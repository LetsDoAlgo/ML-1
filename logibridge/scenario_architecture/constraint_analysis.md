# Constraint Analysis for FreightBridge Cold-Chain Deployment

## Quick Index

- [Latency](#latency)
- [Bandwidth](#bandwidth)
- [Connectivity](#connectivity)
- [Privacy](#privacy)

## Latency

The business SLA is detection and alerting within 90 seconds of a fault signature. Given temperature can rise by 1 C per minute after refrigeration failure, cloud-only inferencing is risky because rural round-trip latency can be highly variable and can include intermittent packet retries. Even if average RTT is around 120-250 ms in connected regions, practical edge-to-cloud pipelines include broker hop, serialization, uplink contention, and service queueing. Under weak signal this can push end-to-end reaction time into several seconds or fail completely during dropouts. Edge inference running every 10 seconds per feature window keeps practical decision latency comfortably below the 90-second bound.

## Bandwidth

Per truck raw stream estimate:

- Temperature: 1 Hz x 86400 samples/day x 16 bytes/sample approx = 1.38 MB/day
- Vibration (3-axis at 500 Hz): 1500 samples/s x 86400 x 4 bytes approx = 518.4 MB/day
- Door events: assume 300 events/day x 32 bytes/event approx = 0.01 MB/day

Total raw per truck approx 519.8 MB/day.

Transmission cost at Rs0.10/MB is about Rs51.98 per truck per day.

For 85 pilot trucks: Rs4418/day.

Edge-processed approach transmits only inference and alerts (for example around 10-second inference cadence with compact payload):

- 8640 inference messages/day x 0.2 KB approx 1.7 MB/day
- Alert/event overhead around 0.1 MB/day

Total around 1.8 MB/day, cost around Rs0.18 per truck per day, which is roughly 288x lower than raw streaming.

## Connectivity

The route has 35-90 minute outages at seven points. In cloud-only architecture, no inference and no alerting can occur in these windows, which directly violates cold-chain safety requirements. In the proposed edge architecture, sensing, preprocessing, inference, and local alert persistence continue offline. Once network returns, buffered alert log entries are synchronized to the operations backend using MQTT replay topic.

## Privacy

Pharmaceutical customers need chain-of-custody assurance. Processing on device minimizes raw sensor exposure over public mobile networks and reduces attack surface. Only derived events and operational alerts are transmitted, and data retention can be restricted to signed local logs plus encrypted uplink sync. This supports contractual privacy guarantees and auditable access boundaries.
