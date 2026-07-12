# MLOps Monitoring Notes

## Quick Index

- [PSI Monitoring Flow](#psi-monitoring-flow)
- [Demo Checklist](#demo-checklist)
- [Ansible Idempotency Evidence](#ansible-idempotency-evidence)

## PSI Monitoring Flow

- Reference distribution stored in monitoring/reference_dist.json.
- Inference confidence values are consumed from logibridge/trucks/{truck_id}/inference.
- Rolling window: last 100 confidence values.
- Every 60 seconds, PSI is computed against reference bins.
- Alert threshold: PSI > 0.25.

## Demo Checklist

1. Run clean simulator input and observe low PSI.
2. Inject anomaly combined mid-run and show PSI crossing 0.25.
3. Restore clean input and show PSI dropping below 0.10.

## Ansible Idempotency Evidence

- Run deployment/logibridge_deploy.yml twice without changes.
- First run should deploy artifacts and container.
- Second run should report changed=0 for idempotent state.
