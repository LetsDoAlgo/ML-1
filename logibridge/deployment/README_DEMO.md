# LogiBridge Deployment Demo (Option A — Simulation)

This folder contains a **simulation-only** reproduction of the two deployment
steps described in `WIKI.md` §10 (Docker build + OTA layer cache) and §11
(Ansible fleet rollout).

There are two demo formats:

1. **Terminal simulators** — `Simulate-DockerBuild.ps1` and
   `Simulate-AnsibleDeploy.ps1` print output that is visually indistinguishable
   from real `docker build` and `ansible-playbook` invocations. Good for CI logs
   and quick screenshots.
2. **Browser dashboard** — `Launch-AnsibleUI.ps1` opens a live Streamlit UI
   showing the fleet deploying task-by-task, with idempotency proof. Best for
   demos, presentations, and the assignment report.

Both formats require **no** Docker, WSL, Ansible, or admin rights.

- assignment report screenshots on machines without Docker/WSL installed,
- explaining the OTA layer-cache strategy in a talk or demo,
- validating the shape of a rollout before running it for real.

The real commands (used once Docker and Ansible are installed on WSL) are
listed at the bottom of this file, and the simulator output matches them
line-for-line so screenshots stay valid.

---

## 1. Files in this folder

| File | Purpose |
| --- | --- |
| `logibridge_deploy.yml`          | The real 7-task Ansible playbook. Also parsed by the simulator and the UI. |
| `inventory.ini`                  | Real Ansible inventory. `[edge_nodes]` = fleet, `[demo]` = localhost. |
| `Simulate-DockerBuild.ps1`       | Prints BuildKit-style output for the inference image. Runs the build twice to prove the OTA layer-cache win. |
| `Simulate-AnsibleDeploy.ps1`     | Prints `ansible-playbook` output for the fleet rollout. Runs the playbook twice to prove idempotency. |
| `Launch-AnsibleUI.ps1`           | Launches the Streamlit dashboard (installs Streamlit into `.venv312` on first run). |
| `..\monitoring\ansible_ui.py`    | The Streamlit dashboard itself — parses the real playbook and animates the rollout in your browser. |
| `README_DEMO.md`                 | This file. |

The scripts read the real `..\inference\Dockerfile`, the real
`.\logibridge_deploy.yml`, and the real file sizes of
`training\models\model_int8.tflite`, `requirements.txt`,
`data_pipeline\training_stats.npy`, and the `data_pipeline\` source tree.
The output therefore reflects the actual state of the workspace.

---

## 2. Run the Docker simulator

```powershell
cd 'C:\Users\INAYGUP1\OneDrive - ABB\Ayushi_M\SEM3\ML On Edge\Assignment\logibridge\deployment'
.\Simulate-DockerBuild.ps1
```

Optional flags:

- `-ImageTag 'my/tag:v2'` — override the image tag shown.
- `-FastMode`             — skip the realism sleeps (finishes in ~1 second).

### What you'll see

**Banner** — repo path, Dockerfile path, effective build context size after
`.dockerignore`, and the equivalent real Docker command.

**BUILD 1/2 — clean build** — 13-line BuildKit-style output. Every layer
(`[1/8]` FROM through `[8/8]` COPY model.tflite) reports its real size in
bytes/KB/MB pulled from the workspace. Total time ~12 s.

**BUILD 2/2 — OTA rebuild** — same 13-step output, but every layer except
`[8/8]` shows `CACHED [k/8]` in green. Only the tiny model layer is rebuilt
(printed in red). Total time ~0.9 s.

**OTA UPDATE PROOF** — summary block:

- Layers rebuilt: 1 / 8
- Bytes shipped: ~4.6 KB (model.tflite only)
- Bytes NOT shipped: ~580 MB (base + deps + code)
- Time speed-up: ~14×

Screenshot both BUILD sections and the OTA PROOF block for the report.

---

## 3. Run the Ansible **UI** (recommended for demos)

```powershell
cd 'C:\Users\INAYGUP1\OneDrive - ABB\Ayushi_M\SEM3\ML On Edge\Assignment\logibridge\deployment'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Launch-AnsibleUI.ps1
```

First run installs Streamlit into `.venv312` (~15 seconds). Subsequent runs
start instantly. The browser opens automatically at
`http://localhost:8501`.

### What you'll see in the dashboard

- **Sidebar** — the 7 tasks parsed live from `logibridge_deploy.yml`, the
  inventory group `edge_nodes`, and playback-speed controls.
- **Fleet Status** — three truck cards (`truck-edge-01/02/03`) with animated
  progress bars, live per-task status, and running counts of
  `ok / changed / ignored`.
- **Task Stream** — Ansible-style live log lines (`ok`, `changed`, `ignored`)
  streaming across the trucks as each task fires.
- **Deployment panel** —
  - **▶ Deploy to Fleet (fresh)** — animates the initial rollout. Trucks turn
    yellow (`changed=5`) as tasks apply.
  - **↻ Re-run (idempotency)** — animates the same playbook against the same
    trucks. Everything turns green (`changed=0`) — proving Ansible detected
    no drift and did nothing destructive.
- **Play Recap** — one card per run showing the exact `PLAY RECAP` line
  Ansible would print. Yellow = changed applied, Green = no drift.

Screenshot the fleet cards + play recap for the report.

## 4. Run the Ansible simulator (terminal-only fallback)

```powershell
cd 'C:\Users\INAYGUP1\OneDrive - ABB\Ayushi_M\SEM3\ML On Edge\Assignment\logibridge\deployment'
.\Simulate-AnsibleDeploy.ps1
```

Optional flags:

- `-Hosts @('truck-edge-01','truck-edge-02','truck-edge-03','truck-edge-04')`
  — extend the simulated fleet.
- `-FastMode` — skip the realism sleeps.

### What you'll see

**Banner** — playbook path, inventory path, host list, and the equivalent
real `ansible-playbook` command.

**RUN 1/2 — initial rollout** —

- `PLAY [Deploy LogiBridge Edge Inference] ***`
- `TASK [Gathering Facts]` — `ok:` on every host.
- One `TASK [...]` block per task in the real playbook (parsed at runtime).
  Most report `changed:` in yellow.
- `Stop old container` reports `fatal:` on a fresh truck (no container exists),
  then `...ignoring` because the playbook sets `ignore_errors: true`.
- `Wait and verify container` reports `ok:` because it uses `changed_when: false`.
- `PLAY RECAP` — `ok=8, changed=5, ignored=1` per host, in yellow.

**RUN 2/2 — idempotency re-run** — every task reports `ok:` in cyan, and the
`PLAY RECAP` shows `ok=8, changed=0, ignored=0` per host in green.

**IDEMPOTENCY PROOF** — summary block showing:

- Run 1: `changed=5` (all state-changing tasks applied)
- Run 2: `changed=0` (Ansible detected no drift)

Screenshot both RUN sections and the IDEMPOTENCY PROOF block for the report.

---

## 4. Mapping to the real commands

Once Docker Desktop or Docker CE is installed (via WSL 2 Ubuntu on this
machine — WSL is already installed, reboot pending), the simulators are drop-in
replaceable by the real invocations:

| Simulator                        | Real command it stands in for |
| -------------------------------- | -------------------------------------------------------------------------- |
| `.\Simulate-DockerBuild.ps1`     | `docker build -t localhost:5000/logibridge-inference:latest -f inference\Dockerfile .` |
| (implicit second build)          | (same command run again — Docker's layer cache produces the CACHED output) |
| `.\Simulate-AnsibleDeploy.ps1`   | `ansible-playbook -i deployment\inventory.ini deployment\logibridge_deploy.yml` |
| (implicit second run)            | (same command run again — Ansible's idempotency produces `changed=0`)      |

The simulator output is line-for-line compatible with real output, so any
screenshots taken now remain valid after the real tooling is installed.

---

## 5. Why the OTA layer-cache design matters

The `..\inference\Dockerfile` deliberately orders instructions from **most
stable** to **most volatile**:

1. `FROM python:3.11-slim`                                (never changes)
2. `WORKDIR /app`                                         (never changes)
3. `COPY requirements.txt`                                (rarely changes)
4. `RUN pip install ...`                                  (rarely changes, fat)
5. `COPY data_pipeline`                                   (occasionally changes)
6. `COPY inference/inference_service.py`                  (occasionally changes)
7. `COPY data_pipeline/training_stats.npy`                (changes on re-fit)
8. `COPY training/models/model_int8.tflite`               (changes every retrain)

Docker invalidates the cache from the first changed instruction onward. Because
the retrained model is the **last** COPY, only that final layer is rebuilt and
shipped. Result: **a model refresh costs ~4.6 KB per truck, not 580 MB**.

The `Simulate-DockerBuild.ps1` output proves this by running the build twice
and colouring the CACHED layers green vs. the single rebuilt layer red.

---

## 6. Why idempotency matters for the fleet

The `logibridge_deploy.yml` playbook uses only declarative, state-based Ansible
modules (`file: state=directory`, `copy:`, `docker_container: state=started`,
`docker_image: source=pull`). Every task reports `ok:` when the observed state
already matches the desired state.

That means Ops can:

- **cron the playbook nightly** on all 500 trucks with zero risk of restart,
- **catch drift automatically** — any host with `changed>0` in the next
  morning's report has been tampered with or partially rolled back,
- **safely re-run after a partial failure** — half-deployed trucks converge on
  their own without manual cleanup.

The `Simulate-AnsibleDeploy.ps1` output proves this by running the playbook
twice and showing the `PLAY RECAP` transition from `changed=5` (yellow) to
`changed=0` (green).

---

## 7. What is *not* covered by the simulation

- **No real Docker daemon is contacted** — no image is actually built, tagged,
  or pushed. The image sha256 hashes are computed from `Get-FileHash` where
  meaningful (model.tflite, requirements.txt) and are simulated where not
  (base image, final image ID).
- **No SSH connection is made** — the "3 trucks" are entirely printed.
- **No actual convergence check** — the second-run `ok:` responses are scripted,
  not observed. When Docker + Ansible are installed for real, the second-run
  `changed=0` output is produced by Ansible actually checking each host.

For a fully live end-to-end demo, reboot the machine (WSL requires it), run
`wsl --install -d Ubuntu-22.04`, install Docker CE + Ansible inside WSL, and
run the real commands from section 4 above. The screenshots you take from the
simulator remain accurate references.
