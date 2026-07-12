"""
LogiBridge Ansible Deployment UI - REAL deployment, not animation.

Each truck is a real folder on disk:
    deployment/fleet_root/<truck-id>/opt/logibridge/
        model.tflite               <- really copied from training/models/
        reference_dist.json        <- really copied from monitoring/
        training_stats.npy         <- really copied from data_pipeline/
        container_state.json       <- JSON file playing the role of the
                                       running Docker container
        deploy_log.jsonl           <- append-only audit trail

Idempotency is real: each task checks whether the desired state already
exists (by SHA-256 for files, by JSON comparison for the container state)
and skips the work if so. That's why the second click of Deploy reports
changed=0 - we actually did nothing, not a cosmetic animation.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st

REPO_ROOT      = Path(__file__).resolve().parent.parent
PLAYBOOK_PATH  = REPO_ROOT / "deployment" / "logibridge_deploy.yml"
INVENTORY_PATH = REPO_ROOT / "deployment" / "inventory.ini"
FLEET_ROOT     = REPO_ROOT / "deployment" / "fleet_root"

SRC_MODEL = REPO_ROOT / "training"      / "models" / "model_int8.tflite"
SRC_STATS = REPO_ROOT / "data_pipeline" / "training_stats.npy"
SRC_PSI   = REPO_ROOT / "monitoring"    / "reference_dist.json"

TRUCKS = ["truck-edge-01", "truck-edge-02", "truck-edge-03"]
IMAGE_TAG = "localhost:5000/logibridge-inference:latest"

# 6-feature order matches data_pipeline/preprocessing.py exactly:
#   [temp_mean, temp_std, temp_roc_C_per_min, vib_rms, vib_peak, vib_kurt]
# Three synthetic sensor scenarios covering the 3 output classes.
INFERENCE_SCENARIOS = [
    {
        "name": "NORMAL - fridge OK, road smooth",
        "features": [3.5, 0.4, 0.05, 0.15, 0.30, 3.0],
        "expected_class": 0,
        "expected_label": "NORMAL",
    },
    {
        "name": "WARNING - fridge warming, mild vibration",
        "features": [8.5, 1.15, 0.85, 0.50, 0.75, 3.4],
        "expected_class": 1,
        "expected_label": "WARNING",
    },
    {
        "name": "CRITICAL - fridge failing, harsh road",
        "features": [12.5, 1.6, 1.20, 1.20, 1.80, 4.2],
        "expected_class": 2,
        "expected_label": "CRITICAL",
    },
]
CLASS_LABELS = ["NORMAL", "WARNING", "CRITICAL"]
CLASS_COLORS = ["#10b981", "#f59e0b", "#ef4444"]


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def truck_root(truck: str) -> Path:
    return FLEET_ROOT / truck / "opt" / "logibridge"


# ---------------------------------------------------------------------------
# Inference verification - runs the DEPLOYED model on each truck
# ---------------------------------------------------------------------------
def run_inference_on_truck(truck: str, features):
    """
    Load the truck's DEPLOYED model.tflite + training_stats.npy from
    fleet_root/<truck>/opt/logibridge/, normalize the features with the
    truck's local stats, run TFLite inference, return (pred_class, probs).

    Uses the exact same preprocessing as inference/inference_service.py.
    Raises RuntimeError if the truck hasn't been deployed to yet.
    """
    import numpy as np

    root = truck_root(truck)
    model_path = root / "model.tflite"
    stats_path = root / "training_stats.npy"
    if not model_path.exists():
        raise RuntimeError(f"{truck}: model not deployed yet")
    if not stats_path.exists():
        raise RuntimeError(f"{truck}: training_stats not deployed yet")

    # Prefer tflite_runtime, fall back to tensorflow.lite.Interpreter.
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    except ImportError:
        try:
            import tensorflow as tf
            TFLiteInterpreter = tf.lite.Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "No TFLite runtime available in .venv312") from exc

    interp = TFLiteInterpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_d  = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    # Normalize using the truck's local stats file (same on all trucks after
    # a successful deploy, so this proves the file arrived intact).
    stats_arr = np.load(stats_path)          # shape (2, 6): [mean, std]
    mean = stats_arr[0].astype(np.float32)
    std  = stats_arr[1].astype(np.float32)

    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    x = (x - mean) / std                     # normalize_features

    if in_d["dtype"].__name__ == "int8":
        scale, zero = in_d["quantization"]
        x = np.round(x / scale + zero).astype(np.int8)
    else:
        x = x.astype(np.float32)

    interp.set_tensor(in_d["index"], x)
    interp.invoke()
    out = interp.get_tensor(out_d["index"])
    if out_d["dtype"].__name__ == "int8":
        scale, zero = out_d["quantization"]
        out = (out.astype(np.float32) - zero) * scale

    # softmax for a probability vector
    e = np.exp(out - np.max(out, axis=1, keepdims=True))
    probs = (e / np.sum(e, axis=1, keepdims=True))[0]
    return int(np.argmax(probs)), [float(p) for p in probs]


def audit_log(truck: str, task: str, verb: str, detail: str) -> None:
    root = truck_root(truck)
    root.mkdir(parents=True, exist_ok=True)
    line = json.dumps({
        "ts":     datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task":   task,
        "verb":   verb,
        "detail": detail,
    })
    with (root / "deploy_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _copy_if_changed(src: Path, dst: Path) -> str:
    if not src.exists():
        return "failed"
    if dst.exists() and sha256_short(src) == sha256_short(dst):
        return "ok"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "changed"


def task_create_opt_dir(truck: str):
    root = truck_root(truck)
    if root.exists():
        return "ok", f"{root.relative_to(REPO_ROOT).as_posix()} already present"
    root.mkdir(parents=True, exist_ok=True)
    return "changed", f"mkdir {root.relative_to(REPO_ROOT).as_posix()}"


def task_copy_model(truck: str):
    dst = truck_root(truck) / "model.tflite"
    verb = _copy_if_changed(SRC_MODEL, dst)
    if verb == "failed":
        return "failed", f"source missing: {SRC_MODEL.name}"
    return verb, (f"copied {SRC_MODEL.name} (sha256:{sha256_short(dst)})"
                    if verb == "changed" else "hash already matches")


def task_copy_psi(truck: str):
    dst = truck_root(truck) / "reference_dist.json"
    verb = _copy_if_changed(SRC_PSI, dst)
    if verb == "failed":
        return "failed", f"source missing: {SRC_PSI.name}"
    return verb, (f"copied {SRC_PSI.name}"
                    if verb == "changed" else "hash already matches")


def task_copy_stats(truck: str):
    dst = truck_root(truck) / "training_stats.npy"
    verb = _copy_if_changed(SRC_STATS, dst)
    if verb == "failed":
        return "failed", f"source missing: {SRC_STATS.name}"
    return verb, (f"copied {SRC_STATS.name}"
                    if verb == "changed" else "hash already matches")


def _read_container_state(truck: str):
    p = truck_root(truck) / "container_state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_container_state(truck: str, state: dict) -> None:
    p = truck_root(truck) / "container_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def task_stop_container(truck: str):
    """
    Idempotent stop: real Ansible + docker_container reports 'ok' when the
    container is already in the desired end state. Since our end state is
    'container running with the target model hash', we only 'stop' when the
    model file has actually changed since the container was started.
    """
    st_ = _read_container_state(truck)
    if st_ is None:
        return "ignored", "no container matching logibridge_inference (ignored)"

    current_hash = sha256_short(SRC_MODEL) if SRC_MODEL.exists() else ""
    running_hash = st_.get("model_sha256_short")

    if st_.get("state") == "started" and running_hash == current_hash:
        return "ok", f"container running the current model ({current_hash}) - no stop needed"

    if st_.get("state") == "stopped":
        return "ok", "already stopped"

    st_["state"] = "stopped"
    st_["stopped_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _write_container_state(truck, st_)
    return "changed", f"stopping to swap model ({running_hash} -> {current_hash})"


def task_pull_image(truck: str):
    st_ = _read_container_state(truck) or {}
    current_hash = sha256_short(SRC_MODEL) if SRC_MODEL.exists() else ""
    if (st_.get("image") == IMAGE_TAG
            and st_.get("model_sha256_short") == current_hash
            and st_.get("image_pulled_at")):
        return "ok", "image already pulled (model hash matches)"
    st_["image"] = IMAGE_TAG
    st_["model_sha256_short"] = current_hash
    st_["image_pulled_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _write_container_state(truck, st_)
    return "changed", f"pulled {IMAGE_TAG} (model:{current_hash})"


def task_start_container(truck: str):
    """
    Idempotent start: matches real docker_container behaviour. If the container
    is already 'started' with the correct image AND model hash, we don't touch
    it - reporting 'ok'. Only when something actually changed (fresh truck,
    stopped by previous task, or model hash mismatch) do we (re)start.
    """
    st_ = _read_container_state(truck) or {}
    current_hash = sha256_short(SRC_MODEL) if SRC_MODEL.exists() else ""
    if (st_.get("state") == "started"
            and st_.get("image") == IMAGE_TAG
            and st_.get("model_sha256_short") == current_hash):
        return "ok", "container already running with target image + model"
    st_.update({
        "name":       "logibridge_inference",
        "state":      "started",
        "image":      IMAGE_TAG,
        "model_sha256_short": current_hash,
        "started_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "env": {
            "MODEL_PATH":       "/opt/logibridge/model.tflite",
            "MQTT_BROKER_HOST": "ops.logibridge.example.com",
            "MQTT_BROKER_PORT": "1883",
            "TRUCK_ID":         truck.upper().replace("-", "_"),
        },
    })
    _write_container_state(truck, st_)
    return "changed", "container started"


def task_verify(truck: str):
    st_ = _read_container_state(truck)
    if st_ and st_.get("state") == "started":
        return "ok", "verify: logibridge_inference is up"
    return "failed", "verify: container not running"


TASK_PIPELINE = [
    ("Create opt directory",       task_create_opt_dir),
    ("Copy model file",            task_copy_model),
    ("Copy PSI reference",         task_copy_psi),
    ("Copy training stats",        task_copy_stats),
    ("Stop old container",         task_stop_container),
    ("Pull updated image",         task_pull_image),
    ("Start inference container",  task_start_container),
    ("Wait and verify container",  task_verify),
]


def parse_playbook_tasks(path: Path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    names = []
    in_tasks = False
    for line in text.splitlines():
        if re.match(r"^\s*tasks\s*:\s*$", line):
            in_tasks = True
            continue
        if in_tasks:
            m = re.match(r"^\s{4}-\s*name:\s*(.+)$", line)
            if m:
                names.append(m.group(1).strip().strip('"').strip("'"))
    return names


@dataclass
class TruckState:
    name: str
    status: str = "pending"
    current_task: str = ""
    tasks_done: int = 0
    ok_count: int = 0
    changed_count: int = 0
    ignored_count: int = 0
    failed_count: int = 0
    task_log: list = field(default_factory=list)

    def reset(self):
        self.status = "pending"
        self.current_task = ""
        self.tasks_done = 0
        self.ok_count = self.changed_count = self.ignored_count = self.failed_count = 0
        self.task_log = []


def _init_state():
    if "trucks" not in st.session_state:
        st.session_state.trucks = {t: TruckState(name=t) for t in TRUCKS}
    if "run_history" not in st.session_state:
        st.session_state.run_history = []
    if "run_number" not in st.session_state:
        st.session_state.run_number = 0
    if "playbook_task_names" not in st.session_state:
        st.session_state.playbook_task_names = parse_playbook_tasks(PLAYBOOK_PATH)


STATUS_COLORS = {
    "pending":     ("#6b7280", "PENDING"),
    "in_progress": ("#3b82f6", "RUNNING"),
    "ok":          ("#10b981", "OK"),
    "changed":     ("#f59e0b", "CHANGED"),
    "ignored":     ("#a78bfa", "IGNORED"),
    "failed":      ("#ef4444", "FAILED"),
}


def truck_card_html(truck: TruckState, total_tasks: int) -> str:
    color, label = STATUS_COLORS.get(truck.status, ("#6b7280", "PENDING"))
    pct = int((truck.tasks_done / max(1, total_tasks)) * 100)
    current = truck.current_task or "waiting"
    root = truck_root(truck.name)
    fs_note = ""
    if root.exists():
        files = sorted(p.name for p in root.iterdir() if p.is_file())
        if files:
            fs_note = ("<div style='margin-top:6px;color:#6b7280;font-size:0.7rem;'>"
                        f"DIR {root.relative_to(REPO_ROOT).as_posix()}<br>"
                        f"&nbsp;&nbsp;{'  '.join(files)}</div>")
    return f"""
    <div style="border:2px solid {color};border-radius:12px;padding:14px;
                margin-bottom:12px;
                background:linear-gradient(135deg,rgba(255,255,255,0.02),rgba(255,255,255,0.05));">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:1.1rem;font-weight:600;color:#e5e7eb;">{truck.name}</div>
        <div style="color:{color};font-weight:700;font-size:0.9rem;">{label}</div>
      </div>
      <div style="margin-top:8px;color:#9ca3af;font-size:0.85rem;">
        Task: <span style="color:#e5e7eb;">{current}</span>
      </div>
      <div style="margin-top:10px;background:#1f2937;border-radius:6px;height:8px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{color};transition:width 0.3s;"></div>
      </div>
      <div style="margin-top:8px;display:flex;gap:12px;font-size:0.8rem;color:#9ca3af;flex-wrap:wrap;">
        <span>ok=<b style="color:#10b981">{truck.ok_count}</b></span>
        <span>changed=<b style="color:#f59e0b">{truck.changed_count}</b></span>
        <span>ignored=<b style="color:#a78bfa">{truck.ignored_count}</b></span>
        <span>failed=<b style="color:#ef4444">{truck.failed_count}</b></span>
        <span>{truck.tasks_done}/{total_tasks}</span>
      </div>
      {fs_note}
    </div>
    """


def render_fleet(placeholder, total_tasks: int):
    html = "".join(truck_card_html(t, total_tasks) for t in st.session_state.trucks.values())
    placeholder.markdown(html, unsafe_allow_html=True)


def render_task_log(placeholder):
    rows = []
    for truck in st.session_state.trucks.values():
        for e in truck.task_log[-6:]:
            color, _ = STATUS_COLORS.get(e["verb"], ("#6b7280", "?"))
            rows.append(
                f"<div style='font-family:Consolas,monospace;font-size:0.8rem;color:#d1d5db;'>"
                f"<span style='color:{color};font-weight:700;'>{e['verb']:<8}</span>"
                f" [{truck.name}] {e['task']}"
                f" <span style='color:#6b7280;'>- {e['detail']}</span>"
                f"</div>"
            )
    placeholder.markdown(("\n".join(rows[-30:])) or
                          "<i style='color:#6b7280'>No task activity yet.</i>",
                          unsafe_allow_html=True)


def render_recap(placeholder):
    if not st.session_state.run_history:
        placeholder.info("No runs yet. Click Deploy to Fleet to start.")
        return
    parts = []
    for h in st.session_state.run_history:
        color = "#f59e0b" if h["changed"] > 0 else "#10b981"
        badge = "CHANGED" if h["changed"] > 0 else "NO DRIFT"
        parts.append(
            f"<div style='padding:10px 14px;margin-bottom:8px;"
            f"border-left:4px solid {color};background:#111827;border-radius:6px;'>"
            f"<div style='font-weight:600;color:#e5e7eb;'>Run #{h['run']} - {h['kind']} &nbsp; "
            f"<span style='color:{color}'>{badge}</span></div>"
            f"<div style='font-family:Consolas,monospace;font-size:0.85rem;color:#d1d5db;margin-top:4px;'>"
            f"PLAY RECAP: ok={h['ok']}  changed={h['changed']}  ignored={h['ignored']}  "
            f"failed={h['failed']}  unreachable=0"
            f"</div></div>"
        )
    placeholder.markdown("".join(parts), unsafe_allow_html=True)


def run_deployment(fleet_placeholder, log_placeholder, recap_placeholder,
                    kind: str, speed: float):
    total_tasks = len(TASK_PIPELINE) + 1

    for truck in st.session_state.trucks.values():
        truck.reset()
        truck.status = "in_progress"

    for truck in st.session_state.trucks.values():
        truck.current_task = "Gathering Facts"
        truck.tasks_done = 1
        truck.ok_count += 1
        exists = truck_root(truck.name).exists()
        truck.task_log.append({
            "task": "Gathering Facts", "verb": "ok",
            "detail": f"fleet_root/{truck.name}/ {'present' if exists else 'not yet present'}",
        })
        audit_log(truck.name, "Gathering Facts", "ok",
                    "present" if exists else "new host")
    render_fleet(fleet_placeholder, total_tasks)
    render_task_log(log_placeholder)
    time.sleep(0.4 / speed)

    last_verb = "ok"
    for i, (task_name, fn) in enumerate(TASK_PIPELINE, start=2):
        for truck in st.session_state.trucks.values():
            verb, detail = fn(truck.name)
            last_verb = verb
            truck.current_task = task_name
            truck.tasks_done = i
            if verb == "changed":
                truck.changed_count += 1
                truck.status = "changed"
            elif verb == "ok":
                truck.ok_count += 1
                truck.status = "in_progress"
            elif verb == "ignored":
                truck.ignored_count += 1
                truck.status = "in_progress"
            elif verb == "failed":
                truck.failed_count += 1
                truck.status = "failed"
            truck.task_log.append({"task": task_name, "verb": verb, "detail": detail})
            audit_log(truck.name, task_name, verb, detail)
        render_fleet(fleet_placeholder, total_tasks)
        render_task_log(log_placeholder)
        time.sleep((0.35 if last_verb == "changed" else 0.2) / speed)

    for truck in st.session_state.trucks.values():
        if truck.failed_count:
            truck.status = "failed"
        elif truck.changed_count:
            truck.status = "changed"
        else:
            truck.status = "ok"
        truck.current_task = "DONE"
    render_fleet(fleet_placeholder, total_tasks)

    ok      = sum(t.ok_count      for t in st.session_state.trucks.values()) // len(TRUCKS)
    changed = sum(t.changed_count for t in st.session_state.trucks.values()) // len(TRUCKS)
    ignored = sum(t.ignored_count for t in st.session_state.trucks.values()) // len(TRUCKS)
    failed  = sum(t.failed_count  for t in st.session_state.trucks.values()) // len(TRUCKS)

    st.session_state.run_number += 1
    st.session_state.run_history.append({
        "run": st.session_state.run_number, "kind": kind,
        "ok": ok, "changed": changed, "ignored": ignored, "failed": failed,
    })
    render_recap(recap_placeholder)


st.set_page_config(page_title="LogiBridge Fleet Deployment", layout="wide")
_init_state()

st.markdown(
    """
    <div style='display:flex;align-items:center;gap:12px;'>
      <div>
        <div style='font-size:1.6rem;font-weight:700;color:#e5e7eb;'>LogiBridge Fleet Deployment</div>
        <div style='color:#9ca3af;'>Real Ansible-style OTA rollout - actually copies files, computes hashes, and manages container state on disk.</div>
      </div>
    </div>
    <hr style='border-color:#374151;'>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Playbook")
    st.caption(f"`{PLAYBOOK_PATH.relative_to(REPO_ROOT).as_posix()}`")
    st.markdown(f"**Tasks parsed from YAML:** {len(st.session_state.playbook_task_names)}")
    for n in st.session_state.playbook_task_names:
        st.markdown(f"- {n}")
    st.markdown("---")
    st.markdown("### Inventory")
    st.caption(f"`{INVENTORY_PATH.relative_to(REPO_ROOT).as_posix()}`")
    for t in TRUCKS:
        st.markdown(f"- `{t}`")
    st.markdown("---")
    st.markdown("### Controls")
    speed = st.select_slider("Playback speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0)
    if st.button("Wipe fleet_root (fresh trucks)", use_container_width=True,
                  help="Deletes deployment/fleet_root so the next deploy really rebuilds every truck."):
        if FLEET_ROOT.exists():
            shutil.rmtree(FLEET_ROOT)
        for truck in st.session_state.trucks.values():
            truck.reset()
        st.session_state.run_history = []
        st.session_state.run_number = 0
        st.success(f"Deleted {FLEET_ROOT.relative_to(REPO_ROOT).as_posix()}")
        st.rerun()
    if st.button("Reset dashboard only", use_container_width=True):
        for truck in st.session_state.trucks.values():
            truck.reset()
        st.session_state.run_history = []
        st.session_state.run_number = 0
        st.rerun()

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### Fleet Status")
    fleet_placeholder = st.empty()
    render_fleet(fleet_placeholder, total_tasks=len(TASK_PIPELINE) + 1)

    st.markdown("### Task Stream (last actions per truck)")
    log_placeholder = st.empty()
    render_task_log(log_placeholder)

    with st.expander("Browse deployed fleet_root (real files on disk)"):
        st.caption(f"`{FLEET_ROOT.relative_to(REPO_ROOT).as_posix()}`")
        if not FLEET_ROOT.exists():
            st.info("fleet_root does not exist yet - click Deploy to Fleet.")
        else:
            for truck in TRUCKS:
                root = truck_root(truck)
                if not root.exists():
                    st.markdown(f"**{truck}** - _no files yet_")
                    continue
                files = sorted(root.iterdir(), key=lambda p: p.name)
                st.markdown(f"**{truck}** -> `{root.relative_to(REPO_ROOT).as_posix()}`")
                rows = []
                for f in files:
                    if f.is_file():
                        sz = f.stat().st_size
                        h  = sha256_short(f) if f.suffix != ".jsonl" else "-"
                        rows.append(
                            f"<div style='font-family:Consolas,monospace;font-size:0.8rem;color:#d1d5db;'>"
                            f"&nbsp;&nbsp;{f.name:<25} {sz:>8} B  sha256:{h}</div>"
                        )
                st.markdown("".join(rows), unsafe_allow_html=True)
                cs = root / "container_state.json"
                if cs.exists():
                    try:
                        st.code(cs.read_text(encoding="utf-8"), language="json")
                    except Exception:
                        pass

with right:
    st.markdown("### Deployment")
    st.markdown(
        "<div style='color:#9ca3af;font-size:0.9rem;'>Real work equivalent to: "
        "<code>ansible-playbook -i inventory.ini logibridge_deploy.yml</code></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        deploy_clicked = st.button("Deploy to Fleet", type="primary",
                                     use_container_width=True)
    with c2:
        rerun_clicked = st.button("Re-run (idempotency)",
                                     use_container_width=True,
                                     disabled=(st.session_state.run_number == 0))

    st.markdown("### Play Recap")
    recap_placeholder = st.empty()
    render_recap(recap_placeholder)

    with st.expander("Proof: this is real, not animation"):
        st.markdown(
            """
            - The **first** click creates `deployment/fleet_root/<truck>/opt/logibridge/`
              on your real disk and copies `model.tflite`, `reference_dist.json`,
              `training_stats.npy` into every truck folder.
            - The **second** click computes SHA-256 of source vs. destination for
              every file. Because they match, `_copy_if_changed()` skips the
              write and returns `ok` - that's genuine idempotency.
            - **container_state.json** in each truck folder is our stand-in for
              a running Docker container. Its `image` and `model_sha256_short`
              fields are checked before pulling or starting.
            - Every task appends a line to **deploy_log.jsonl** in each truck
              folder - a real audit trail Ops could grep across the fleet.
            - **Wipe fleet_root** in the sidebar deletes everything so you
              can watch a genuine cold-start rollout.
            """
        )
        if SRC_MODEL.exists():
            st.markdown(f"**Source model hash:** `sha256:{sha256_short(SRC_MODEL)}` "
                          f"({SRC_MODEL.stat().st_size} B)")

    # -----------------------------------------------------------------
    # Inference verification - loads DEPLOYED model on each truck and
    # runs test scenarios through it. Proves the deployment actually works.
    # -----------------------------------------------------------------
    st.markdown("### Inference Verification")
    st.markdown(
        "<div style='color:#9ca3af;font-size:0.85rem;'>"
        "Runs three test sensor scenarios through each truck's <b>deployed</b> "
        "model.tflite. Same TFLite runtime + preprocessing as the production "
        "<code>inference_service.py</code>. Green = truck correctly classifies "
        "the sample."
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button("Run inference on all trucks", use_container_width=True):
        results_by_truck = {}
        for truck in TRUCKS:
            try:
                truck_results = []
                for sc in INFERENCE_SCENARIOS:
                    pred, probs = run_inference_on_truck(truck, sc["features"])
                    ok = (pred == sc["expected_class"])
                    truck_results.append({
                        "scenario": sc["name"],
                        "expected": sc["expected_label"],
                        "expected_class": sc["expected_class"],
                        "predicted": CLASS_LABELS[pred],
                        "predicted_class": pred,
                        "probs": probs,
                        "ok": ok,
                    })
                results_by_truck[truck] = {"ok": True, "results": truck_results}
            except Exception as exc:
                results_by_truck[truck] = {"ok": False, "error": str(exc)}
        st.session_state.inference_results = results_by_truck

    inf_results = st.session_state.get("inference_results")
    if inf_results:
        for truck, payload in inf_results.items():
            st.markdown(f"**{truck}**")
            if not payload.get("ok"):
                st.error(payload["error"])
                continue
            correct = sum(1 for r in payload["results"] if r["ok"])
            total   = len(payload["results"])
            summary_color = "#10b981" if correct == total else "#ef4444"
            st.markdown(
                f"<div style='color:{summary_color};font-weight:600;'>"
                f"{correct}/{total} scenarios classified correctly</div>",
                unsafe_allow_html=True,
            )
            for r in payload["results"]:
                mark_color = "#10b981" if r["ok"] else "#ef4444"
                mark_text  = "PASS" if r["ok"] else "FAIL"
                prob_bars = ""
                for cls_idx, p in enumerate(r["probs"]):
                    w = int(p * 100)
                    col = CLASS_COLORS[cls_idx]
                    prob_bars += (
                        f"<div style='display:flex;align-items:center;gap:6px;"
                        f"font-size:0.75rem;color:#9ca3af;'>"
                        f"<span style='width:60px;'>{CLASS_LABELS[cls_idx]}</span>"
                        f"<div style='flex:1;background:#1f2937;border-radius:3px;"
                        f"height:6px;overflow:hidden;'>"
                        f"<div style='width:{w}%;height:100%;background:{col};'></div>"
                        f"</div>"
                        f"<span style='width:44px;text-align:right;'>{p*100:.1f}%</span>"
                        f"</div>"
                    )
                st.markdown(
                    f"<div style='border-left:3px solid {mark_color};"
                    f"padding:6px 10px;margin:4px 0;background:#111827;"
                    f"border-radius:4px;'>"
                    f"<div style='font-size:0.85rem;color:#e5e7eb;'>"
                    f"<b style='color:{mark_color};'>[{mark_text}]</b> {r['scenario']} "
                    f"<span style='color:#9ca3af;'>&mdash; expected "
                    f"<b>{r['expected']}</b>, got <b>{r['predicted']}</b>"
                    f"</span></div>"
                    f"<div style='margin-top:6px;'>{prob_bars}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

if deploy_clicked:
    run_deployment(fleet_placeholder, log_placeholder, recap_placeholder,
                    kind="first rollout", speed=speed)

if rerun_clicked:
    run_deployment(fleet_placeholder, log_placeholder, recap_placeholder,
                    kind="idempotency re-run", speed=speed)
