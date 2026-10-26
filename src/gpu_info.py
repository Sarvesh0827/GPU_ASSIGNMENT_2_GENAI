"""GPU provenance helpers. Every measurement is labelled with the UUID it came from."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import SYSTEM_DIR


def query_nvidia_smi(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return "nvidia-smi not found on PATH"
    return (completed.stdout or "") + (completed.stderr or "")


def gpu_uuid() -> str:
    raw = query_nvidia_smi(["--query-gpu=uuid", "--format=csv,noheader"])
    line = raw.strip().splitlines()[0].strip() if raw.strip() else "UNKNOWN"
    return line or "UNKNOWN"


def gpu_snapshot() -> dict:
    snap = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_uuid": None,
        "vram_bytes": None,
        "driver_version": None,
        "power_limit_w": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        snap["gpu_name"] = torch.cuda.get_device_name(0)
        snap["vram_bytes"] = int(props.total_memory)
        snap["gpu_uuid"] = gpu_uuid()

    basic = query_nvidia_smi(
        [
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem",
            "--format=csv",
        ]
    )
    snap["nvidia_smi_basic"] = basic.strip()
    for line in basic.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            snap["driver_version"] = parts[2]
            snap["power_limit_w"] = parts[4]
            if not snap["gpu_uuid"] or snap["gpu_uuid"] == "UNKNOWN":
                snap["gpu_uuid"] = parts[1]
    return snap


def capture_system_files() -> dict:
    """Write nvidia-smi -q and a JSON snapshot under system/. Returns the snapshot."""
    SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    snap = gpu_snapshot()
    slug = (snap.get("gpu_name") or "cpu").replace(" ", "_")
    uuid_short = (snap.get("gpu_uuid") or "unknown").replace("GPU-", "")[:12]

    q_path = SYSTEM_DIR / f"nvidia_smi_q_{slug}_{uuid_short}.txt"
    q_path.write_text(query_nvidia_smi(["-q"]), encoding="utf-8")

    json_path = SYSTEM_DIR / f"gpu_snapshot_{slug}_{uuid_short}.json"
    json_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    basic_path = SYSTEM_DIR / f"gpu_basic_{slug}_{uuid_short}.txt"
    basic_path.write_text(snap.get("nvidia_smi_basic") or "", encoding="utf-8")

    snap["nvidia_smi_q_path"] = str(q_path)
    snap["snapshot_path"] = str(json_path)
    print(f"[gpu] {snap.get('gpu_name')}  uuid={snap.get('gpu_uuid')}")
    print(f"[gpu] wrote {q_path.name}")
    return snap


def card_tag() -> str:
    """Short filesystem-safe tag for result filenames."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0).replace(" ", "_")
    uuid = gpu_uuid().replace("GPU-", "")[:8]
    return f"{name}_{uuid}"
