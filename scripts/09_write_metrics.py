"""Rebuild METRICS.md from whatever result JSON/CSV files exist."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT, RESULTS_DIR, personal_params_block


def _load(name: str):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return p.read_text(encoding="utf-8")


def _fmt(x, nd=4):
    if x is None or x == "":
        return "TBD (run not yet completed)"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    baseline = None
    bpath = RESULTS_DIR / "hw4a_baseline_summary.json"
    if bpath.exists():
        baseline = json.loads(bpath.read_text(encoding="utf-8"))

    lines = [
        "# HW4.5 — Training Throughput and Memory Engineering",
        "",
        "## Personal parameters (Section 0.1)",
        "",
        "```",
        personal_params_block().rstrip(),
        "```",
        "",
        "This file is the HW4.5 summary. It is **not** an HW5 baseline.",
        "",
        "## HW4.a baseline (kept for HW4.5)",
        "",
    ]
    if baseline:
        lines += [
            f"- GPU: `{baseline.get('gpu_name')}`  UUID `{baseline.get('gpu_uuid')}`",
            f"- Batch size: {baseline.get('batch_size')}",
            f"- Peak GPU memory: **{_fmt(baseline.get('peak_memory_gb'), 3)} GB**",
            f"- Wall-clock training time: **{_fmt(baseline.get('wall_clock_sec'), 1)} s**",
            f"- Test accuracy: {_fmt(baseline.get('test_accuracy'), 4)}",
            f"- Validation stopped improving at epoch: {baseline.get('stopped_improving_epoch')}",
            f"- Checkpoint: `{baseline.get('checkpoint_dir')}`",
            "",
            "### Per-epoch log",
            "",
            "| Epoch | Train loss | Val loss | Val acc | Epoch s | samp/s |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in baseline.get("epoch_log") or []:
            lines.append(
                f"| {r['epoch']} | {_fmt(r['train_loss'])} | {_fmt(r['val_loss'])} | "
                f"{_fmt(r['val_accuracy'])} | {_fmt(r['epoch_seconds'], 1)} | {_fmt(r['samples_per_sec'], 2)} |"
            )
    else:
        lines.append("_Run `python scripts/01_bert_finetune.py` to fill this section._")

    lines += [
        "",
        "## Table HW4.1 — Serving comparison",
        "",
        "| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) | GPU |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    serving_files = sorted(RESULTS_DIR.glob("serving_*.json"))
    if serving_files:
        for f in serving_files:
            if f.name.endswith("_summary.json"):
                continue
            rows = json.loads(f.read_text(encoding="utf-8"))
            for r in rows:
                label = f"{r.get('engine')}, batch {r.get('batch_size')}"
                lines.append(
                    f"| {label} | {_fmt(r.get('tokens_per_sec'), 2)} | {_fmt(r.get('p50_latency_ms'), 1)} | "
                    f"{_fmt(r.get('p95_latency_ms'), 1)} | {r.get('gpu_name')} |"
                )
    else:
        lines.append("| HF pipeline, batch 1 | TBD | TBD | TBD | |")
        lines.append("| vLLM, batch 1 | TBD | TBD | TBD | |")
        lines.append("| vLLM, batch 8 | TBD | TBD | TBD | |")
        lines.append("| vLLM, batch 32 | TBD | TBD | TBD | |")

    lines += [
        "",
        "## Table HW4.5.1 — Memory techniques (one table per card)",
        "",
    ]
    tech_files = sorted(RESULTS_DIR.glob("partB_techniques_*.csv"))
    if tech_files:
        import csv

        for f in tech_files:
            lines += [f"### {f.stem}", "", "| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy |", "| :--- | ---: | ---: | ---: | ---: |"]
            with f.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    lines.append(
                        f"| {row.get('technique')} | {row.get('max_batch')} | "
                        f"{_fmt(float(row['throughput_samp_s']) if row.get('throughput_samp_s') else None, 2)} | "
                        f"{_fmt(float(row['peak_memory_gb']) if row.get('peak_memory_gb') else None, 3)} | "
                        f"{_fmt(float(row['eval_accuracy']) if row.get('eval_accuracy') else None, 4)} |"
                    )
            lines.append("")
    else:
        lines += [
            "| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy |",
            "| :--- | ---: | ---: | ---: | ---: |",
            "| Baseline (HW4 settings) | TBD | TBD | TBD | TBD |",
            "| Mixed precision | TBD | TBD | TBD | TBD |",
            "| Gradient checkpointing | TBD | TBD | TBD | TBD |",
            "| Gradient accumulation | TBD | TBD | TBD | TBD |",
            "| 8-bit optimiser | TBD | TBD | TBD | TBD |",
            "| Best combination | TBD | TBD | TBD | TBD |",
            "",
        ]

    lines += [
        "## HW4.5 summary configuration (best peak-memory / throughput / wall-clock)",
        "",
        "_Fill after Part B. Do not label this as an HW5 baseline._",
        "",
        "| Item | Value |",
        "| :--- | :--- |",
        "| Best configuration | TBD |",
        "| Peak memory (GB) | TBD |",
        "| Throughput (samp/s) | TBD |",
        "| Wall-clock (s) | TBD |",
        "| GPU-hours for one full fine-tune | TBD |",
        "",
    ]
    out = REPO_ROOT / "METRICS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
