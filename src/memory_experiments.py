"""HW4.5 Parts A and B: batch scaling to the memory wall, then one-at-a-time techniques."""

from __future__ import annotations

import json

from .bert_engine import TrainConfig, run_training
from .config import BASE_BATCH_SIZE, BASE_EPOCHS, BASE_LR, FIGURES_DIR, RESULTS_DIR
from .gpu_info import gpu_snapshot
from .utils import write_csv


def part_a_batch_scaling(start_batch: int = 1, max_batch: int = 256, epochs: int = 1) -> list[dict]:
    """Scale batch size by powers of two until OOM. One epoch each to keep wall-clock reasonable."""
    snap = gpu_snapshot()
    rows = []
    largest_ok = None
    bs = start_batch
    while bs <= max_batch:
        cfg = TrainConfig(
            batch_size=bs,
            epochs=epochs,
            lr=BASE_LR,
            use_amp=False,
            gradient_checkpointing=False,
            grad_accum_steps=1,
            optimizer_name="adamw",
            run_name=f"partA_bs{bs}",
            save=False,
            eval_test=False,
        )
        result = run_training(cfg)
        row = {
            "part": "A",
            "batch_size": bs,
            "status": result["status"],
            "throughput_samp_s": result["mean_throughput_samp_s"],
            "peak_memory_gb": result["peak_memory_gb"],
            "time_per_epoch_sec": result["mean_epoch_sec"],
            "wall_clock_sec": result["wall_clock_sec"],
            "gpu_uuid": result["gpu_uuid"],
            "gpu_name": result["gpu_name"],
        }
        rows.append(row)
        if result["status"] != "SUCCESS":
            print(f"[partA] OOM/fail at batch_size={bs}")
            break
        largest_ok = bs
        bs *= 2

    tag = (snap.get("gpu_name") or "cpu").replace(" ", "_")
    write_csv(RESULTS_DIR / f"partA_batch_scaling_{tag}.csv", rows)
    (RESULTS_DIR / f"partA_batch_scaling_{tag}.json").write_text(
        json.dumps({"largest_ok_batch": largest_ok, "rows": rows}, indent=2), encoding="utf-8"
    )
    _plot_throughput(rows, tag)
    print(f"[partA] largest batch that ran: {largest_ok}")
    return rows


def part_b_techniques(baseline_batch: int | None = None, epochs: int = 2) -> list[dict]:
    """Apply AMP, checkpointing, accumulation, 8-bit optimiser independently, then the best combo."""
    snap = gpu_snapshot()
    baseline_batch = baseline_batch or BASE_BATCH_SIZE

    experiments = [
        ("baseline", TrainConfig(batch_size=baseline_batch, epochs=epochs, run_name="partB_baseline", eval_test=True)),
        (
            "mixed_precision",
            TrainConfig(
                batch_size=baseline_batch,
                epochs=epochs,
                use_amp=True,
                run_name="partB_amp",
                eval_test=True,
            ),
        ),
        (
            "gradient_checkpointing",
            TrainConfig(
                batch_size=baseline_batch,
                epochs=epochs,
                gradient_checkpointing=True,
                run_name="partB_ckpt",
                eval_test=True,
            ),
        ),
        (
            "gradient_accumulation",
            TrainConfig(
                batch_size=max(1, baseline_batch // 2),
                epochs=epochs,
                grad_accum_steps=4,
                run_name="partB_accum",
                eval_test=True,
            ),
        ),
        (
            "8bit_optimiser",
            TrainConfig(
                batch_size=baseline_batch,
                epochs=epochs,
                optimizer_name="adamw8bit",
                run_name="partB_8bit",
                eval_test=True,
            ),
        ),
        (
            "best_combination",
            TrainConfig(
                batch_size=baseline_batch,
                epochs=epochs,
                use_amp=True,
                gradient_checkpointing=True,
                optimizer_name="adamw8bit",
                run_name="partB_combo",
                eval_test=True,
            ),
        ),
    ]

    rows = []
    for name, cfg in experiments:
        print("\n" + "=" * 70)
        print(f"Part B technique: {name}")
        print("=" * 70)
        result = run_training(cfg)
        rows.append(
            {
                "technique": name,
                "max_batch": cfg.batch_size,
                "effective_batch": cfg.batch_size * cfg.grad_accum_steps,
                "throughput_samp_s": result["mean_throughput_samp_s"],
                "peak_memory_gb": result["peak_memory_gb"],
                "eval_accuracy": result["test_accuracy"],
                "wall_clock_sec": result["wall_clock_sec"],
                "optimizer_steps": result["optimizer_steps"],
                "status": result["status"],
                "gpu_uuid": result["gpu_uuid"],
                "gpu_name": result["gpu_name"],
            }
        )

    tag = (snap.get("gpu_name") or "cpu").replace(" ", "_")
    write_csv(RESULTS_DIR / f"partB_techniques_{tag}.csv", rows)
    ranking = _rank_memory_per_throughput(rows)
    (RESULTS_DIR / f"partB_ranking_{tag}.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    print("\n[partB] ranking (memory saved per unit throughput lost):")
    for r in ranking["ranked"]:
        print(f"  {r}")
    return rows


def _rank_memory_per_throughput(rows: list[dict]) -> dict:
    base = next((r for r in rows if r["technique"] == "baseline"), None)
    ranked = []
    if base is None or base["peak_memory_gb"] is None:
        return {"ranked": [], "note": "baseline missing"}
    b_mem = float(base["peak_memory_gb"])
    b_thr = float(base["throughput_samp_s"] or 0.0)
    for r in rows:
        if r["technique"] == "baseline" or r["status"] != "SUCCESS":
            continue
        mem = float(r["peak_memory_gb"] or b_mem)
        thr = float(r["throughput_samp_s"] or b_thr)
        mem_saved = b_mem - mem
        thr_lost = b_thr - thr
        if thr_lost <= 1e-9:
            score = float("inf") if mem_saved > 0 else 0.0
            note = "throughput did not decrease (or improved)"
        else:
            score = mem_saved / thr_lost
            note = "GB saved per (samp/s) lost"
        ranked.append(
            {
                "technique": r["technique"],
                "memory_saved_gb": mem_saved,
                "throughput_lost_samp_s": thr_lost,
                "score_gb_per_samp_s_lost": score,
                "note": note,
            }
        )
    ranked.sort(key=lambda x: (x["score_gb_per_samp_s_lost"] == float("inf"), x["score_gb_per_samp_s_lost"]), reverse=True)
    first = ranked[0]["technique"] if ranked else None
    skip_on_32gb = None
    for r in ranked:
        # On a 32 GB card, tiny memory wins that cost a lot of throughput are not worth it.
        if r["memory_saved_gb"] < 0.4 and r["throughput_lost_samp_s"] > 0:
            skip_on_32gb = r["technique"]
            break
    if skip_on_32gb is None and ranked:
        skip_on_32gb = ranked[-1]["technique"]
    return {
        "ranked": ranked,
        "apply_first": first,
        "skip_on_32gb": skip_on_32gb,
        "rationale": (
            "Apply first the technique with the best memory-saved / throughput-lost ratio "
            "(AMP often wins because it may save memory AND raise throughput). "
            "On a 32 GB card skip the weakest memory win that still costs speed — typically "
            "gradient checkpointing, whose whole point is trading compute for memory you already have."
        ),
    }


def _plot_throughput(rows: list[dict], tag: str) -> None:
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r["status"] == "SUCCESS" and r["throughput_samp_s"]]
    if not ok:
        return
    xs = [r["batch_size"] for r in ok]
    ys = [r["throughput_samp_s"] for r in ok]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title(f"Part A — throughput vs batch size\n{tag}")
    ax.grid(True, alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"partA_throughput_{tag}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")
