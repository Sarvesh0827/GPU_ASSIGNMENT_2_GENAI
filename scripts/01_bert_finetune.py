"""HW4.a — BERT sentiment classifier on the personal Twitter slice.

Saves the checkpoint used as the HW4.5 baseline and records peak GPU memory
plus wall-clock time.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bert_engine import TrainConfig, run_training
from src.config import BASE_BATCH_SIZE, BASE_EPOCHS, BASE_LR, RESULTS_DIR, SEED, personal_params_block
from src.error_analysis import plot_confusion_matrix, plot_training_curves, worst_class_gallery
from src.gpu_info import capture_system_files
from src.utils import seed_everything

if __name__ == "__main__":
    print(personal_params_block())
    seed_everything(SEED)
    capture_system_files()

    cfg = TrainConfig(
        batch_size=BASE_BATCH_SIZE,
        epochs=BASE_EPOCHS,
        lr=BASE_LR,
        run_name="hw4a_baseline",
        save=True,
        eval_test=True,
    )
    result = run_training(cfg)

    if result["status"] != "SUCCESS":
        print("Baseline batch size OOM'd — retrying at batch_size=4, then 2.")
        for bs in (4, 2, 1):
            cfg.batch_size = bs
            cfg.run_name = f"hw4a_baseline_bs{bs}"
            result = run_training(cfg)
            if result["status"] == "SUCCESS":
                break

    if result["status"] != "SUCCESS":
        raise SystemExit("BERT fine-tune failed even at batch_size=1.")

    plot_training_curves(
        result["epoch_log"],
        title="HW4.a BERT train/val loss and val accuracy",
        outfile="hw4a_training_curves.png",
    )
    if result["confusion_matrix"] is not None:
        plot_confusion_matrix(result["confusion_matrix"])
        gallery = worst_class_gallery(result)
        print(json.dumps({k: gallery[k] for k in ("worst_class", "worst_class_recall", "examples")}, indent=2)[:2000])

    summary = {
        "peak_memory_gb": result["peak_memory_gb"],
        "wall_clock_sec": result["wall_clock_sec"],
        "test_accuracy": result["test_accuracy"],
        "stopped_improving_epoch": result["stopped_improving_epoch"],
        "epoch_log": result["epoch_log"],
        "gpu_uuid": result["gpu_uuid"],
        "gpu_name": result["gpu_name"],
        "batch_size": result["config"]["batch_size"],
        "checkpoint_dir": result["checkpoint_dir"],
        "slice_meta": result["slice_meta"],
    }
    (RESULTS_DIR / "hw4a_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nHW4.a baseline (used by HW4.5):")
    print(json.dumps(summary, indent=2)[:2500])
