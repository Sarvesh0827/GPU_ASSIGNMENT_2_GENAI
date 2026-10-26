"""After both cards have Part A CSVs, plot throughput vs batch size on one figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, RESULTS_DIR


def main() -> None:
    files = sorted(RESULTS_DIR.glob("partA_batch_scaling_*.csv"))
    if not files:
        raise SystemExit("No partA_batch_scaling_*.csv files yet. Run Part A on each card first.")
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in files:
        df = pd.read_csv(f)
        ok = df[df["status"] == "SUCCESS"]
        label = f.stem.replace("partA_batch_scaling_", "")
        ax.plot(ok["batch_size"], ok["throughput_samp_s"], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title("Part A — throughput vs batch size (all cards)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "partA_throughput_both_cards.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
