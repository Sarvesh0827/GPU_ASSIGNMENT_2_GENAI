"""HW4.5 Part B — AMP, checkpointing, accumulation, 8-bit optimiser, then combo."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BASE_BATCH_SIZE, personal_params_block
from src.gpu_info import capture_system_files
from src.memory_experiments import part_b_techniques

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BASE_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    print(personal_params_block())
    capture_system_files()
    part_b_techniques(baseline_batch=args.batch_size, epochs=args.epochs)
