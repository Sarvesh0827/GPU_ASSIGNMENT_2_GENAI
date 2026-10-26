"""HW4.5 Part A — scale batch size by powers of two until OOM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import personal_params_block
from src.gpu_info import capture_system_files
from src.memory_experiments import part_a_batch_scaling

if __name__ == "__main__":
    print(personal_params_block())
    capture_system_files()
    part_a_batch_scaling(start_batch=1, max_batch=2048, epochs=1)
