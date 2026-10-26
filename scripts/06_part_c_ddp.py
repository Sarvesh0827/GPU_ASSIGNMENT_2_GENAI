"""HW4.5 Part C — DDP.

Single GPU:
    python scripts/06_part_c_ddp.py

Two GPUs on one node:
    torchrun --nproc_per_node=2 scripts/06_part_c_ddp.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import personal_params_block
from src.ddp_experiment import run_ddp
from src.gpu_info import capture_system_files

if __name__ == "__main__":
    print(personal_params_block())
    capture_system_files()
    run_ddp(epochs=1, batch_size=8)
