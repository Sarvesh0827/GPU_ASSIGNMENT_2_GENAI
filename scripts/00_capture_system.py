"""Capture nvidia-smi -q and a GPU snapshot. Run this first on every card."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import personal_params_block
from src.gpu_info import capture_system_files

if __name__ == "__main__":
    print(personal_params_block())
    capture_system_files()
