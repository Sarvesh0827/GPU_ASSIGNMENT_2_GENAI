"""HW4.5 Part D — re-run serving on this card and sweep KV/context length."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import personal_params_block
from src.gpu_info import capture_system_files
from src.serving import kv_cache_sweep, run_serving_suite

if __name__ == "__main__":
    print(personal_params_block())
    capture_system_files()
    run_serving_suite()
    kv_cache_sweep(batch_size=8)
