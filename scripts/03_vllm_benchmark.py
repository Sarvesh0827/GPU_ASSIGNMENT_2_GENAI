"""HW4.a Table HW4.1 — HF pipeline vs vLLM at batch 1, 8, 32."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import personal_params_block
from src.gpu_info import capture_system_files
from src.serving import run_serving_suite

if __name__ == "__main__":
    print(personal_params_block())
    capture_system_files()
    rows = run_serving_suite()
    print(json.dumps(rows, indent=2)[:4000])
