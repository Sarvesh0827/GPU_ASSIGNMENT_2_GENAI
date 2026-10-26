"""Personal parameters and shared experiment settings for DATA 266 HW4.a / HW4.5.

Standing requirements (Section 0.1):
    SID4  = last four digits of SJSU ID
    SEED  = SID4
    SLICE = SID4 mod 1000
    HP_ID = SID4 mod 6
    CLS_A = SID4 mod 10
    CLS_B = (CLS_A + 1 + ((SID4 // 10) mod 9)) mod 10

HW4.a / HW4.5 explicitly use SEED and SLICE. HP_ID is reported but unused
(Table 0.1: only HW1 supplies an HP_ID mapping). CLS_A / CLS_B are reported
and unused (tweet_eval sentiment has 3 classes, not 10).
"""

from pathlib import Path

SJSU_ID = "019129357"
SID4 = int(SJSU_ID[-4:])  # 9357
SEED = SID4  # 9357
SLICE = SID4 % 1000  # 357
HP_ID = SID4 % 6  # 3  -> Learning-rate-low (unused in this homework)
CLS_A = SID4 % 10  # 7
CLS_B = (CLS_A + 1 + ((SID4 // 10) % 9)) % 10  # 6

HP_ID_ARM = {
    0: "Capacity-low",
    1: "Capacity-high",
    2: "Learning-rate-high",
    3: "Learning-rate-low",
    4: "Schedule-short",
    5: "Schedule-long",
}[HP_ID]

N_ROWS = 5000
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

HF_DATASET = "cardiffnlp/tweet_eval"
HF_CONFIG = "sentiment"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
NUM_LABELS = 3

BERT_MODEL = "bert-base-uncased"
MAX_LENGTH = 128
BASE_BATCH_SIZE = 8
BASE_EPOCHS = 3
BASE_LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

# Small enough to attempt on a 4 GB RTX 3050 Ti; Colab can override.
SERVE_MODEL = "gpt2"
SERVE_MODEL_LARGE = "Qwen/Qwen2.5-0.5B-Instruct"
CUSTOM_PROMPT = (
    "SJSU DATA 266 student 9357: In one sentence, explain why mixed-precision "
    "training can raise batch size on a memory-limited GPU."
)
SERVE_MAX_NEW_TOKENS = 32
SERVE_N_PROMPTS = 64
SERVE_WARMUP = 4

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
SYSTEM_DIR = REPO_ROOT / "system"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DATA_CACHE_DIR = REPO_ROOT / "results" / "data_cache"

for _d in (RESULTS_DIR, FIGURES_DIR, SYSTEM_DIR, CHECKPOINTS_DIR, DATA_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def personal_params_block() -> str:
    row_start = SLICE
    row_end = SLICE + N_ROWS - 1  # inclusive, assuming 5000 rows exist
    return (
        f"SJSU_ID = {SJSU_ID}\n"
        f"SID4    = {SID4}\n"
        f"SEED    = {SEED}\n"
        f"SLICE   = {SLICE}\n"
        f"HP_ID   = {HP_ID} ({HP_ID_ARM}; unused in HW4.a/HW4.5)\n"
        f"CLS_A   = {CLS_A} (unused in this homework)\n"
        f"CLS_B   = {CLS_B} (unused in this homework)\n"
        f"Twitter working subset: consecutive rows [{row_start}, {row_end}] "
        f"({N_ROWS} rows, 70/15/15 split, seed {SEED})\n"
    )
