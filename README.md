# DATA 266 — HW4.a + HW4.5

GPU Assignment II: Training Throughput and Memory Engineering.

Private course repo name (Section 0.3): **`data266-9357`**. GPU-lab tag: **`hw4-5`**.

## Personal parameters (first cell of every notebook)

```
SJSU_ID = 019129357
SID4    = 9357
SEED    = 9357
SLICE   = 357
HP_ID   = 3 (Learning-rate-low; unused in this homework)
CLS_A   = 7
CLS_B   = 6
```

Twitter working subset: **5,000 consecutive `tweet_eval`/`sentiment` train rows beginning at SLICE**, i.e. rows **[357, 5356]**. Split 70 / 15 / 15 with `SEED=9357`.

HP_ID is reported because Section 0.1 requires it at the top of every submission. Table 0.1 says only HW1 has an HP_ID mapping, and HW4.5 says to skip a second hyperparameter model.

## Hardware

| Card | Role | VRAM | UUID |
| :--- | :--- | ---: | :--- |
| NVIDIA GeForce RTX 4090 (this workstation) | Card 1 | 24564 MiB | `GPU-3094e1cd-3205-910b-9555-93d8401b1ce0` |
| Tesla T4 (Google Colab) | Card 2 | 15360 MiB | `GPU-bfa082ed-9aa0-5fcd-0f6b-582b67447333` |

Part C asks for an instructor two-GPU pairing. This PC has one 4090. DDP ran at world_size=1; all-reduce was not exercised. vLLM **does not import on Windows**; the real vLLM table is from the Colab T4.

## Setup

```powershell
cd C:\Users\sarvesh\Downloads\gpu_assignment2_genai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run order (commit after each successful script)

PowerShell: append console output to `RUN_LOG.txt`.

```powershell
function Invoke-Logged($cmd) {
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "`n==== $stamp  $cmd ====`n" | Tee-Object -FilePath RUN_LOG.txt -Append
  Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath RUN_LOG.txt -Append
}

Invoke-Logged "python scripts/00_capture_system.py"
Invoke-Logged "python scripts/01_bert_finetune.py"      # HW4.a baseline memory/time
Invoke-Logged "python scripts/02_custom_inference.py"
Invoke-Logged "python scripts/03_vllm_benchmark.py"     # may fail on Windows; rerun on Colab
Invoke-Logged "python scripts/04_part_a_batch_scaling.py"
Invoke-Logged "python scripts/05_part_b_memory_techniques.py --batch-size 4 --epochs 2"
Invoke-Logged "python scripts/06_part_c_ddp.py"
Invoke-Logged "python scripts/07_part_d_serving_and_kv.py"
python scripts/09_write_metrics.py
```

Then **repeat 00, 04, 07 on Colab** (second card). Download `results/` and `system/` back into this repo. Run:

```powershell
python scripts/08_make_combined_figures.py
python scripts/09_write_metrics.py
```

Incremental commits matter (Section 0.3). Do not squash this into one commit on deadline day.

## Notebook

`HW4_5.ipynb` is the executed-notebook deliverable. Run it top to bottom on each card, or run the scripts and keep the notebook as the narrative + plots.

## Deliverables checklist

- [ ] Repo tagged `hw4-5`
- [ ] Both UUID captures under `system/` (one file per card, `nvidia-smi -q`)
- [ ] All figures under `figures/`
- [ ] `METRICS.md` with Table HW4.1, Table HW4.5.1 per card, HW4.5 summary (**not** labelled as an HW5 baseline)
- [ ] `RUN_LOG.txt`
- [ ] `AI_USE.md` (specific wrong output, how you found it, what you changed)
- [ ] `RESERVATION.md` and `GPU_HOURS.md`
- [ ] Checkpoint command in `METRICS.md` (weights are gitignored; they are ~400 MB)
