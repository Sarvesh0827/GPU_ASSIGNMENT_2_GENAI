# HW4.5 — Training Throughput and Memory Engineering

This file is the HW4.5 summary. It is **not** an HW5 baseline.

Two UUID-labelled cards:

| Card | Name | VRAM | UUID | Driver |
| :--- | :--- | ---: | :--- | :--- |
| 1 | NVIDIA GeForce RTX 4090 | 24564 MiB | `GPU-3094e1cd-3205-910b-9555-93d8401b1ce0` | 610.60 / CUDA 13.3 / 450 W |
| 2 | Tesla T4 (Colab) | 15360 MiB | `GPU-bfa082ed-9aa0-5fcd-0f6b-582b67447333` | 580.82.07 / 70 W |

`nvidia-smi -q` captures: `system/nvidia_smi_q_NVIDIA_GeForce_RTX_4090_3094e1cd-320.txt` and `system/nvidia_smi_q_Tesla_T4_bfa082ed-9aa.txt`.

## Personal parameters (Section 0.1)

```
SJSU_ID = 019129357
SID4    = 9357
SEED    = 9357
SLICE   = 357
HP_ID   = 3 (Learning-rate-low; unused in HW4.a/HW4.5)
CLS_A   = 7
CLS_B   = 6
Twitter working subset: consecutive rows [357, 5356]
  (5000 rows of cardiffnlp/tweet_eval sentiment, 70/15/15, seed 9357)
  train=3499  val=750  test=751
```

Train command: `python scripts/01_bert_finetune.py`

## HW4.a baseline (kept for HW4.5)

Same recipe on both cards: `bert-base-uncased`, batch 8, 3 epochs, LR 2e-5, AdamW.

| Item | RTX 4090 | Tesla T4 |
| :--- | ---: | ---: |
| Peak GPU memory | **2.088 GB** | **2.086 GB** |
| Wall-clock | **51.4 s** | **283.0 s** |
| Test accuracy | 0.6991 | 0.6977 |
| Val stopped improving | epoch 2 | epoch 2 |
| Cost of epoch 3 | +16.2 s, val loss 0.73→0.86 | +90.0 s, val loss 0.72→0.90 |

Peak memory matches (same model/batch). Wall-clock is ~5.5× longer on the T4. Validation stopped improving at **epoch 2** on both; epoch 3 overfit.

### 4090 per-epoch log

| Epoch | Train loss | Val loss | Val acc | Epoch s | samp/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8498 | 0.7174 | 0.6973 | 16.8 | 208.64 |
| 2 | 0.5460 | 0.7326 | 0.7067 | 16.2 | 215.76 |
| 3 | 0.3150 | 0.8584 | 0.7053 | 16.2 | 215.91 |

### T4 per-epoch log

| Epoch | Train loss | Val loss | Val acc | Epoch s | samp/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8487 | 0.6847 | 0.7040 | 85.9 | 40.74 |
| 2 | 0.5308 | 0.7175 | 0.6987 | 90.2 | 38.78 |
| 3 | 0.3070 | 0.9010 | 0.7080 | 90.0 | 38.87 |

**Worst class (4090 test): negative** (recall 0.5385). Confusion matrix `[[63,45,9],[28,254,61],[8,75,208]]`. Five misses: `results/error_gallery.md`.

## Table HW4.1 — Serving comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`. 64 prompts × 32 new tokens.

### Card 1 — RTX 4090 (Windows; `import vllm` failed)

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |
| :--- | ---: | ---: | ---: |
| HF pipeline, batch 1 | 41.12 | 773.8 | 800.7 |
| vLLM, batch 1 | UNAVAILABLE | — | — |
| vLLM, batch 8 | UNAVAILABLE | — | — |
| vLLM, batch 32 | UNAVAILABLE | — | — |
| HF batched generate, batch 1 | 41.38 | 769.0 | 806.8 |
| HF batched generate, batch 8 | 328.54 | 97.3 amort. / ~778 wait | 100.1 |
| HF batched generate, batch 32 | 1254.66 | 25.5 amort. / ~816 wait | 25.6 |

### Card 2 — Tesla T4 (Linux Colab; real vLLM)

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |
| :--- | ---: | ---: | ---: |
| HF pipeline, batch 1 | 27.41 | 1159.8 | 1221.6 |
| vLLM, batch 1 | **157.26** | **203.2** | 208.2 |
| vLLM, batch 8 | **1125.47** | 28.4 amort. / **~227 wait** | 29.4 |
| vLLM, batch 32 | **3281.75** | 9.8 amort. / **~312 wait** | 9.8 |

vLLM wait = amortized p50 × batch size (the user waits for the whole batch).

**Interactive vs throughput (T4 vLLM, vs HF pipeline baseline).** Throughput keeps rising 157 → 1125 → 3282 tok/s. **Wait stays better than the HF baseline at every tested batch** (203 / 227 / 312 ms vs 1160 ms). Relative to *vLLM batch 1*, wait first exceeds the 203 ms baseline at **batch 8** (~227 ms) while tokens/s is still climbing. For an interactive app, **per-request wait** is the number that matters, not tokens/s.

4090 vs T4 difference is **hardware + tooling**: Windows has no vLLM; Colab Linux does. HF pipeline batch-1 on the 4090 (41 tok/s) vs T4 (27 tok/s) tracks the card, not a broken timer.

## Part A — Batch scaling to the memory wall

Unchanged HW4.a fine-tune, 1 epoch, batch sizes powers of two. Plot: `figures/partA_throughput_both_cards.png`.

| Batch | 4090 samp/s | 4090 GB | 4090 status | T4 samp/s | T4 GB | T4 status |
| ---: | ---: | ---: | :--- | ---: | ---: | :--- |
| 1 | 29.22 | 2.062 | SUCCESS | 13.33 | 2.061 | SUCCESS |
| 2 | 60.74 | 2.062 | SUCCESS | 22.85 | 2.061 | SUCCESS |
| 4 | 119.25 | 2.067 | SUCCESS | 31.97 | 2.066 | SUCCESS |
| 8 | 223.40 | 2.092 | SUCCESS | 38.86 | 2.090 | SUCCESS |
| 16 | 319.73 | 2.483 | SUCCESS | 44.33 | 2.482 | SUCCESS |
| 32 | 427.30 | 3.743 | SUCCESS | 48.66 | 3.736 | SUCCESS |
| 64 | 477.88 | 6.154 | SUCCESS | 49.77 | 6.127 | SUCCESS |
| 128 | 502.10 | 11.045 | SUCCESS | **50.47** | 10.971 | SUCCESS (largest T4) |
| 256 | **524.33** | 20.703 | SUCCESS (largest in 24 GB) | — | 14.16 | **OOM** |
| 512 | 15.93 | 40.03 | WDDM overcommit | — | — | not reached |

**Largest batch that runs:** 4090 = **256** (in dedicated VRAM); T4 = **128** (256 CUDA OOM).

**Throughput knee:** not the same batch on both cards. T4 is flat after **64** (49.8 → 50.5 samp/s, +1.4%). 4090 is still rising slowly through **128–256** (502 → 524, +4%). Small matrices/batches never plateau because launch overhead and underfilled SMs dominate; once the GPU is busy, extra batch mostly buys memory pressure.

**Max-batch ratio vs VRAM ratio**

```
max batch  256 / 128 = 2.00
VRAM       24564 / 15360 MiB = 1.60
```

These are not equal. About **2.06 GB** is a fixed BERT + AdamW footprint on both cards (see batch 1–8). Usable activation memory is therefore (24.6 − 2.1) / (15.4 − 2.1) ≈ **1.69**, still not 2.00. The rest of the gap is the **power-of-two search**: T4 dies at 256, 4090 still fits 256, so the reported max-batch ratio snaps to 2× even though VRAM is only 1.6×. WDDM on the 4090 also refused to throw OOM at 512 and overcommitted instead.

## Table HW4.5.1 — Card 1 (RTX 4090)

| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy |
| :--- | ---: | ---: | ---: | ---: |
| Baseline (HW4 settings) | 8 | 216.54 | 2.088 | 0.7124 |
| Mixed precision | 8 | 187.68 | 2.500 | 0.7031 |
| Gradient checkpointing | 8 | 157.71 | 2.499 | 0.7124 |
| Gradient accumulation | 4 (eff. 16) | 160.95 | 2.480 | 0.7004 |
| 8-bit optimiser | 8 | 212.84 | **1.674** | 0.7057 |
| Best combination | 8 | 111.78 | **1.469** | 0.7031 |

Apply first on 4090: **8-bit AdamW**. Skip on a 32 GB card: **gradient checkpointing**.

## Table HW4.5.1 — Card 2 (Tesla T4)

| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy |
| :--- | ---: | ---: | ---: | ---: |
| Baseline (HW4 settings) | 8 | 38.70 | 2.086 | 0.7084 |
| Mixed precision | 8 | **84.34** | 2.501 | 0.7004 |
| Gradient checkpointing | 8 | 30.40 | 2.507 | 0.7084 |
| Gradient accumulation | 4 (eff. 16) | 40.51 | 2.479 | 0.7017 |
| 8-bit optimiser | 8 | 39.30 | **1.676** | **0.7177** |
| Best combination | 8 | 66.22 | **1.470** | 0.6991 |

On the T4, **AMP more than doubled throughput** (38.7 → 84.3 samp/s) because the card actually uses Tensor Cores; it did **not** save memory at batch 8 (2.09 → 2.50 GB). **8-bit** still saves ~0.41 GB with no speed loss. Combination: 1.47 GB and 66 samp/s (AMP recovers some of the checkpointing tax). Skip on 32 GB: checkpointing (only technique that lost speed and saved no memory).

## Part C — DDP

One GPU on each machine (`n_visible_cuda_devices=1`). Homogeneous, single-node, no interconnect, all-reduce not exercised. 4090 DDP-style throughput 237.5 samp/s at batch 8. Multi-GPU is not worth the overhead until a second device exists. Instructor pairing was not available; card 2 is Colab T4 as the listed Colab alternative.

## Part D — Serving + KV cache

vLLM ran only on the T4. 4090 used HF generate. Difference is **tooling (Windows vs Linux) plus hardware**, evidenced by: (1) `No module named 'vllm'` on 4090, (2) successful vLLM import on Colab, (3) HF pipeline batch-1 scaling 41 vs 27 tok/s with the cards.

KV at batch 8, `Qwen/Qwen2.5-0.5B-Instruct`:

| Context | 4090 HF generate GB | T4 vLLM GB (torch.allocated) |
| ---: | ---: | ---: |
| 64 | 0.962 | 0.008 |
| 1024 | 1.566 | 0.008 |
| 4096 | 8.690 | 0.008 |
| 8192 | 30.49 (past 24 GB) | 0.008, status SUCCESS |

**Longest in-VRAM context at batch 8:** 4090 HF = **4096**. T4 vLLM reported SUCCESS through **8192**, but peak memory stuck at 0.008 GB — **measurement flaw**: vLLM’s paged KV cache is not in `torch.cuda.max_memory_allocated()`. Do not treat 0.008 GB as the true KV footprint.

## Part E — GPU-hours and chosen config

One full fine-tune (3 epochs, batch 8, AdamW):

```
4090:  51.4 s / 3600 = 0.0143 GPU-hours
T4:   283.0 s / 3600 = 0.0786 GPU-hours
```

**Config I would use on this lab’s hardware (4090):** `AdamW8bit`, batch **16**, stop when val loss stops improving (epoch 2 here). 8-bit is the only 4090 knob that saved memory without wrecking speed; batch 16 is past the latency-bound region (320 samp/s, 2.48 GB) and far from the 24 GB wall. On a T4-only Colab box I would add **AMP** because it doubled T4 throughput.

### HW4.5 summary (not an HW5 baseline)

| Item | Value |
| :--- | :--- |
| Best peak-memory | AMP+ckpt+8bit, batch 8, **1.47 GB** (both cards) |
| Best in-VRAM throughput | 4090 FP32 batch **256**, **524 samp/s**, 20.7 GB |
| Best wall-clock quality run | 4090 HW4.a / 8-bit batch 8, **~34–51 s** |
| GPU-hours, one full fine-tune | **0.0143** on 4090; 0.0786 on T4 |

## Checkpoint command

Weights are gitignored (~400 MB). The 4090 HW4.a run that produced the numbers above:

```
python scripts/01_bert_finetune.py
```

Saved locally at `checkpoints/bert_sentiment_hw4a/`.

