# Written interpretation

UUIDs: 4090 `GPU-3094e1cd-3205-910b-9555-93d8401b1ce0`; T4 `GPU-bfa082ed-9aa0-5fcd-0f6b-582b67447333`.

## HW4.a — serving

On Colab, real vLLM vs HF pipeline batch 1 (27.4 tok/s, p50 1160 ms): vLLM is 157 / 1125 / 3282 tok/s at batch 1 / 8 / 32. Wait (amortized p50 × batch) is 203 / 227 / 312 ms — still faster than HF at every size. Against *vLLM batch 1*, wait first gets worse at **batch 8** while throughput is still rising. Interactive apps should watch that wait, not tokens/s. Windows 4090 could not import vLLM; HF batched generate there showed wait crossing the pipeline baseline at batch 32 (~816 vs 774 ms).

## Part A — memory wall

Largest in-VRAM batch: **256 on 4090**, **128 on T4** (256 OOM). Throughput saturates near **64 on T4** and **128–256 on 4090** — not the same batch. Max-batch ratio 256/128 = 2.00 vs VRAM ratio 24.6/15.4 = 1.60 because ~2.1 GB is a fixed model+optimizer tax and the search is quantized to powers of two.

## Part B — ranking

Apply **8-bit AdamW** first on both cards (~0.41 GB saved, negligible speed loss). On the T4 also apply **AMP** (38.7 → 84.3 samp/s). Do not bother with **gradient checkpointing on a 32 GB card**: it saved no memory at batch 8 and cost 21–27% throughput.

## Part C — DDP

Single 4090, world_size=1, no interconnect. Card 2 is Colab T4, not a second GPU in one job. Multi-GPU is not worth overhead until a second device exists.

## Part D

vLLM numbers exist only on the T4 (Linux). 4090 difference is Windows tooling, not a bad HF timer (pipeline batch-1 41 vs 27 tok/s tracks the cards). Longest HF context at batch 8 on 4090: **4096**. T4 vLLM “0.008 GB” at all lengths is a **measurement flaw** (KV lives in vLLM’s allocator, not `max_memory_allocated`).

## Part E

GPU-hours for one full fine-tune: `51.4/3600 = 0.0143` on the 4090. Chosen config: **AdamW8bit, batch 16, early stop at epoch 2**.
