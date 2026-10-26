# GPU-hour record — HW4.5

```
GPU-hours = wall-clock seconds / 3600
```

One full fine-tune (HW4.a, 3 epochs, batch 8):

```
4090:  51.4 s / 3600 = 0.0143 GPU-hours
T4:   283.0 s / 3600 = 0.0786 GPU-hours
```

| Run | GPU | Wall-clock (s) | GPU-hours |
| :--- | :--- | ---: | ---: |
| HW4.a baseline | 4090 | 51.4 | 0.0143 |
| Part A 1–256 + 512 | 4090 | ~635 | 0.176 |
| Part B six techniques | 4090 | ~290 | 0.081 |
| Part C DDP | 4090 | 14.7 | 0.004 |
| Part D serving + KV | 4090 | ~218 | 0.061 |
| HW4.a baseline | T4 Colab | 283.0 | 0.0786 |
| Part A 1–128 + OOM 256 | T4 Colab | ~952 | 0.264 |
| Part B six techniques | T4 Colab | ~1017 | 0.283 |
| vLLM + KV | T4 Colab | (inside 07) | included above / same session |
| **4090 total** | | **~1210** | **~0.34** |
| **T4 total (approx.)** | | **~2250** | **~0.63** |

No lab workstation reservation. Card 1 = owned 4090. Card 2 = Colab free-tier T4.
