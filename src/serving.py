"""HW4.a / HW4.5 Part D: HF pipeline vs vLLM throughput and latency.

On Windows or a 4 GB card, vLLM may fail to import or to load. That failure is
recorded as a tooling result (same spirit as HW2.5 lower-precision check).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from .config import (
    CUSTOM_PROMPT,
    RESULTS_DIR,
    SERVE_MAX_NEW_TOKENS,
    SERVE_MODEL,
    SERVE_MODEL_LARGE,
    SERVE_N_PROMPTS,
    SERVE_WARMUP,
)
from .gpu_info import gpu_snapshot
from .utils import cleanup, peak_memory_gb, reset_peak_memory, write_csv

PROMPTS_POOL = [
    CUSTOM_PROMPT,
    "Summarize mixed precision training in one tweet.",
    "Why does gradient checkpointing trade compute for memory?",
    "Give one reason a small batch cannot saturate a GPU.",
    "Explain KV cache growth with context length in one sentence.",
    "What is tokens per second measuring in an LLM server?",
    "Name one reason vLLM is faster than a naive HF generate loop.",
    "Is per-request latency or total throughput more important for chat?",
]


def _prompts(n: int) -> list[str]:
    return [PROMPTS_POOL[i % len(PROMPTS_POOL)] + f" [{i}]" for i in range(n)]


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _pick_model() -> str:
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 12:
            return SERVE_MODEL_LARGE
    return SERVE_MODEL


def benchmark_hf_pipeline(model_id: str, n_prompts: int = SERVE_N_PROMPTS) -> dict:
    from transformers import pipeline

    snap = gpu_snapshot()
    device = 0 if torch.cuda.is_available() else -1
    cleanup()
    reset_peak_memory()
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device=device,
        max_new_tokens=SERVE_MAX_NEW_TOKENS,
    )
    prompts = _prompts(n_prompts)
    for p in prompts[:SERVE_WARMUP]:
        pipe(p, do_sample=False, pad_token_id=pipe.tokenizer.eos_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies_ms = []
    n_new_tokens = 0
    t0 = time.perf_counter()
    for p in prompts:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        s = time.perf_counter()
        out = pipe(p, do_sample=False, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - s) * 1000.0)
        text = out[0]["generated_text"]
        n_new_tokens += len(pipe.tokenizer.encode(text))
    wall = time.perf_counter() - t0
    rec = {
        "engine": "hf_pipeline",
        "batch_size": 1,
        "model_id": model_id,
        "gpu_uuid": snap.get("gpu_uuid"),
        "gpu_name": snap.get("gpu_name"),
        "n_prompts": n_prompts,
        "max_new_tokens": SERVE_MAX_NEW_TOKENS,
        "tokens_per_sec": n_new_tokens / wall if wall else float("nan"),
        "p50_latency_ms": _percentile(latencies_ms, 50),
        "p95_latency_ms": _percentile(latencies_ms, 95),
        "mean_latency_ms": statistics.mean(latencies_ms),
        "wall_sec": wall,
        "peak_memory_gb": peak_memory_gb(),
        "status": "SUCCESS",
        "custom_prompt": CUSTOM_PROMPT,
        "custom_prompt_output": pipe(
            CUSTOM_PROMPT, do_sample=False, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False
        )[0]["generated_text"],
    }
    print(
        f"[serve] HF pipeline batch=1  {rec['tokens_per_sec']:.2f} tok/s  "
        f"p50={rec['p50_latency_ms']:.1f}ms  p95={rec['p95_latency_ms']:.1f}ms"
    )
    del pipe
    cleanup()
    return rec


def benchmark_vllm(model_id: str, batch_size: int, n_prompts: int = SERVE_N_PROMPTS) -> dict:
    snap = gpu_snapshot()
    base = {
        "engine": "vllm",
        "batch_size": batch_size,
        "model_id": model_id,
        "gpu_uuid": snap.get("gpu_uuid"),
        "gpu_name": snap.get("gpu_name"),
        "n_prompts": n_prompts,
        "max_new_tokens": SERVE_MAX_NEW_TOKENS,
        "custom_prompt": CUSTOM_PROMPT,
        "status": "SUCCESS",
    }
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        print(f"[serve] vLLM import failed: {exc}")
        base.update(
            {
                "status": "UNAVAILABLE",
                "error": f"import failed: {exc}",
                "tokens_per_sec": None,
                "p50_latency_ms": None,
                "p95_latency_ms": None,
            }
        )
        return base

    prompts = _prompts(n_prompts)
    cleanup()
    reset_peak_memory()
    try:
        llm = LLM(
            model=model_id,
            dtype="float16" if torch.cuda.is_available() else "float32",
            gpu_memory_utilization=0.7,
            max_model_len=256,
        )
        params = SamplingParams(temperature=0.0, max_tokens=SERVE_MAX_NEW_TOKENS)
        llm.generate(prompts[:SERVE_WARMUP], params)

        latencies_ms = []
        n_new_tokens = 0
        t0 = time.perf_counter()
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            s = time.perf_counter()
            outs = llm.generate(chunk, params)
            latencies_ms.append((time.perf_counter() - s) * 1000.0 / len(chunk))
            for o in outs:
                n_new_tokens += len(o.outputs[0].token_ids)
        wall = time.perf_counter() - t0
        # Also measure a single custom prompt
        custom = llm.generate([CUSTOM_PROMPT], params)[0].outputs[0].text
        rec = {
            **base,
            "tokens_per_sec": n_new_tokens / wall if wall else float("nan"),
            "p50_latency_ms": _percentile(latencies_ms, 50),
            "p95_latency_ms": _percentile(latencies_ms, 95),
            "mean_latency_ms": statistics.mean(latencies_ms),
            "wall_sec": wall,
            "peak_memory_gb": peak_memory_gb(),
            "custom_prompt_output": custom,
        }
        print(
            f"[serve] vLLM batch={batch_size}  {rec['tokens_per_sec']:.2f} tok/s  "
            f"p50={rec['p50_latency_ms']:.1f}ms  p95={rec['p95_latency_ms']:.1f}ms"
        )
        del llm
        cleanup()
        return rec
    except Exception as exc:
        print(f"[serve] vLLM run failed at batch={batch_size}: {exc}")
        cleanup()
        base.update({"status": "ERROR", "error": str(exc)})
        return base


def benchmark_hf_batched(model_id: str, batch_size: int, n_prompts: int = SERVE_N_PROMPTS) -> dict:
    """Batched HF generate — used when vLLM cannot import (typical on Windows)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snap = gpu_snapshot()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cleanup()
    reset_peak_memory()
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    prompts = _prompts(n_prompts)

    def run_chunk(chunk: list[str]):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=SERVE_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out.shape[1] - enc["input_ids"].shape[1]
        return new_tokens * len(chunk)

    for p in prompts[:SERVE_WARMUP]:
        run_chunk([p])
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    n_new_tokens = 0
    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        if device.type == "cuda":
            torch.cuda.synchronize()
        s = time.perf_counter()
        n_new_tokens += run_chunk(chunk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - s) * 1000.0 / len(chunk))
    wall = time.perf_counter() - t0
    rec = {
        "engine": "hf_batched_generate",
        "batch_size": batch_size,
        "model_id": model_id,
        "gpu_uuid": snap.get("gpu_uuid"),
        "gpu_name": snap.get("gpu_name"),
        "n_prompts": n_prompts,
        "max_new_tokens": SERVE_MAX_NEW_TOKENS,
        "tokens_per_sec": n_new_tokens / wall if wall else float("nan"),
        "p50_latency_ms": _percentile(latencies_ms, 50),
        "p95_latency_ms": _percentile(latencies_ms, 95),
        "mean_latency_ms": statistics.mean(latencies_ms) if latencies_ms else None,
        "wall_sec": wall,
        "peak_memory_gb": peak_memory_gb(),
        "status": "SUCCESS",
        "note": "vLLM unavailable on Windows; batched HF generate is the on-box substitute",
        "custom_prompt": CUSTOM_PROMPT,
    }
    print(
        f"[serve] HF batched batch={batch_size}  {rec['tokens_per_sec']:.2f} tok/s  "
        f"p50={rec['p50_latency_ms']:.1f}ms  p95={rec['p95_latency_ms']:.1f}ms"
    )
    del model
    cleanup()
    return rec


def run_serving_suite(model_id: str | None = None) -> list[dict]:
    model_id = model_id or _pick_model()
    print(f"[serve] model={model_id}")
    rows = [benchmark_hf_pipeline(model_id)]
    vllm_ok = True
    for bs in (1, 8, 32):
        rec = benchmark_vllm(model_id, batch_size=bs)
        rows.append(rec)
        if rec.get("status") != "SUCCESS":
            vllm_ok = False
    if not vllm_ok:
        print("[serve] vLLM missing; measuring batched HF generate at batch 1/8/32")
        for bs in (1, 8, 32):
            rows.append(benchmark_hf_batched(model_id, batch_size=bs))
    snap = gpu_snapshot()
    tag = (snap.get("gpu_name") or "cpu").replace(" ", "_")
    write_csv(RESULTS_DIR / f"serving_{tag}.csv", rows)
    (RESULTS_DIR / f"serving_{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def kv_cache_sweep(model_id: str | None = None, batch_size: int = 8, lengths: list[int] | None = None) -> list[dict]:
    """Peak memory vs context length at a fixed batch. Finds longest context before OOM."""
    model_id = model_id or _pick_model()
    lengths = lengths or [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    snap = gpu_snapshot()
    rows = []
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        print(f"[kv] vLLM unavailable ({exc}); using HF generate fallback")
        return _kv_hf_fallback(model_id, batch_size, lengths, snap)

    last_ok = None
    first_fail = None
    for L in lengths:
        cleanup()
        reset_peak_memory()
        rec = {
            "engine": "vllm",
            "model_id": model_id,
            "batch_size": batch_size,
            "context_length": L,
            "gpu_uuid": snap.get("gpu_uuid"),
            "gpu_name": snap.get("gpu_name"),
        }
        try:
            llm = LLM(
                model=model_id,
                dtype="float16",
                gpu_memory_utilization=0.90,
                max_model_len=L,
                max_num_seqs=batch_size,
            )
            prompt = "SJSU " * (L // 2)
            prompts = [prompt[: L - 8] for _ in range(batch_size)]
            llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))
            rec.update({"status": "SUCCESS", "peak_memory_gb": peak_memory_gb()})
            last_ok = L
            print(f"[kv] L={L} OK  peak={rec['peak_memory_gb']:.3f} GB")
            del llm
        except Exception as exc:
            rec.update({"status": "OOM_OR_ERROR", "error": str(exc), "peak_memory_gb": None})
            first_fail = L if first_fail is None else first_fail
            print(f"[kv] L={L} FAIL: {exc}")
            cleanup()
            break
        rows.append(rec)
        cleanup()
    summary = {"largest_success_len": last_ok, "smallest_fail_len": first_fail, "batch_size": batch_size}
    tag = (snap.get("gpu_name") or "cpu").replace(" ", "_")
    write_csv(RESULTS_DIR / f"kv_cache_{tag}.csv", rows)
    (RESULTS_DIR / f"kv_cache_{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows


def _kv_hf_fallback(model_id, batch_size, lengths, snap) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    rows = []
    last_ok = None
    first_fail = None
    for L in lengths:
        cleanup()
        reset_peak_memory()
        rec = {
            "engine": "hf_generate",
            "model_id": model_id,
            "batch_size": batch_size,
            "context_length": L,
            "gpu_uuid": snap.get("gpu_uuid"),
            "gpu_name": snap.get("gpu_name"),
        }
        try:
            text = ("token " * L)[: 4 * L]
            enc = tok([text] * batch_size, return_tensors="pt", truncation=True, max_length=L, padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=8, do_sample=False)
            rec.update({"status": "SUCCESS", "peak_memory_gb": peak_memory_gb()})
            last_ok = L
            print(f"[kv-hf] L={L} OK  peak={rec['peak_memory_gb']}")
        except torch.cuda.OutOfMemoryError as exc:
            rec.update({"status": "OOM", "error": str(exc), "peak_memory_gb": None})
            first_fail = L if first_fail is None else first_fail
            print(f"[kv-hf] L={L} OOM")
            cleanup()
            rows.append(rec)
            break
        rows.append(rec)
    tag = (snap.get("gpu_name") or "cpu").replace(" ", "_")
    write_csv(RESULTS_DIR / f"kv_cache_{tag}.csv", rows)
    (RESULTS_DIR / f"kv_cache_{tag}_summary.json").write_text(
        json.dumps({"largest_success_len": last_ok, "smallest_fail_len": first_fail, "batch_size": batch_size}, indent=2),
        encoding="utf-8",
    )
    del model
    cleanup()
    return rows
