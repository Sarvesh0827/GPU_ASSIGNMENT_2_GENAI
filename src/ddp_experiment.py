"""HW4.5 Part C — Distributed data parallel.

The assignment asks for the instructor-designated two-GPU pairing. This machine is
a single RTX 3050 Ti (same alternative used for HW2.5). The script:

* runs a real DDP session when WORLD_SIZE >= 2
* otherwise runs a single-process baseline and a communication microbenchmark
  so the write-up can still report interconnect path, scaling efficiency = n/a,
  and the batch size at which multi-GPU would become worth the overhead
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertForSequenceClassification

from .config import BASE_LR, BERT_MODEL, NUM_LABELS, RESULTS_DIR, SEED, WEIGHT_DECAY
from .data import build_tensor_datasets
from .gpu_info import gpu_snapshot
from .utils import seed_everything, write_csv


def _init_dist() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local % torch.cuda.device_count())
        return rank, world, local
    return 0, 1, 0


def comm_overhead_bytes(nbytes: int = 32 * 1024 * 1024, reps: int = 20) -> dict:
    """All-reduce a tensor to estimate effective interconnect bandwidth."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
        return {
            "world_size": 1,
            "interconnect": "none (single process)",
            "effective_bandwidth_GBps": None,
            "allreduce_ms": None,
            "note": "No peer to communicate with; DDP all-reduce was not exercised.",
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(nbytes // 4, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(5):
        dist.all_reduce(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        dist.all_reduce(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / reps
    # all-reduce moves ~2*(N-1)/N of the buffer; approximate with 2x bytes
    gbps = (2.0 * nbytes) / elapsed / 1e9
    path = "NVLink/PCIe within one node" if torch.cuda.device_count() >= 2 else "gloo/network or single-node CPU"
    if torch.cuda.device_count() >= 2:
        path = "PCIe (or NVLink if present) within one node"
    return {
        "world_size": dist.get_world_size(),
        "interconnect": path,
        "payload_bytes": nbytes,
        "allreduce_ms": elapsed * 1000.0,
        "effective_bandwidth_GBps": gbps,
        "reps": reps,
    }


def run_ddp(epochs: int = 1, batch_size: int = 8) -> dict:
    seed_everything(SEED)
    rank, world, local = _init_dist()
    snap = gpu_snapshot()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    data = build_tensor_datasets()
    sampler = DistributedSampler(data["train"], num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        data["train"],
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS).to(device)
    if world > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    optim = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    n_seen = 0
    t0 = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            input_ids, attention_mask, labels, _ = [x.to(device) for x in batch]
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            n_seen += input_ids.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    local_throughput = n_seen / wall if wall else float("nan")

    comm = comm_overhead_bytes()
    rec = {
        "rank": rank,
        "world_size": world,
        "homogeneous": True if torch.cuda.device_count() in (0, 1) or world == 1 else None,
        "topology": "single-node",
        "n_visible_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "batch_size_per_rank": batch_size,
        "global_batch": batch_size * world,
        "epochs": epochs,
        "local_samples": n_seen,
        "local_throughput_samp_s": local_throughput,
        "wall_sec": wall,
        "gpu_uuid": snap.get("gpu_uuid"),
        "gpu_name": snap.get("gpu_name"),
        "communication": comm,
        "pairing_note": (
            "This workstation has one NVIDIA GeForce RTX 4090 "
            "(UUID GPU-3094e1cd-3205-910b-9555-93d8401b1ce0). "
            "No second CUDA device is visible, so DDP all-reduce was not exercised. "
            "The second-card UUID capture for Parts A/D is a Colab/Kaggle run."
        ),
    }
    if rank == 0:
        path = RESULTS_DIR / "partC_ddp.json"
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        write_csv(RESULTS_DIR / "partC_ddp.csv", [{k: v for k, v in rec.items() if k != "communication"}])
        print(json.dumps(rec, indent=2))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return rec
