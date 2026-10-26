"""Reusable BERT fine-tune loop used by HW4.a (baseline) and HW4.5 (memory techniques)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

from .config import (
    BASE_BATCH_SIZE,
    BASE_EPOCHS,
    BASE_LR,
    BERT_MODEL,
    CHECKPOINTS_DIR,
    LABEL_NAMES,
    NUM_LABELS,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
)
from .data import build_tensor_datasets, make_loader
from .gpu_info import gpu_snapshot
from .utils import cleanup, peak_memory_gb, reset_peak_memory, seed_everything, write_csv


@dataclass
class TrainConfig:
    batch_size: int = BASE_BATCH_SIZE
    epochs: int = BASE_EPOCHS
    lr: float = BASE_LR
    weight_decay: float = WEIGHT_DECAY
    use_amp: bool = False
    gradient_checkpointing: bool = False
    grad_accum_steps: int = 1
    optimizer_name: str = "adamw"  # adamw | adamw8bit | adafactor
    max_train_steps: int | None = None
    run_name: str = "baseline"
    save: bool = False
    eval_test: bool = True


def _make_optimizer(model, cfg: TrainConfig):
    if cfg.optimizer_name == "adamw8bit":
        try:
            import bitsandbytes as bnb

            print("[train] using bitsandbytes AdamW8bit")
            return bnb.optim.AdamW8bit(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        except Exception as exc:
            print(f"[train] AdamW8bit unavailable ({exc}); falling back to Adafactor")
            from transformers import Adafactor

            return Adafactor(
                model.parameters(),
                lr=cfg.lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
                weight_decay=cfg.weight_decay,
            )
    if cfg.optimizer_name == "adafactor":
        from transformers import Adafactor

        print("[train] using Adafactor (memory-efficient optimiser)")
        return Adafactor(
            model.parameters(),
            lr=cfg.lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=cfg.weight_decay,
        )
    return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _autocast(device: torch.device, enabled: bool):
    # torch.amp.autocast is the current API; fall back for older wheels.
    try:
        return torch.amp.autocast("cuda", enabled=enabled and device.type == "cuda")
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def _grad_scaler(enabled: bool, device: torch.device):
    if not (enabled and device.type == "cuda"):
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=True)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    losses = []
    preds, gold = [], []
    criterion = torch.nn.CrossEntropyLoss()
    for batch in loader:
        input_ids, attention_mask, labels, _ = [x.to(device) for x in batch]
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(out.loss.item())
        pred = out.logits.argmax(dim=-1)
        preds.extend(pred.cpu().tolist())
        gold.extend(labels.cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(gold, preds)) if gold else float("nan"),
        "preds": preds,
        "labels": gold,
    }


def run_training(cfg: TrainConfig) -> dict:
    """Fine-tune BERT and return metrics, epoch log, and optional test predictions."""
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snap = gpu_snapshot()
    data = build_tensor_datasets()

    train_loader = make_loader(data["train"], cfg.batch_size, shuffle=True)
    val_loader = make_loader(data["val"], max(cfg.batch_size, 8), shuffle=False)
    test_loader = make_loader(data["test"], max(cfg.batch_size, 8), shuffle=False)

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model.to(device)

    optimizer = _make_optimizer(model, cfg)
    optimizer_cls = type(optimizer).__name__
    print(f"[train] optimizer={optimizer_cls}  amp={cfg.use_amp}  ckpt={cfg.gradient_checkpointing}  accum={cfg.grad_accum_steps}")
    steps_per_epoch = max(1, (len(train_loader) + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_train_steps is not None:
        total_steps = min(total_steps, cfg.max_train_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.06 * total_steps)),
        num_training_steps=total_steps,
    )
    scaler = _grad_scaler(cfg.use_amp, device)

    cleanup()
    reset_peak_memory()
    wall_start = time.perf_counter()
    epoch_log = []
    optimizer_steps = 0
    stopped_improving_epoch = None
    best_val_loss = float("inf")
    oom = False
    status = "SUCCESS"

    try:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0
            optimizer.zero_grad(set_to_none=True)
            epoch_t0 = time.perf_counter()
            pbar = tqdm(train_loader, desc=f"{cfg.run_name} epoch {epoch}/{cfg.epochs}", leave=False)
            for step, batch in enumerate(pbar, start=1):
                input_ids, attention_mask, labels, _ = [x.to(device) for x in batch]
                with _autocast(device, cfg.use_amp):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out.loss / cfg.grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                running_loss += out.loss.item() * input_ids.size(0)
                n_seen += input_ids.size(0)

                if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if cfg.max_train_steps is not None and optimizer_steps >= cfg.max_train_steps:
                        break
                pbar.set_postfix(loss=float(out.loss.item()))

            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_sec = time.perf_counter() - epoch_t0
            train_loss = running_loss / max(n_seen, 1)
            val_stats = evaluate(model, val_loader, device)
            samples_per_sec = n_seen / epoch_sec if epoch_sec > 0 else float("nan")
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "epoch_seconds": epoch_sec,
                "samples_per_sec": samples_per_sec,
                "n_train_seen": n_seen,
            }
            epoch_log.append(row)
            print(
                f"[{cfg.run_name}] epoch {epoch}: train_loss={train_loss:.4f}  "
                f"val_loss={val_stats['loss']:.4f}  val_acc={val_stats['accuracy']:.4f}  "
                f"{samples_per_sec:.2f} samp/s  {epoch_sec:.1f}s"
            )
            if val_stats["loss"] + 1e-4 < best_val_loss:
                best_val_loss = val_stats["loss"]
            elif stopped_improving_epoch is None:
                stopped_improving_epoch = epoch
            if cfg.max_train_steps is not None and optimizer_steps >= cfg.max_train_steps:
                break
    except torch.cuda.OutOfMemoryError as exc:
        oom = True
        status = "OOM"
        print(f"[{cfg.run_name}] CUDA OOM: {exc}")
        cleanup()

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_sec = time.perf_counter() - wall_start
    peak_gb = peak_memory_gb()

    test_stats = None
    report = None
    cm = None
    if status == "SUCCESS" and cfg.eval_test:
        test_stats = evaluate(model, test_loader, device)
        gold = test_stats["labels"]
        preds = test_stats["preds"]
        cm = confusion_matrix(gold, preds, labels=list(LABEL_NAMES.keys())).tolist()
        prec, rec, f1, support = precision_recall_fscore_support(
            gold, preds, labels=list(LABEL_NAMES.keys()), zero_division=0
        )
        report = {
            LABEL_NAMES[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in LABEL_NAMES
        }
        print(classification_report(gold, preds, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)], digits=4))

    save_dir = None
    if status == "SUCCESS" and cfg.save:
        save_dir = str(CHECKPOINTS_DIR / "bert_sentiment_hw4a")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        from transformers import BertTokenizer

        BertTokenizer.from_pretrained("bert-base-uncased").save_pretrained(save_dir)
        print(f"[train] saved model to {save_dir}")

    result = {
        "run_name": cfg.run_name,
        "status": status,
        "oom": oom,
        "config": asdict(cfg),
        "gpu_name": snap.get("gpu_name"),
        "gpu_uuid": snap.get("gpu_uuid"),
        "vram_bytes": snap.get("vram_bytes"),
        "device": str(device),
        "n_train": len(data["train"]),
        "n_val": len(data["val"]),
        "n_test": len(data["test"]),
        "slice_meta": data["meta"],
        "epoch_log": epoch_log,
        "wall_clock_sec": wall_sec,
        "peak_memory_gb": peak_gb,
        "optimizer_steps": optimizer_steps,
        "optimizer_class": optimizer_cls,
        "effective_batch": cfg.batch_size * cfg.grad_accum_steps,
        "stopped_improving_epoch": stopped_improving_epoch,
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
        "test_loss": None if test_stats is None else test_stats["loss"],
        "test_accuracy": None if test_stats is None else test_stats["accuracy"],
        "per_class": report,
        "confusion_matrix": cm,
        "checkpoint_dir": save_dir,
        "mean_epoch_sec": float(np.mean([r["epoch_seconds"] for r in epoch_log])) if epoch_log else None,
        "mean_throughput_samp_s": float(np.mean([r["samples_per_sec"] for r in epoch_log])) if epoch_log else None,
    }
    out_json = RESULTS_DIR / f"train_{cfg.run_name}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if epoch_log:
        write_csv(RESULTS_DIR / f"train_{cfg.run_name}_epochs.csv", epoch_log)
    print(
        f"[{cfg.run_name}] status={status}  wall={wall_sec:.1f}s  "
        f"peak_mem={peak_gb} GB  throughput={result['mean_throughput_samp_s']} samp/s"
    )

    # Keep predictions for the error gallery
    if test_stats is not None:
        result["_test_preds"] = test_stats["preds"]
        result["_test_labels"] = test_stats["labels"]
        result["_texts"] = data["texts"]
        result["_splits"] = data["splits"]
        result["_model"] = model
        result["_device"] = device
    cleanup()
    return result
