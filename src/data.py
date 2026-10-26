"""Twitter subset construction for HW4.a / HW4.5.

Working subset: up to 5,000 consecutive tweet_eval sentiment rows beginning at SLICE.
Split 70 / 15 / 15 with SEED. Exact row range is printed and saved.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

from .config import (
    DATA_CACHE_DIR,
    HF_CONFIG,
    HF_DATASET,
    LABEL_NAMES,
    MAX_LENGTH,
    N_ROWS,
    RESULTS_DIR,
    SEED,
    SLICE,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)
from .utils import seed_everything


def load_twitter_subset() -> dict:
    """Download tweet_eval sentiment and cut the consecutive SLICE window."""
    from datasets import load_dataset

    raw = load_dataset(HF_DATASET, HF_CONFIG, split="train")
    n_available = len(raw)
    start = SLICE
    if start >= n_available:
        raise ValueError(f"SLICE={start} is past the end of {HF_DATASET}/{HF_CONFIG} ({n_available} rows)")
    end = min(start + N_ROWS, n_available)
    n_used = end - start
    texts = list(raw["text"][start:end])
    labels = list(raw["label"][start:end])

    meta = {
        "dataset": f"{HF_DATASET}/{HF_CONFIG}",
        "full_train_rows": n_available,
        "slice_start": start,
        "slice_end_exclusive": end,
        "slice_end_inclusive": end - 1,
        "n_rows_requested": N_ROWS,
        "n_rows_used": n_used,
        "label_names": LABEL_NAMES,
        "note": (
            "Used all remaining rows" if n_used < N_ROWS else f"Used {N_ROWS} consecutive rows beginning at SLICE"
        ),
    }
    (RESULTS_DIR / "slice_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[data] {HF_DATASET}/{HF_CONFIG} train has {n_available} rows; "
        f"working subset is rows [{start}, {end - 1}] ({n_used} rows)"
    )
    return {"texts": texts, "labels": labels, "meta": meta}


def split_indices(labels: list[int]) -> dict:
    """70% train, 15% val, 15% test. Stratified, seeded."""
    seed_everything(SEED)
    idx = np.arange(len(labels))
    y = np.array(labels)
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(1.0 - TRAIN_FRAC),
        random_state=SEED,
        stratify=y,
    )
    # temp is 30%; split in half -> 15% / 15%
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        random_state=SEED,
        stratify=y[temp_idx],
    )
    print(
        f"[data] split (seed={SEED}): "
        f"train={len(train_idx)} ({TRAIN_FRAC:.0%})  "
        f"val={len(val_idx)} ({VAL_FRAC:.0%})  "
        f"test={len(test_idx)} ({TEST_FRAC:.0%})"
    )
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def tokenize_texts(texts: list[str], tokenizer: BertTokenizer | None = None) -> dict:
    tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "tokenizer": tokenizer,
    }


def build_tensor_datasets(force_rebuild: bool = False) -> dict:
    """Create TensorDatasets for train/val/test and cache them on disk."""
    cache_path = DATA_CACHE_DIR / "tensors.pt"
    meta_path = DATA_CACHE_DIR / "split_indices.json"
    if cache_path.exists() and meta_path.exists() and not force_rebuild:
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[data] loaded cached tensors from {cache_path}")
        return blob

    subset = load_twitter_subset()
    splits = split_indices(subset["labels"])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized = tokenize_texts(subset["texts"], tokenizer)
    labels = torch.tensor(subset["labels"], dtype=torch.long)

    def make_ds(indices: list[int]) -> TensorDataset:
        idx = torch.tensor(indices, dtype=torch.long)
        return TensorDataset(
            tokenized["input_ids"][idx],
            tokenized["attention_mask"][idx],
            labels[idx],
            idx,
        )

    blob = {
        "train": make_ds(splits["train"]),
        "val": make_ds(splits["val"]),
        "test": make_ds(splits["test"]),
        "texts": subset["texts"],
        "labels": subset["labels"],
        "splits": splits,
        "meta": subset["meta"],
        "max_length": MAX_LENGTH,
    }
    torch.save(
        {
            "train": blob["train"],
            "val": blob["val"],
            "test": blob["test"],
            "texts": blob["texts"],
            "labels": blob["labels"],
            "splits": blob["splits"],
            "meta": blob["meta"],
            "max_length": blob["max_length"],
        },
        cache_path,
    )
    meta_path.write_text(json.dumps({"splits": splits, "meta": subset["meta"]}, indent=2), encoding="utf-8")
    print(f"[data] cached tensors at {cache_path}")
    return blob


def make_loader(dataset: TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
