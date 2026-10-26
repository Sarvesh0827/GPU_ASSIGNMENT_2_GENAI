"""Reload the saved BERT checkpoint and run custom tweets."""

import json
import sys
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CHECKPOINTS_DIR, LABEL_NAMES, MAX_LENGTH, RESULTS_DIR

EXAMPLES = [
    "I love this class, the GPU labs actually make the architecture click.",
    "This assignment is due tonight and the cluster is down. worst day ever.",
    "The shuttle to campus is at 7:15 as usual.",
    "Can't wait for the 4090 slot — hope mixed precision lets me raise the batch.",
    "nobody understands how vLLM paging works and I have a quiz in the morning",
]


def main() -> None:
    ckpt = CHECKPOINTS_DIR / "bert_sentiment_hw4a"
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt}. Run scripts/01_bert_finetune.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(ckpt)
    model = BertForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()
    enc = tokenizer(EXAMPLES, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        pred = logits.argmax(dim=-1).cpu().tolist()
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    rows = []
    for text, y, p in zip(EXAMPLES, pred, probs):
        rows.append(
            {
                "text": text,
                "pred_label": LABEL_NAMES[y],
                "probs": {LABEL_NAMES[i]: round(p[i], 4) for i in LABEL_NAMES},
            }
        )
        print(f"{LABEL_NAMES[y]:8s}  {text}")
    (RESULTS_DIR / "custom_inference.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
