"""Confusion-matrix dump and five misclassified examples from the worst class."""

from __future__ import annotations

import json

from .config import LABEL_NAMES, RESULTS_DIR, FIGURES_DIR


def worst_class_gallery(result: dict, k: int = 5) -> dict:
    per_class = result.get("per_class") or {}
    if not per_class:
        raise ValueError("run_training must be called with eval_test=True")

    # Worst class = lowest recall (the class the model most often misses)
    worst_name = min(per_class, key=lambda c: per_class[c]["recall"])
    worst_id = {v: k_ for k_, v in LABEL_NAMES.items()}[worst_name]

    texts = result["_texts"]
    splits = result["_splits"]
    test_idx = splits["test"]
    preds = result["_test_preds"]
    gold = result["_test_labels"]

    misses = []
    for local_i, (p, y) in enumerate(zip(preds, gold)):
        if y == worst_id and p != y:
            global_i = test_idx[local_i]
            misses.append(
                {
                    "row_in_working_subset": int(global_i),
                    "text": texts[global_i],
                    "true_label": LABEL_NAMES[y],
                    "pred_label": LABEL_NAMES[p],
                    "diagnosis": _diagnose(texts[global_i], LABEL_NAMES[y], LABEL_NAMES[p]),
                }
            )

    gallery = misses[:k]
    payload = {
        "worst_class": worst_name,
        "worst_class_id": worst_id,
        "worst_class_recall": per_class[worst_name]["recall"],
        "worst_class_precision": per_class[worst_name]["precision"],
        "n_misclassified_in_worst": len(misses),
        "examples": gallery,
        "confusion_matrix": result.get("confusion_matrix"),
        "label_order": [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        "per_class": per_class,
    }
    path = RESULTS_DIR / "error_gallery.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = RESULTS_DIR / "error_gallery.md"
    md.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"[error] worst class = {worst_name} (recall={per_class[worst_name]['recall']:.3f})")
    print(f"[error] wrote {path} and {md}")
    return payload


def _diagnose(text: str, true_label: str, pred_label: str) -> str:
    lowered = text.lower()
    if true_label == "neutral" and pred_label == "positive":
        return "Neutral wording with a mildly upbeat token was pulled toward positive."
    if true_label == "neutral" and pred_label == "negative":
        return "Complaint-adjacent or negated wording made a neutral tweet look negative."
    if true_label == "positive" and pred_label == "negative":
        if any(w in lowered for w in ["not", "n't", "never", "no "]):
            return "Negation flipped a positive statement; bag-of-cues BERT often misses this."
        return "Positive intent is implicit or sarcastic; surface cues look negative."
    if true_label == "negative" and pred_label == "positive":
        if any(w in lowered for w in ["love", "great", "good", "happy", "lol"]):
            return "Positive lexical items (irony/sarcasm) overrode the negative intent."
        return "Negative stance is indirect; the model followed a more positive local phrase."
    if true_label == "positive" and pred_label == "neutral":
        return "Weak/positive-but-factual phrasing looked like a reporting sentence."
    if true_label == "negative" and pred_label == "neutral":
        return "The complaint is understated; without strong sentiment words it scored as news-like neutral."
    return f"True {true_label} predicted {pred_label}; mixed or weak sentiment cues."


def _to_markdown(payload: dict) -> str:
    lines = [
        f"# Error gallery — worst class: **{payload['worst_class']}**",
        "",
        f"Recall {payload['worst_class_recall']:.4f}, precision {payload['worst_class_precision']:.4f}, "
        f"{payload['n_misclassified_in_worst']} misses on the test split.",
        "",
        "Label order for the confusion matrix: " + ", ".join(payload["label_order"]),
        "",
        "Confusion matrix (rows = true, cols = predicted):",
        "",
        "```",
        str(payload["confusion_matrix"]),
        "```",
        "",
        "## Five misclassified examples",
        "",
    ]
    for i, ex in enumerate(payload["examples"], start=1):
        lines.append(f"### {i}. true={ex['true_label']}  pred={ex['pred_label']}  (row {ex['row_in_working_subset']})")
        lines.append("")
        lines.append(f"> {ex['text']}")
        lines.append("")
        lines.append(f"Diagnosis: {ex['diagnosis']}")
        lines.append("")
    return "\n".join(lines)


def plot_training_curves(epoch_log: list[dict], title: str, outfile: str) -> None:
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in epoch_log]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, [r["train_loss"] for r in epoch_log], marker="o", label="train loss")
    ax1.plot(epochs, [r["val_loss"] for r in epoch_log], marker="o", label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [r["val_accuracy"] for r in epoch_log], marker="s", linestyle="--", color="green", label="val acc")
    ax2.set_ylabel("Validation accuracy")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title(title)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / outfile
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_confusion_matrix(cm: list[list[int]], outfile: str = "confusion_matrix.png") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("HW4.a BERT confusion matrix (test)")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / outfile
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")
