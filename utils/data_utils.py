from __future__ import annotations

import numpy as np


def load_text_from_path(example: dict, path_key: str = "text_path", out_key: str = "text") -> dict:
    """Read raw text from a file path in the example and store it under out_key.

    Expects example[path_key] to be a valid UTF-8 text file path.
    """
    with open(example[path_key], "r", encoding="utf-8") as f:
        example[out_key] = f.read().strip()
    return example


def build_add_label_id(label_column: str, label2id: dict):
    """Return a function that maps example[label_column] to integer id under 'labels'."""

    def _fn(example: dict) -> dict:
        example["labels"] = label2id[example[label_column]]
        return example

    return _fn


def build_compute_metrics(average: str = "macro"):
    """Return a compute_metrics callable compatible with HF Trainer.

    Loads accuracy and F1 from `evaluate` and returns a function that
    computes both metrics given the prediction output `p`.
    """
    import evaluate

    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def _compute(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            **acc.compute(predictions=preds, references=p.label_ids),
            **f1.compute(predictions=preds, references=p.label_ids, average=average),
        }

    return _compute

