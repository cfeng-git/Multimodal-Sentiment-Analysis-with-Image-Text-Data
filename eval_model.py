# eval_model.py
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

from utils.config import load_config, merge_config_and_args, validate_config
from utils.data_utils import build_add_label_id, build_compute_metrics, load_text_from_path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Config path or name under configs/")
    p.add_argument("--checkpoint", help="Checkpoint dir to evaluate (defaults to latest checkpoint-* in output_dir)")
    p.add_argument("--batch_size", type=int, help="Override eval batch size")
    return p.parse_args()


def resolve_checkpoint_dir(output_dir: Path, override: str | None) -> Path:
    if override:
        ckpt = Path(override).expanduser().resolve()
        if not ckpt.is_dir():
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt}")
        return ckpt
    # pick newest checkpoint-XXX
    candidates = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]) if "-" in p.name else -1,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories under {output_dir}")
    return candidates[-1]


def load_label_map(output_dir: Path):
    lm_path = output_dir / "label_map.json"
    if not lm_path.exists():
        raise FileNotFoundError(f"label_map.json not found in {output_dir}")
    with open(lm_path, "r") as f:
        data = json.load(f)
    return data["label2id"], data["id2label"]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    cfg = merge_config_and_args(cfg, args)

    # core config values
    data_cfg = cfg["data"]
    test_csv = data_cfg.get("test_csv")
    if not test_csv:
        raise ValueError("Config must specify data.test_csv for evaluation.")
    label_column = data_cfg.get("label_column")
    target = data_cfg.get("target", "text")
    model_name = cfg["model"].get("model_name")
    max_length = cfg["model"].get("max_length", 128)
    batch_size = args.batch_size or cfg["training"].get("batch_size", 8)
    seed = cfg["training"].get("seed", 42)
    output_dir = Path(cfg.get("logging", {}).get("output_dir", PROJECT_ROOT / "outputs"))

    set_seed(seed)

    ckpt_dir = resolve_checkpoint_dir(output_dir, args.checkpoint)
    label2id, id2label = load_label_map(output_dir)

    # load data
    ds = load_dataset("csv", data_files={"test": test_csv})

    if label_column is None:
        target_map = {"text": "text_label", "image": "image_label", "combined": "combined_label"}
        label_column = target_map.get(target, target)
    if label_column not in ds["test"].column_names:
        raise KeyError(f"Label column '{label_column}' not found in CSV columns: {ds['test'].column_names}")
    if "text_path" not in ds["test"].column_names:
        raise KeyError("CSV must contain a 'text_path' column with file paths to text files.")

    ds = ds.map(load_text_from_path)
    text_source_col = "text"
    ds = ds.map(build_add_label_id(label_column, label2id))
    tok = AutoTokenizer.from_pretrained(ckpt_dir if (ckpt_dir / "tokenizer_config.json").exists() else model_name)

    def preprocess(batch):
        return tok(batch[text_source_col], truncation=True, padding="max_length", max_length=max_length)

    ds = ds.map(preprocess, batched=True)
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in ds["test"].features:
        keep.add("token_type_ids")
    ds = ds.remove_columns([c for c in ds["test"].column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    eval_args = TrainingArguments(
        output_dir=str(output_dir / "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=ds["test"],
        tokenizer=tok,
        compute_metrics=build_compute_metrics(average="macro"),
    )

    metrics = trainer.evaluate()
    print("Evaluation metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
