import argparse
import os
from pathlib import Path
import json
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from utils.config import load_config, validate_config, merge_config_and_args
from utils.data_utils import (
    load_text_from_path,
    build_add_label_id,
    build_compute_metrics,
)



PROJECT_ROOT = Path(__file__).resolve().parent





def parse_args():
    parser = argparse.ArgumentParser()

    # The only argument we require is the config file.
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config name or path. If a bare name is given, searches 'configs/<name>.yaml'."
    )

    # Optional overrides (can add more if needed)
    parser.add_argument("--learning_rate", type=float, help="Override LR")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--num_epochs", type=int, help="Override epochs")

    return parser.parse_args()





def main():

    # load configuration file as a dictionary
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    config = merge_config_and_args(config, args)

    print("Loaded config:", config.get("_config_path"))
    print("Final Config:")
    print(config)

    

    # Extracting values from config:

    # get parameters related to training
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    seed = config["training"].get("seed", 42)

    # get parameters related to model
    model_section = config["model"]
    model_name = model_section.get("model_name")  # text-only
    text_model = model_section.get("text_model")  # multimodal
    vision_model = model_section.get("vision_model")  # multimodal
    max_length = model_section.get("max_length", 128)

    # get parameters related to data
    data_cfg = config["data"]
    train_csv = data_cfg["train_csv"]
    val_csv = data_cfg["val_csv"]
    test_csv = data_cfg.get("test_csv")
    label_column = data_cfg.get("label_column")  # optional
    target = data_cfg.get("target", "text")  # only used if label_column is not provided

    # get parameters related logging: 
    out_cfg = config.get("logging", {})
    output_dir = out_cfg.get("output_dir", str((PROJECT_ROOT / "outputs" / "checkpoints" / "text").resolve()))
    
    # Branch: text-only vs multimodal. Multimodal path can use a custom Dataset.
    is_multimodal = text_model is not None and vision_model is not None and model_name is None

    if is_multimodal:
        raise NotImplementedError(
            "Multimodal training from main.py not implemented yet. "
            "Provide only 'model_name' in config to run text-only training, "
            "or extend main.py to use your MVSA_MV dataset and model."
        )


    # Text-only training using Hugging Face datasets/transformers
    

    set_seed(seed)

    ds = load_dataset(
        "csv", data_files={"train": train_csv, "validation": val_csv, **({"test": test_csv} if test_csv else {})}
    )

    # Decide which label column to use
    if label_column is None:
        target_map = {"text": "text_label", "image": "image_label", "combined": "combined_label"}
        label_column = target_map.get(target, target)

    if label_column not in ds["train"].column_names:
        raise KeyError(f"Label column '{label_column}' not found in CSV columns: {ds['train'].column_names}")

    # Always read raw text from file paths under 'text_path'
    if "text_path" not in ds["train"].column_names:
        raise KeyError("CSV must contain a 'text_path' column with file paths to text files.")

    ds = ds.map(load_text_from_path)
    text_source_col = "text"

    # Build consistent label mapping from train split
    labels = sorted(set(ds["train"][label_column]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    ds = ds.map(build_add_label_id(label_column, label2id))

    tok = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        return tok(batch[text_source_col], truncation=True, padding="max_length", max_length=max_length)

    ds = ds.map(preprocess, batched=True)

    # Keep only the required columns for the model
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in ds["train"].features:
        keep.add("token_type_ids")
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    compute_metrics = build_compute_metrics(average="macro")

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        weight_decay=weight_decay,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save label map for inference
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)


if __name__ == "__main__":
    main()
