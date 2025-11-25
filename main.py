import argparse
import os
from pathlib import Path
import json
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
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



class TrainEvalCallback(TrainerCallback):
    """Evaluate on the training set at the end of each epoch."""

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self.trainer = None  # to be set after Trainer is created

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control

        # 1) Evaluate on the training set
        train_metrics = self.trainer.evaluate(
            eval_dataset=self.train_dataset,
            metric_key_prefix="train",
        )

        # 2) Evaluate on the validation set (uses trainer.eval_dataset by default)
        eval_metrics = self.trainer.evaluate(
            metric_key_prefix="eval",
        )

        print(f"[Train metrics] epoch={state.epoch}: {train_metrics}")
        print(f"[Eval metrics]  epoch={state.epoch}: {eval_metrics}")
        return control


class EvalPrintCallback(TrainerCallback):
    """Print eval metrics when evaluation runs (e.g., end of each epoch)."""

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"[Eval metrics] epoch={state.epoch}: {metrics}")
        return control


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
    model_name = model_section.get("model_name")  # model used to predict sentiment
    text_model = model_section.get("text_model")  # model used to process text data and generate tokens
    vision_model = model_section.get("vision_model")  # model used to process image data
    max_length = model_section.get("max_length", 128)

    # get parameters related to data
    data_cfg = config["data"]
    train_csv = data_cfg["train_csv"]
    val_csv = data_cfg["val_csv"]
    test_csv = data_cfg.get("test_csv")
    target = data_cfg.get("target", "text")  # the label we want to predict, could be text (default), image, or combined

    # get parameters related logging: 
    out_cfg = config.get("logging", {})
    output_dir = out_cfg.get("output_dir", str((PROJECT_ROOT / "outputs" / "checkpoints" / "text").resolve()))
    
    # set random seed 
    set_seed(seed)

    # load dataset 
    ds = load_dataset(
            "csv", data_files={"train": train_csv, "validation": val_csv, **({"test": test_csv} if test_csv else {})}
        )
    
    # Decide which label column to use
    target_map = {
        "text": "text_label", 
        "image": "image_label", 
        "combined": "combined_label"
        }
    label_column = target_map.get(target, target)

    if label_column not in ds["train"].column_names:
        raise KeyError(f"Label column '{label_column}' not found in CSV columns: {ds['train'].column_names}")

    



    # different training loops for text only, image only, or combined 
    # for prediction based on text or image, use Hugging Face modules to finetune the model

    
    if target == 'combined':
        raise NotImplementedError(
            "Multimodal training from main.py not implemented yet. "
            "Provide only 'model_name' in config to run text-only training, "
            "or extend main.py to use your MVSA_MV dataset and model."
        )


    # training loop for text only models. 
    if target == 'text': 
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

        # Freeze everything first
        for p in model.parameters():
            p.requires_grad = False
        
        # Unfreeze everything
        for p in model.parameters():
            p.requires_grad = True

        # Unfreeze classifier head
        for p in model.classifier.parameters():
            p.requires_grad = True

        # # Optionally unfreeze last N encoder layers across common architectures
        # def _find_encoder_layers(m):
        #     # Try common base model attributes
        #     for attr in [
        #         "bert",
        #         "roberta",
        #         "deberta",
        #         "deberta_v2",
        #         "xlm_roberta",
        #         "longformer",
        #         "electra",
        #         "albert",
        #         "mobilebert",
        #         "distilbert",
        #     ]:
        #         sub = getattr(m, attr, None)
        #         if sub is None:
        #             continue
        #         # Check common encoder containers
        #         for enc_name, layer_name in [
        #             ("encoder", "layer"),
        #             ("encoder", "block"),
        #             ("encoder", "h"),
        #             ("transformer", "layer"),
        #             ("transformer", "h"),
        #         ]:
        #             enc = getattr(sub, enc_name, None)
        #             if enc is None:
        #                 continue
        #             layers = getattr(enc, layer_name, None)
        #             if layers is not None:
        #                 return layers
        #     # Fallback: base_model.encoder.layer/block/h
        #     base = getattr(m, "base_model", None)
        #     if base is not None:
        #         enc = getattr(base, "encoder", None)
        #         if enc is not None:
        #             for name in ["layer", "block", "h"]:
        #                 layers = getattr(enc, name, None)
        #                 if layers is not None:
        #                     return layers
        #     return None

        # encoder_layers = _find_encoder_layers(model)
        # N = 2
        # if encoder_layers is not None:
        #     for layer in encoder_layers[-N:]:
        #         for p in layer.parameters():
        #             p.requires_grad = True
        # else:
        #     print("[WARN] Could not locate encoder layers to unfreeze; only classifier is trainable.")

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
        train_cb = TrainEvalCallback(ds["train"])
        trainer.add_callback(train_cb)
        train_cb.trainer = trainer

        trainer.train()

        # Save label map for inference
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "label_map.json"), "w") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f)


if __name__ == "__main__":
    main()
