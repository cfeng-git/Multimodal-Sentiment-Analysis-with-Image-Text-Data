import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate
from transformers import AutoImageProcessor, AutoTokenizer, set_seed

from data.mvsa_mv import MVSA_MV
from models.image_only import ImageClassifier
from models.multimodal import MultimodalClassifier
from models.text_only import TextClassifier
from utils.config import load_config, merge_config_and_args, validate_config
from tqdm.auto import tqdm 

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_SPLIT_DIR = PROJECT_ROOT / "data" / "MVSA" / "splits"
TRAIN_CSV = DATA_SPLIT_DIR / "train.csv"
VAL_CSV = DATA_SPLIT_DIR / "valid.csv"
TEST_CSV = DATA_SPLIT_DIR / "test.csv"
TEXT_MODEL_ID = "vinai/bertweet-base"
VISION_MODEL_ID = "google/vit-base-patch16-224"





def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


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
    parser.add_argument(
        "--training",
        type=str2bool,
        default=True,
        help="Whether to run training (train/val) or just test evaluation.",
    )

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
    training_mode = args.training

    # model selection (name corresponds to a class under models/)
    model_section = config["model"]
    model_key = model_section.get("name")
    max_length = model_section.get("max_length", 128)

    target_by_model = {"text_only": "text", "image_only": "image", "multimodal": "combined"}
    if model_key not in target_by_model:
        raise ValueError(f"model.name must be one of {list(target_by_model)}, got {model_key}")
    target = target_by_model[model_key]

    # set random seed
    set_seed(seed)

    # Build preprocessors based on modality
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, use_fast=False) if target in ("text", "combined") else None
    image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_ID) if target in ("image", "combined") else None

    # Build datasets (hardcoded split paths)
    train_ds = MVSA_MV(
        csv_path=TRAIN_CSV,
        target=target,
        max_len=max_length,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    label2id = train_ds.label2id
    id2label = {i: l for l, i in label2id.items()}

    val_ds = MVSA_MV(
        csv_path=VAL_CSV,
        target=target,
        max_len=max_length,
        tokenizer=tokenizer,
        image_processor=image_processor,
        label2id=label2id,
    )
    test_ds = MVSA_MV(
        csv_path=TEST_CSV,
        target=target,
        max_len=max_length,
        tokenizer=tokenizer,
        image_processor=image_processor,
        label2id=label2id,
    )

    num_labels = len(label2id)

    # Instantiate model
    if model_key == "text_only":
        model = TextClassifier(num_labels=num_labels)
        tokenizer = model.tokenizer
    elif model_key == "image_only":
        model = ImageClassifier(num_labels=num_labels, label2id=label2id, id2label=id2label)
        image_processor = model.processor
    else:
        model = MultimodalClassifier(num_labels=num_labels)
        tokenizer = model.tokenizer
        image_processor = model.image_processor


    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu") 
    model.to(device)

    def to_device(batch):
        return {k: v.to(device) for k, v in batch.items()}

    def run_epoch(loader, train: bool):
        model.train(mode=train)
        total_loss = total_correct = total_samples = 0
        iterator = tqdm(loader, desc="train" if train else "eval", leave=False)

        if train:
            optimizer.zero_grad(set_to_none=True)

        for batch in iterator:
            if batch is None:
                continue
            batch = to_device(batch)
            with torch.set_grad_enabled(train):
                outputs = model(**batch)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                if loss is None:
                    loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch["labels"]).sum().item()
            total_samples += preds.size(0)
            total_loss += loss.item() * preds.size(0)

            avg_loss = total_loss / max(total_samples, 1)
            acc = total_correct / max(total_samples, 1)
            iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

        return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)

    # DataLoaders
    def safe_collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return default_collate(batch)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=safe_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=safe_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=safe_collate)

    # Optimizer on trainable params only
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    if training_mode:
        print(f"Training {model_key} on {target} labels...")
        for epoch in range(num_epochs):
            train_loss, train_acc = run_epoch(train_loader, train=True)
            val_loss, val_acc = run_epoch(val_loader, train=False)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
    else:
        print(f"Evaluating {model_key} on test split...")

    test_loss, test_acc = run_epoch(test_loader, train=False)
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
