
# Dataset class file for the MVSA multiview dataset. 
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MVSA_MV(Dataset):
    """Dataset wrapper that emits tokenized text, processed images, and labels."""

    def __init__(self, csv_path, target="text", max_len=128, tokenizer=None, image_processor=None, label2id=None):
        self.df = pd.read_csv(csv_path)

        target_map = {"text": "text_label", "image": "image_label", "combined": "combined_label"}
        self.target_col = target_map.get(target, target)
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in CSV")

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len

        labels = sorted(self.df[self.target_col].dropna().unique()) if label2id is None else label2id.keys()
        self.label2id = label2id or {l: i for i, l in enumerate(labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        out = {}

        if self.target_col in ("text_label", "combined_label"):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for text or combined targets.")
            with open(r["text_path"], "r", encoding="utf-8") as f:
                text = f.read().strip()
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            out["input_ids"] = enc["input_ids"].squeeze(0)
            out["attention_mask"] = enc["attention_mask"].squeeze(0)
            if "token_type_ids" in enc:
                out["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        if self.target_col in ("image_label", "combined_label"):
            if self.image_processor is None:
                raise ValueError("Image processor is required for image or combined targets.")
            img = Image.open(r["image_path"]).convert("RGB")
            img_enc = self.image_processor(images=img, return_tensors="pt")
            out["pixel_values"] = img_enc["pixel_values"].squeeze(0)

        out["labels"] = torch.tensor(self.label2id[r[self.target_col]], dtype=torch.long)
        return out
