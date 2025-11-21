
# Dataset class file for the MVSA multiview dataset. 
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor
import pandas as pd

class MVSA_MV(Dataset):
    def __init__(self, csv_path, target="text", max_len=128,
                 text_model="distilbert-base-uncased", vision_model="google/vit-base-patch16-224"):
        
        #   Ensure your model’s forward can accept the keys you return:
        #   Text-only: input_ids, attention_mask, labels
        #   Image-only: pixel_values, labels
        #   Multimodal: include all three inputs
        #   For consistent label ids across splits, pass label2id from the train split into validation/test, or persist and reuse the mapping.
        
        self.df = pd.read_csv(csv_path)
        
        target_map = {"text": "text_label", "image": "image_label", "combined": "combined_label"}
        self.target_col = target_map.get(target, target)
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in CSV")

        self.tok = AutoTokenizer.from_pretrained(text_model)
        self.max_len = max_len                          # max length for the text encoder

        self.proc = ViTImageProcessor.from_pretrained(vision_model)

        self.labels = sorted(self.df[self.target_col].dropna().unique())      # store all possible labels
        self.label2id = {l:i for i,l in enumerate(self.labels)}     # used to convert label to number
        

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        
        # get data row 
        r = self.df.iloc[i]             

        # initialize what to return 
        out = {}

        # get text and process
        if self.target_col in ("text_label", "combined_label"):
            with open(r["text_path"], "r", encoding="utf-8") as f:
                text = f.read().strip()
            enc = self.tok(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_len, 
                return_tensors="pt"
            )
            out["input_ids"] = enc["input_ids"].squeeze(0)
            out["attention_mask"] = enc["attention_mask"].squeeze(0)
            if "token_type_ids" in enc:
                out["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        # get image and process
        if self.target_col in ("image_label", "combined_label"):
            img = Image.open(r["image_path"]).convert("RGB")
            img_enc = self.proc(images=img, return_tensors="pt")
            out["pixel_values"] = img_enc["pixel_values"].squeeze(0)

        out["labels"] = torch.tensor(self.label2id[r[self.target_col]], dtype=torch.long)

        return out 