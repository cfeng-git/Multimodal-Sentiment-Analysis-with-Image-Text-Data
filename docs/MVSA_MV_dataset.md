# MVSA_MV Dataset

`data/mvsa_mv.py` provides a PyTorch `Dataset` for the MVSA multimodal corpus (text + image). It tokenizes text via a Hugging Face `AutoTokenizer` and preprocesses images via `ViTImageProcessor`. You can choose which modality’s labels to predict.

## CSV Schema

Required columns:
- `text_path`: path to a UTF‑8 text file for the sample
- `image_path`: path to the corresponding image (RGB-readable)
- `text_label`: class label for the text modality
- `image_label`: class label for the image modality

Optional:
- `combined_label`: joint label if training on both modalities

## Initialization

```python
from data.mvsa_mv import MVSA_MV

ds = MVSA_MV(
    csv_path="data/MVSA/splits/train.csv",
    target="text",                        # "text", "image", or "combined"; or pass a column name directly
    max_len=128,                          # tokenizer max length
    text_model="distilbert-base-uncased", # HF text checkpoint
    vision_model="google/vit-base-patch16-224",  # HF ViT checkpoint
)
```

Arguments:
- `csv_path` (str): Path to the split CSV.
- `target` (str): Which label to predict. Use `"text"`, `"image"`, or `"combined"`. You can also pass the exact column name (e.g., `"text_label"`).
- `max_len` (int): Max tokenized sequence length (default: 128).
- `text_model` (str): Hugging Face text model checkpoint (default: `distilbert-base-uncased`).
- `vision_model` (str): Hugging Face ViT checkpoint (default: `google/vit-base-patch16-224`).

## Returned Sample

Each `__getitem__` returns a dict of tensors ready for a model:

- Text inputs (when `target` uses text or combined):
  - `input_ids`: token IDs
  - `attention_mask`: attention mask
  - `token_type_ids` (if tokenizer provides it)
- Image inputs (when `target` uses image or combined):
  - `pixel_values`: image tensor normalized for ViT
- Target:
  - `labels`: `torch.long` class id mapped from the chosen target column

Attributes:
- `labels`: Sorted list of class names from the target column.
- `label2id`: Mapping from class name → integer id.

## Typical Usage

```python
import torch
from torch.utils.data import DataLoader
from data.mvsa_mv import MVSA_MV

train = MVSA_MV("data/MVSA/splits/train.csv", target="text")
valid = MVSA_MV("data/MVSA/splits/valid.csv", target="text")

train_loader = DataLoader(train, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid, batch_size=16)
```

- Create separate instances per split by pointing at each split CSV.
- Default PyTorch collation batches the dict of tensors.

## Notes and Best Practices

- Ensure the selected target column exists in your CSV (e.g., `combined_label` for `target="combined"`).
- Keep label ids consistent across splits:
  - Derive `label2id` from the train split and reuse it for validation/test.
- Choose `target` based on your training setup:
  - Text classification: `target="text"`
  - Image classification: `target="image"`
  - Multimodal models: `target="combined"` and ensure your model accepts both text and image inputs.
