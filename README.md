# Multimodal Sentiment Analysis with Image-Text Data

Training and evaluation are driven by `main.py` and the MVSA splits under `data/MVSA/splits/`.

## Quick start

```bash
# Text model (default config)
python main.py --config configs/train_text.yaml --training true

# Test only with a saved checkpoint
python main.py --config configs/train_text.yaml --training false --checkpoint outputs/checkpoints/text_only_best.pt
```

- Data paths are fixed to `data/MVSA/splits/train.csv`, `valid.csv`, `test.csv`.
- Models live under `models/` and are subclasses of `nn.Module`, with frozen backbones and trainable heads.
- Best validation checkpoints and training curves are saved to `outputs/checkpoints/`.

## Config

`configs/train_text.yaml` fields:

- `training`: `batch_size`, `num_epochs`, `learning_rate`, `weight_decay`, `seed`
- `model`: `name` (one of `text_only`, `image_only`, `multimodal`), `max_length`

The config no longer needs data paths or Hugging Face IDs; those are hardcoded (`vinai/bertweet-base` for text, `google/vit-base-patch16-224` for images).

## Training vs testing

- `--training true`: runs train/val loops, tracks best validation accuracy, saves the best model to `outputs/checkpoints/<model>_best.pt`, and saves a loss/accuracy plot to `outputs/checkpoints/<model>_metrics.png`. Test evaluation runs using the best checkpoint.
- `--training false`: skips training/validation and only evaluates on the test split. Provide `--checkpoint` to load a saved `state_dict`.

## Models

- `text_only`: BERTweet text classifier (`vinai/bertweet-base`), frozen encoder, trainable head.
- `image_only`: ViT image classifier (`google/vit-base-patch16-224`), frozen encoder, trainable head.
- `multimodal`: concatenates frozen text and vision encoders; trains a fusion head.

## Dataset

[MVSA Multiview Dataset](./docs/MVSA_MV_dataset.md)
