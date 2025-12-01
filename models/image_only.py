import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    CLIPVisionModel,
    SiglipVisionModel,
    ViTModel,
)


class ImageClassifier(nn.Module):
    def __init__(self, num_labels, label2id=None, id2label=None, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Pick the right vision backbone based on the model type
        cfg = AutoConfig.from_pretrained(model_name)
        model_type = cfg.model_type
        if model_type in ("siglip_vision_model", "siglip"):
            self.vision = SiglipVisionModel.from_pretrained(model_name)
        elif model_type in ("clip_vision_model", "openai-clip"):
            self.vision = CLIPVisionModel.from_pretrained(model_name)
        elif model_type == "vit":
            self.vision = ViTModel.from_pretrained(model_name)
        else:
            # Fallback: some CLIP-family checkpoints report custom model_type strings.
            if "clip" in model_type:
                self.vision = CLIPVisionModel.from_pretrained(model_name)
            else:
                raise ValueError(f"Unsupported vision model type '{model_type}' for {model_name}")

        hidden = self.vision.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

        # Freeze backbone, train only the classification head
        for param in self.vision.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, **batch):
        labels = batch.get("labels", None)
        vision_outputs = self.vision(pixel_values=batch["pixel_values"])
        pooled = vision_outputs.pooler_output
        if pooled is None:
            # Some backbones (e.g., ViT) may not return pooler_output; fall back to CLS token.
            pooled = vision_outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
