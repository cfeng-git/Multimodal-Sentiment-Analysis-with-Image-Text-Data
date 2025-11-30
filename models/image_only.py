import torch.nn as nn
from transformers import AutoImageProcessor, ViTForImageClassification


class ImageClassifier(nn.Module):
    """Image-only classifier with a frozen vision backbone and trainable head."""

    def __init__(self, num_labels, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, **batch):
        return self.model(**batch)
