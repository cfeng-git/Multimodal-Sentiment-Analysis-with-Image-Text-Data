import torch.nn as nn
from transformers import AutoImageProcessor, ViTForImageClassification


class ImageClassifier(nn.Module):
    def __init__(
        self,
        num_labels,
        label2id=None,
        id2label=None,
        model_name="google/vit-base-patch16-224",
        unfreeze_last_layers=0,  # e.g., 2 to unfreeze the top two transformer blocks
    ):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  # rebuild classifier head for your label count
        )

        # freeze everything first
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # optionally unfreeze the top N transformer blocks (and final layernorm)
        if unfreeze_last_layers > 0:
            for layer in self.model.vit.encoder.layer[-unfreeze_last_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.model.vit.layernorm.parameters():
                param.requires_grad = True

        # always train the classification head
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, **batch):
        return self.model(**batch)
