import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TextClassifier(nn.Module):
    """Text-only classifier with a frozen backbone and trainable head."""

    def __init__(self, num_labels, model_name="vinai/bertweet-base"):
        super().__init__()
        # BERTweet requires the slow tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        # Freeze all parameters except the classifier head
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, **batch):
        return self.model(**batch)
