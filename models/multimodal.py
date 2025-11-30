import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, ViTModel


class MultimodalClassifier(nn.Module):
    """Concatenate frozen text and vision encoders, train only the classifier head."""

    def __init__(
        self,
        num_labels,
        text_model_name="vinai/bertweet-base",
        vision_model_name="google/vit-base-patch16-224",
    ):
        super().__init__()
        # BERTweet uses the slow tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=False)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)

        fused_hidden = self.text_encoder.config.hidden_size + self.vision_encoder.config.hidden_size
        self.classifier = nn.Linear(fused_hidden, num_labels)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None, labels=None):
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state[:, 0]

        vision_out = self.vision_encoder(pixel_values=pixel_values).pooler_output
        fused = torch.cat([text_out, vision_out], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
