import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, ViTModel, AutoConfig, CLIPVisionModel, SiglipVisionModel 


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
        self.text_encoder = AutoModel.from_pretrained(text_model_name)



        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name) 

        # Pick the right vision backbone based on the model type
        cfg = AutoConfig.from_pretrained(vision_model_name)
        model_type = cfg.model_type
        if model_type in ("siglip_vision_model", "siglip"):
            self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model_name)
        elif model_type in ("clip_vision_model", "openai-clip"):
            self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        elif model_type == "vit":
            self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        else:
            # Fallback: some CLIP-family checkpoints report custom model_type strings.
            if "clip" in model_type:
                self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
            else:
                raise ValueError(f"Unsupported vision model type '{model_type}' for {vision_model_name}")



        fused_hidden = self.text_encoder.config.hidden_size + self.vision_encoder.config.hidden_size
        self.classifier = nn.Linear(fused_hidden, num_labels)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        labels=None,
    ):
        if input_ids is None or pixel_values is None:
            raise ValueError("Both text (input_ids) and image (pixel_values) inputs are required for multimodal forward.")

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Prefer pooler_output if available, otherwise fall back to CLS token
        if getattr(text_outputs, "pooler_output", None) is not None:
            text_repr = text_outputs.pooler_output
        else:
            text_repr = text_outputs.last_hidden_state[:, 0]

        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        if getattr(vision_outputs, "pooler_output", None) is not None:
            vision_repr = vision_outputs.pooler_output
        else:
            vision_repr = vision_outputs.last_hidden_state[:, 0]

        fused = torch.cat([text_repr, vision_repr], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
