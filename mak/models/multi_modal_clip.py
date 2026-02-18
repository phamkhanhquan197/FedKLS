import torch
from transformers import CLIPModel


class MultiModalCLIP(torch.nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()

        clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)

        # Keep original CLIP modules
        self.vision_model = clip_model.vision_model
        self.text_model = clip_model.text_model
        self.visual_projection = clip_model.visual_projection
        self.text_projection = clip_model.text_projection

        embed_dim = clip_model.config.projection_dim
        self.classifier = torch.nn.Linear(embed_dim * 2, num_labels)

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        vision_embeds = self.visual_projection(vision_outputs.pooler_output)
        text_embeds = self.text_projection(text_outputs.pooler_output)

        combined = torch.cat([vision_embeds, text_embeds], dim=1)
        logits = self.classifier(combined)
        return logits
