import open_clip
import torch
from torch import nn


class ImageClip(nn.Module):
    """Wrapper around open source ImageClip model."""

    def __init__(self, model_name: str = "ViT-B/16", pretrained: str | None = "openai"):
        super().__init__()
        self.model = open_clip.create_model(
            model_name=model_name, pretrained=pretrained
        )

        # Remove text related layers
        del self.model.transformer
        del self.model.token_embedding
        del self.model.ln_final
        del self.model.positional_embedding
        del self.model.text_projection

        # Remove projection layer
        self.model.visual.proj = None

    def forward(self, image1, image2=None):
        if image2 is not None:
            images = torch.cat([image1, image2], dim=0)
            image_embeddings = self.model.encode_image(images)

            # Split the embeddings
            image1_embedding, image2_embedding = (
                image_embeddings[: len(image1)],
                image_embeddings[len(image1) :],
            )

            return image1_embedding, image2_embedding
        else:
            return self.model.encode_image(image1)
