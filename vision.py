import torch
from transformers import pipeline

def load_vision_model():
    device = 0 if torch.cuda.is_available() else -1

    caption_pipeline = pipeline(
        task="image-text-to-text",
        model="nlpconnect/vit-gpt2-image-captioning",
        device=device,
    )
    return caption_pipeline


def generate_caption(image, caption_pipeline):
    result = caption_pipeline(image)
    return result[0]["generated_text"]
