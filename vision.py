import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

def load_vision_model():
    caption_pipeline = pipeline(
        task="image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning",
        device=device
    )
    return caption_pipeline


def generate_caption(image, caption_pipeline):
    result = caption_pipeline(image)
    return result[0]["generated_text"]
