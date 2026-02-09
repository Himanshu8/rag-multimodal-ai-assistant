import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"


def load_vision_model():
    image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    return image_processor, tokenizer, model


def generate_caption(image, vision_bundle):
    image_processor, tokenizer, model = vision_bundle

    pixel_values = image_processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=40,
            num_beams=4
        )

    caption = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    return caption
