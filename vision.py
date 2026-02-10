import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/blip-image-captioning-base"


def load_vision_model():
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    return processor, model


def generate_caption(image, vision_bundle):
    processor, model = vision_bundle

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,
            early_stopping=True
        )

    caption = processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    return caption
