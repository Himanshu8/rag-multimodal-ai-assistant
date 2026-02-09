import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-base"


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def generate_answer(question, context, tokenizer, model):
    prompt = (
        "You are a helpful AI assistant. Use the context below to answer "
        "the question in a clear and complete manner.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,      # 🔥 KEY FIX
            do_sample=True,          # allow richer outputs
            temperature=0.7,         # balanced creativity
            top_p=0.9,               # nucleus sampling
            repetition_penalty=1.1,  # avoid loops
            early_stopping=True
        )

    answer = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    return answer.strip()
