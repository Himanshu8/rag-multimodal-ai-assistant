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
        "Answer the question using the context below. "
        "Be clear, concise, and informative.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,   # 🔥 reduced for speed
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,      # 🔥 CRITICAL: fast on CPU
            do_sample=False,         # 🔥 deterministic & faster
            temperature=1.0,         # ignored when do_sample=False
            repetition_penalty=1.1,
            early_stopping=True
        )

    answer = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    return answer.strip()
