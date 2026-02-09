import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    ).to(device)
    return tokenizer, model

def generate_answer(question, context, tokenizer, model):
    prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=200)

    return tokenizer.decode(output[0], skip_special_tokens=True)
