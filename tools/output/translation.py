# Load model directly
from transformers import AutoTokenizer, AutoModel
import torch

def load_translator(model_name: str = "google/madlad400-3b-mt"):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model

def translate_to_english(text: str, tokenizer, model) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

import time

if __name__ == "__main__":
    start_time = time.time()
    print("Loading Translation Model...")

    input = """"""

    print("Translating Query...")
    tokenizer, model = load_translator()
    translated = translate_to_english(input, tokenizer=tokenizer, model=model)

    end_time = time.time()

    with open('translationAF.txt', 'w', encoding='utf-8') as file:
        file.write(translated)

    print(f"Translated document in {end_time-start_time:.3f} seconds")
