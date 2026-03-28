import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
 #from llama_cpp import Llama



def load_model_transformer(model_name: str, device: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name).to(device)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()

    return tokenizer, model

def load_model_quant(repo, model_name):

    llm = Llama.from_pretrained(
        repo_id=repo,
        filename=model_name,
        n_gpu_layers=-1
    )

    return llm

def generate_letter_transformer(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 800,
    temperature=0.2):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0)

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True)

    # Remove prompt from output
    return decoded[len(prompt):].strip()

def generate_letter_quant(
        prompt: str, 
        llm, 
        max_new_tokens=800, 
        temperature=0.2):
    
    formatted_prompt = f"[INST] {prompt} [/INST]"

    output = llm(
        formatted_prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        stop=["</s>"],
        echo=True
    )

    return output["choices"][0]["text"].strip()
