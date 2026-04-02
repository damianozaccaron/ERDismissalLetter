import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_transformer(model_name: str, device: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name).to(device)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()

    return tokenizer, model

def load_model_quant(repo, model_name):
    from llama_cpp import Llama

    llm = Llama.from_pretrained(
        repo_id=repo,
        filename=model_name,
        n_gpu_layers=-1,
        n_ctx=4096,
    )

    return llm


def generate_letter(prompt: str, tokenizer, model, max_new_tokens: int = 800, temperature=0.2):
    if tokenizer is not None:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                echo=True)

        decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True)

        return decoded[len(prompt):].strip()
    
    else:
        formatted_prompt = f"[INST] {prompt} [/INST]"

        output = model(
            formatted_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["</s>"],
            echo=True
        )

        return output["choices"][0]["text"].strip()
