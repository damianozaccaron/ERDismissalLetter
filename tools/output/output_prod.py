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


def build_prompt(patient_data: dict, retrieved_chunks: list[dict]) -> str:
    """
    patient_data: dict containing age, gender, diagnosis, exams, etc.
    retrieved_chunks: list of selected chunks (after MMR)
    """

    prompt = []

    prompt.append("You are a medical doctor writing a discharge document for the patient. The document is addressed to the patient. Follow instructions strictly. Write specific, patient-tailored recommendations. Avoid generic statements.\n")

    prompt.append("PATIENT INFORMATION:")
    for key, value in patient_data.items():
        prompt.append(f"- {key}: {value}")
    prompt.append("\n")

    prompt.append("GUIDELINE EXCERPTS:\n")

    for i, chunk in enumerate(retrieved_chunks, 1):
        prompt.append(
            f"[E{i}] "
            f"({chunk['doc_id']}, p.{chunk['page_start']}-{chunk['page_end']})\n"
        )
        prompt.append(chunk["text"])
        prompt.append("\n")

    prompt.append(
        "INSTRUCTIONS:\n"
        "- Use ONLY the provided guideline excerpts. Do not rely on prior knowledge.\n"
        "- Base your recommendations strictly and exclusively on the provided guideline excerpts.\n"    
        "- Do NOT invent or assume any information.\n"
        "- If a recommendation cannot be supported by the excerpts, state that no guidance is available.\n"
        "- Write the discharge document using the extracted recommendations.\n"
        "- Each recommendation MUST explicitly cite at least one excerpt.\n"
        "- Structure the output EXACTLY as follows:\n\n"
        "Patient Data:\n"
        "...(Copy from input)\n\n"
        "Diagnosis:\n"
        "... (Copy from input)\n\n"
        "Exams Performed:\n"
        "...(Copy from input)\n\n"
        "Recommended Therapy:\n"
        "- ... (cite Excerpt number)\n\n"
        "Precautions:\n"
        "- ... (cite Excerpt number)\n\n"
    )

    return "\n".join(prompt)

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
