import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str, device: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    return tokenizer, model


def build_prompt(patient_data: dict, retrieved_chunks: list[dict]) -> str:
    """
    patient_data: dict containing age, gender, diagnosis, exams, etc.
    retrieved_chunks: list of selected chunks (after MMR)
    """

    prompt = []

    prompt.append("You are a physician writing a medical dismissal letter from the ER department.\n")

    prompt.append("PATIENT INFORMATION:")
    for key, value in patient_data.items():
        prompt.append(f"- {key}: {value}")
    prompt.append("\n")

    prompt.append("GUIDELINE EXCERPTS (use only these):\n")

    for i, chunk in enumerate(retrieved_chunks, 1):
        prompt.append(
            f"[Excerpt {i}] "
            f"({chunk['doc_id']}, p.{chunk['page_start']}-{chunk['page_end']})\n"
        )
        prompt.append(chunk["text"])
        prompt.append("\n")

    prompt.append(
        "INSTRUCTIONS:\n"
        "- Base your recommendations strictly and exclusively on the provided guideline excerpts.\n"
        "- If a recommendation cannot be supported by the excerpts, state that no guidance is available.\n"
        "- Do not change the diagnosis.\n"
        "- Write a clear, structured dismissal letter including therapy recommendations and precautions.\n"
        "- Cite sources in parentheses.\n"
    )

    return "\n".join(prompt)

def generate_letter(
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



"""messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": ex_query}
]"""