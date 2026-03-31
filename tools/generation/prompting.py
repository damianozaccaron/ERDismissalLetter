import deepl

def deepl_translation(text, auth_key = "", glossary = "627f273f-d457-400f-a171-d04b9c13ddf3"):

    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(text, source_lang="IT", target_lang="EN-GB", glossary=glossary)

    return result.text

def deepl_translation_en_it(text, auth_key = ""):

    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(text, source_lang="EN-GB", target_lang="IT")

    return result.text


def build_prompt(patient_data: str, retrieved_chunks: list[dict]) -> str:

    prompt = []
 
    prompt.append(
        "ROLE:\n"
        "You are an emergency department physician writing a discharge "
        "document for a patient. The document will "
        "be read by the patient's general practitioner and by the patient. "
        "Write in a clinical but accessible tone.\n"
    )
 
    prompt.append("CLINICAL NOTE:\n")
    prompt.append(patient_data)
    prompt.append("\n")
 
    prompt.append("GUIDELINE EXCERPTS:\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        prompt.append(
            f"[E{i}] ({chunk['doc_id']}, "
            f"p.{chunk['page_start']}-{chunk['page_end']})\n"
        )
        prompt.append(chunk["text"])
        prompt.append("\n")
 
    prompt.append(
        "TASK:\n"
        "Using the clinical note and guideline excerpts above, produce a "
        "discharge document structured EXACTLY as follows.\n\n"
 
        "1. MEDICAL HISTORY\n"
        "   Copy the medical history from the clinical note. Do not alter, "
        "summarize, or omit any detail.\n\n"
 
        "2. OBJECTIVE EXAMINATION\n"
        "   Copy the objective examination from the clinical note verbatim.\n\n"
 
        "3. CLINICAL DIARY\n"
        "   Copy the clinical diary from the clinical note verbatim.\n\n"
 
        "4. PROGNOSIS\n"
        "   Copy the prognosis from the clinical note verbatim.\n\n"
 
        "5. RECOMMENDATIONS AND PRESCRIPTIONS\n"
        "   Provide specific, actionable recommendations. Each recommendation "
        "MUST:\n"
        "   - Be supported by at least one guideline excerpt, cited as [E#]\n"
        "   - Reference the patient's specific clinical data that makes the "
        "recommendation applicable (e.g. \"Given your CHA₂DS₂-VASc score "
        "of 3...\" or \"Given preserved LVEF...\")\n"
        "   - Include drug names and dosages where applicable\n"
        "   - Be grouped into: Pharmacological therapy, Follow-up "
        "investigations, Lifestyle modifications, and Follow-up appointments\n\n"
        "   If a clinically relevant topic (e.g. anticoagulation, rate "
        "control) cannot be addressed by the provided excerpts, state:\n"
        "   \"No specific guidance available from provided sources. Refer to "
        "specialist for [topic].\"\n"
    )
 
    prompt.append(
        "RULES:\n"
        "- Use ONLY the provided guideline excerpts. Do not rely on prior "
        "medical knowledge for recommendations.\n"
        "- Do NOT invent dosages, drug names, or clinical facts not present "
        "in the clinical note or excerpts.\n"
        "- When multiple excerpts address the same topic, synthesize them "
        "into a single coherent recommendation and cite all relevant "
        "excerpts.\n"
        "- Do not repeat the same recommendation under different wording.\n"
    )
 
    return "\n".join(prompt)
