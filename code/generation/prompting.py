def build_preamble(patient_data: str, retrieved_chunks: list[dict], role: str) -> list[str]:
    """Shared preamble: ROLE + CLINICAL NOTE + GUIDELINE EXCERPTS."""

    prompt = []

    prompt.append(f"ROLE:\n{role}\n")

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

    return prompt


def build_discharge_prompt(patient_data: str, retrieved_chunks: list[dict]) -> str:
    """Prompt for patients discharged home."""

    role = (
        "You are an emergency department physician writing discharge "
        "recommendations for a patient. The document will be read by "
        "the patient and their general practitioner."
    )

    prompt = build_preamble(patient_data, retrieved_chunks, role)

    # ── TASK ──
    prompt.append(
        "TASK:\n"
        "Produce discharge recommendations in two sections.\n\n"

        "SECTION 1 — DISCHARGE RECOMMENDATIONS\n"
        "A numbered list of concise instructions for the patient.\n"
        "Rules:\n"
        "- One or two sentences per item. Be direct: "
        "\"Continue...\", \"Stop...\", \"A prescription is issued for...\".\n"
        "- Combine closely related actions into a single item "
        "(e.g. stopping a medication and adopting an alternative "
        "is one recommendation, not two). "
        "Never combine medication prescriptions with non-medication actions.\n"
        "- For medications: always include drug name, dose, route, "
        "and frequency (e.g. \"every 12 hours\"). "
        "Copy the exact dosing from the clinical note. "
        "If frequency is not in the clinical note or excerpts, "
        "write \"as directed by the specialist\".\n"
        "- Write out all medical abbreviations in full (e.g. 'subcutaneously' not 's.c.', 'intravenously' not 'i.v.').\n"
        "- For lifestyle factors (smoking, alcohol, diet, exercise): "
        "use strong advisory tone (\"strongly recommended\", "
        "\"strongly advised\") rather than direct orders.\n"
        "- For follow-up appointments and specialist referrals, "
        "use the phrasing: \"A prescription is issued for "
        "[specialist type] evaluation for [clinical purpose]\" "
        "and always specify the specialist type from the clinical note "
        "(e.g. \"angiologist\", \"cardiologist\", \"internist\") "
        "and the clinical purpose of the visit "
        "(e.g. \"for thrombophilia screening\", "
        "\"for initiation of anticoagulant therapy\", "
        "\"for echocardiographic follow-up\").\n"
        "- Do NOT explain why. No rationale, no \"because...\", "
        "no \"to reduce the risk of...\", no pathophysiology.\n"
        "- Do NOT speculate about future treatment decisions "
        "(duration of therapy, drug switches). "
        "Write \"to be determined by the specialist\" and move on.\n"
        "- No sub-bullets, no bold, no markdown headers, "
        "no category groupings. Just a numbered list.\n"
        "- Do NOT cite [E#] in this section. Citations belong "
        "only in Section 2.\n"
        "- The second-to-last item must always be: "
        "\"Go to the nearest emergency department immediately "
        "if you experience [relevant warning symptoms "
        "for this diagnosis].\"\n"
        "- The last item must always be: "
        "\"You are referred back to your general practitioner "
        "for any further diagnostic or therapeutic needs "
        "and ongoing care.\"\n"
        "- If a topic cannot be addressed by the excerpts: "
        "\"Refer to specialist for [topic].\"\n\n"

        "SECTION 2 — CLINICAL REFERENCES (for the reviewing physician)\n"
        "For each recommendation number in Section 1, write:\n"
        "[number] - [E#] - clinical justification\n"
        "Example:\n"
        "1 - [E4][E7] - Anticoagulation recommended for confirmed "
        "distal DVT with high pretest probability.\n"
        "3 - [E7] - Oral contraceptives classified as moderate VTE "
        "risk factor (OR 2-9).\n"
    )

    # ── RULES ──
    prompt.append(
        "RULES:\n"
        "- Use ONLY the provided guideline excerpts. "
        "Do not rely on prior medical knowledge.\n"
        "- Do NOT invent drug names, dosages, frequencies, or clinical "
        "facts not in the clinical note or excerpts.\n"
        "- When multiple excerpts cover the same topic, write one "
        "recommendation and cite all relevant excerpts.\n"
        "- Do not repeat the same recommendation in different wording.\n"
        "- Section 1 must be directly usable as a discharge document with no editing needed.\n"
    )

    return "\n".join(prompt)


def build_hospitalisation_prompt(patient_data: str, retrieved_chunks: list[dict]) -> str:
    """Prompt for patients admitted to hospital from the ER."""

    role = (
        "You are an emergency department physician writing recommendations "
        "for a patient being admitted to hospital. The document "
        "will be read by the admitting team, the patient, and their family."
    )

    prompt = build_preamble(patient_data, retrieved_chunks, role)

    # ── TASK ──
    prompt.append(
        "TASK:\n"
        "The patient is being admitted to hospital. The ER does not "
        "prescribe ongoing therapy or follow-up — that is the admitting "
        "team's responsibility. Produce a brief handover note in two "
        "sections.\n\n"

        "SECTION 1 — ADMISSION NOTE\n"
        "In a numbered list: \n"
        "- State that the patient is admitted to [department] "
        "for [key planned interventions], extracting the department "
        "and planned interventions from the clinical note "
        "(e.g. \"Patient is admitted urgently to the Paediatrics "
        "department for high-dose intravenous immunoglobulin therapy, "
        "dedicated transthoracic echocardiography with coronary "
        "evaluation, and clinical-laboratory monitoring.\").\n"
        "- State that the patient and/or family have been informed "
        "of the diagnosis, the expected clinical course based on the "
        "guideline excerpts (e.g. treatment protocol, possible "
        "complications, need for monitoring), and the planned "
        "treatment pathway. Do not write examples.\n"
        "- Do NOT give examples of complications or risks in parentheses. "
        "State the information directly without \"e.g.\", \"such as\", "
        "or \"for example\".\n"
        "- If the patient is being hospitalised to perform surgery,"
        "state that the patient has been prepped. \n"
        "- State that the patient is handed over to the care of "
        "the [department] team.\n\n"
        "- Do NOT prescribe medications, follow-up visits, or "
        "lifestyle changes.\n"
        "- Do NOT cite [E#] in this section.\n"
        "- Keep each item to one or two sentences.\n\n"

        "SECTION 2 — CLINICAL REFERENCES (for the reviewing physician)\n"
        "For each item in Section 1 that draws on guideline content, "
        "write:\n"
        "[number] - [E#] - clinical justification\n"
        "Example:\n"
        "2 - [E1][E9] - IVIG 2g/kg and aspirin 30-50mg/kg/day are "
        "first-line treatment for Kawasaki disease to prevent coronary "
        "artery aneurysms.\n"
    )

    # ── RULES ──
    prompt.append(
        "RULES:\n"
        "- Use ONLY the provided guideline excerpts. "
        "Do not rely on prior medical knowledge.\n"
        "- Do NOT invent clinical facts not in the clinical note "
        "or excerpts.\n"
        "- Section 1 must be directly usable as a handover note "
        "with no editing needed.\n"
    )

    return "\n".join(prompt)


def build_prompt(
    patient_data: str,
    retrieved_chunks: list[dict],
    prognosis: str = "",
) -> str:
    """
    Selects the appropriate prompt template based on the prognosis of hospitalisation.
    If the patient is hospitalised, the prompt gives more generic instructions since patient is handed over to a different team.
    """

    hospitalisation_keywords = [
        "hospitalisation", "hospitalization",
        "admission", "admitted", "urgent hospitalisation",
    ]

    is_hospitalised = any(kw in prognosis.lower() for kw in hospitalisation_keywords)

    if is_hospitalised:
        return build_hospitalisation_prompt(patient_data, retrieved_chunks)
    else:
        return build_discharge_prompt(patient_data, retrieved_chunks)