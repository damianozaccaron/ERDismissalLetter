import re


def collect_patient_input() -> str:
    """
    Mock function to collect patient data. In a real implementation, this would be replaced by a form or an interface to input the data.
    Only gives the idea that the user follows a pre-determined structure to input the data.
    """

    def prompt_required(prompt_text: str) -> str:
        value = ""
        while not value.strip():
            print(prompt_text)
            value = input()
            if not value.strip():
                print("Questo campo non può essere vuoto.")
        return value.strip()

    def prompt_optional(prompt_text: str) -> str:
        print(prompt_text)
        return input().strip()
    
    clinical_note = []

    print("Genere (M/F/A): ")
    gender = ""
    while gender.lower() not in ('m', 'f', 'a'):
        gender = input()
        if gender.lower() not in ('m', 'f', 'a'):
            print("Genere deve essere M (maschio), F (femmina) o A (altro): ")
    
    gender_exp = "Maschio"
    if gender.lower() == 'f':
        gender_exp = "Femmina"
    elif gender.lower() == 'a':
        gender_exp = "Altro"

    age = prompt_required("\n\nEtà: ")
    clinical_note.append(gender_exp+", "+age)

    clinical_note.append("\nAnamnesi: ")

    pat_prox = prompt_required("\n\nPatologica Prossima: ")
    clinical_note.append("- Patologia Prossima:\n" + pat_prox)

    pat_rem = prompt_optional("\n\nPatologica Remota (lasciare vuoto se non presente): ")
    if pat_rem:
        clinical_note.append("- Patologia Remota:\n" + pat_rem)

    exam = prompt_required("\n\nEsame Obiettivo: ")
    clinical_note.append("\nEsame Obiettivo:\n" + exam)
    
    diary = prompt_required("\n\nDiario Clinico: ")
    clinical_note.append("\nDiario Clinico: \n" + diary)

    diagnosis = prompt_required("\n\nDiagnosi: ")
    clinical_note.append("\nDiagnosi: "+diagnosis)

    prognosis = prompt_required("\n\nPrognosi: ")
    clinical_note.append("\nPrognosi: "+prognosis)


    clinical_note_str = "\n".join(clinical_note)

    return clinical_note_str


def extract_patient_fields(translated_note: str) -> dict:
    """
    Takes as input the translated note as raw text and returns a dictionary with the relevant fields extracted.
    """
    fields = {}

    # demographics (everything before comma is gender, the first digits after the comma are the age)
    demo_match = re.match(r'^[^,]+,\s*(?:(\d+)\s*(?:years?\s*old|aa))?\s*(?:(\d+)\s*months?)?', translated_note, re.IGNORECASE)

    gender = translated_note.split(",", 1)[0].strip()
    if gender.lower() == "other":
        gender = ""
    fields["gender"] = gender

    # age
    years = int(demo_match.group(1)) if demo_match.group(1) else None
    months = int(demo_match.group(2)) if demo_match.group(2) else None

    if years is not None:
        fields["age"] = years
    elif months is not None:
        fields["age"] = 0
    else:
        fields["age"] = None

    # neonatal
    if years is None and months is not None:
        fields["neonatal"] = True
    else: 
        fields["neonatal"] = False

    # medical history
    recent_match = re.search(
        r'Pathological proximate[:\s]+(.*?)(?=Remote pathology\s*:)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["medical_history"] = recent_match.group(1).strip() if recent_match else ""

    # remote medical history
    recent_match = re.search(
        r'Remote pathology[:\s]+(.*?)(?=Objective examination\s*:)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["remote_medical_history"] = recent_match.group(1).strip() if recent_match else ""

    # objective examination
    exam_match = re.search(
        r'Objective examination[:\s]+(.*?)(?=Clinical Diary\s*:)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["exams"] = exam_match.group(1).strip() if exam_match else ""

    # clinical diary
    diary_match = re.search(
        r'(?:Clinical record|Clinical diary)[:\s]+(.*?)(?=Diagnosis\s*:)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["clinical_diary"] = diary_match.group(1).strip() if diary_match else ""

    # diagnosis
    diag_match = re.search(r'Diagnosis[:\s]+(.*?)(?=Prognosis\s*:)', translated_note, re.IGNORECASE | re.DOTALL)
    fields["diagnosis"] = diag_match.group(1).strip() if diag_match else ""

    # prognosis
    prog_match = re.search(r'Prognosis[:\s]+(.+?)$', translated_note, re.IGNORECASE | re.DOTALL)
    fields["prognosis"] = prog_match.group(1).strip() if prog_match else ""

    return fields
