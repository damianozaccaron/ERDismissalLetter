# Load model directly
from transformers import AutoTokenizer, AutoModel

import torch
import deepl

def load_translator(model_name: str = "facebook/seamless-m4t-v2-large"):
    
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

def deepl_translation(text, auth_key = "3e8dba21-6f6c-4f49-9d7f-9fea17bd3f3f:fx"):

    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(text, source_lang="IT", target_lang="EN-GB")

    return result.text

import time

if __name__ == "__main__":
    start_time = time.time()
    print("Loading Translation Model...")

    input = """MASCHIO, 42 AA
            Anamnesi:
            - Patologica prossima:
            Il paziente accede per dolore toracico retrosternale in posizione supina presente da circa 3 giorni,
            non esacerbato dal respiro o dal movimento. Lamenta tosse secca da diverse settimane. Nega febbre
            o dispnea.
            - Patologica remota:
            Non noti eventi cardiovascolari ne familiarità.
            Non affetto da diabete.
            Normopeso.
            Ex fumatore di circa mezzo pacchetto al giorno, STOP dieci anni fa.
            Pregressa infezione da helicobacter pilori, eradicato circa 5 anni fa. Per il resto no patologie note.
            Nega farmacoallergie.
            Esame obiettivo:
            Murmure vescicolare presente in tutti gli ambiti polmonari, non rumori aggiunti.
            Toni cardiaci normofrequenti, ritmici e con pause libere.
            Addome trattabile, non dolente alla palpazione superficiale o profonda. Giordano e Blumberg
            negativi. Indolenti i punti cistici.
            Non edemi declivi o segni di trombosi venosa.
            Diario clinico:
            - Reperito accesso venoso ed esecuzione di troponina più biochimica, emocromo e pro-BNP. Si
            esegue ECG: ritmo sinusale, non segni sotto/sopraslivellamento del tratto ST.
            - Si esegue EGA arterioso: scambi fisiologici
            - Si invia il paziente in radiologia per esecuzione RX torace in 2 proiezioni.
            - Si visionano gli esami: troponina <14 pertanto negativa. Non indici di flogosi. RX torace negativo
            per acuzie.
            - Si somministrano 40mg di pantoprazolo in 100cc soluzione fisiologica.
            - Si esegue eco bedside: non versamento pleuro-pericardico o endoaddominale. Non linee B nei
            campi polmonari, sliding pleurico valido. Colecisti normodistesa, non formazioni litiasiche
            osservabili con tale metodica. Non idronefrosi bilateralmente.
            - A distanza di un’ora riduzione fino a completa risoluzione del quadro sintomatologico. Si dimette
            a domicilio.
            Diagnosi: MALATTIA DA REFLUSSO GASTROESOFAGEO"""

    print("Translating Query...")
    tokenizer, model = load_translator()
    translated = translate_to_english(input, tokenizer=tokenizer, model=model)
    # translated = deepl_translation(text=input)

    end_time = time.time()

    with open('translation.txt', 'w', encoding='utf-8') as file:
        file.write(translated)

    print(f"Translated document in {end_time-start_time:.3f} seconds")
