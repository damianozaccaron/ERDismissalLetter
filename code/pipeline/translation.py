from config import DEEPL_AUTH_KEY, DEEPL_GLOSSARY_ID
import deepl

def deepl_translation(text, auth_key = DEEPL_AUTH_KEY, glossary = DEEPL_GLOSSARY_ID):

    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(text, source_lang="IT", target_lang="EN-GB", glossary=glossary)

    return result.text

def deepl_translation_en_it(text, auth_key = DEEPL_AUTH_KEY):

    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(text, source_lang="EN", target_lang="IT", formality="prefer_more")

    return result.text

# Instruction for glossary creation can be found in this folder in the file CreatGlossaryDeepL.txt (to use from terminal)
