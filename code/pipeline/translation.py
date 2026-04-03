from config import DEEPL_AUTH_KEY, DEEPL_GLOSSARY_ID
import deepl

def deepl_translation(text, auth_key = DEEPL_AUTH_KEY, glossary = DEEPL_GLOSSARY_ID):

    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(text, source_lang="IT", target_lang="EN-GB", glossary=glossary)

    return result.text

def deepl_translation_en_it(text, auth_key = DEEPL_AUTH_KEY):

    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(text, source_lang="EN-GB", target_lang="IT")

    return result.text