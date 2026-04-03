EMBEDDING_MODEL = "lokeshch19/ModernPubMedBERT"
CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"
NER_MODEL = "Clinical-AI-Apollo/Medical-NER"

# TEST=True skips LLM generation, used to test retrieval quality.
TEST = True

# Output LLM models
QUANT = True
MODEL_QUANT = "Mistral-7B-Instruct-v0.3-Q6_K.gguf"
REPO = "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
MODEL_NAME = "meta-llama/Llama-3.2-3B"

# Retrieval and generation parameters
RETRIEVAL_K = 10
FINAL_N = 10
TEMPERATURE = 0.2

# DeepL API parameters
DEEPL_AUTH_KEY = ""
DEEPL_GLOSSARY_ID = "627f273f-d457-400f-a171-d04b9c13ddf3"