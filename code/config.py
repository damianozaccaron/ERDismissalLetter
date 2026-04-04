EMBEDDING_MODEL = "lokeshch19/ModernPubMedBERT"
CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"
NER_MODEL = "Clinical-AI-Apollo/Medical-NER"

# TEST=True skips LLM generation, used to test retrieval quality.
TEST = False

# Output LLM models
API = True
QUANT = True # if using local models, whether to use quantized versions from llama-cpp (if False, uses non-quantized models via transformers library)
REPO = "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
MODEL_NAME = "qwen/qwen3.6-plus:free"

# Retrieval and generation parameters
RETRIEVAL_K = 10
FINAL_N = 10
TEMPERATURE = 0.2

# Tokens
OPENROUTER_KEY = "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"
DEEPL_AUTH_KEY = "3e8dba21-6f6c-4f49-9d7f-9fea17bd3f3f:fx"
DEEPL_GLOSSARY_ID = "627f273f-d457-400f-a171-d04b9c13ddf3"