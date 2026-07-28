import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Modele vLLM
# --------------------------------------------------------------------------
# ATTENTION : vLLM charge des poids HuggingFace -> identifiant HF ou chemin
# local, PAS un tag Ollama comme "llama3.1:8b".
VLLM_MODEL = os.environ.get("RECALLNET_VLLM_MODEL",
                            "meta-llama/Llama-3.1-8B-Instruct")

# Nom court et sur pour les chemins de sortie (pas de "/").
MODEL_NAME_LIGHT = os.environ.get("RECALLNET_MODEL_LIGHT", "llama31-8b-fp8")

QUANTIZATION = os.environ.get("RECALLNET_VLLM_QUANT", "fp8")   # 8-bit natif Hopper
GPU_UTIL = float(os.environ.get("RECALLNET_VLLM_GPU_UTIL", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("RECALLNET_VLLM_MAX_LEN", "8192"))

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
BATCH_SIZE = int(os.environ.get("RECALLNET_BATCH_SIZE", "10"))
TEMPERATURE = float(os.environ.get("RECALLNET_TEMPERATURE", "0"))

# Un batch de BATCH_SIZE triples produit BATCH_SIZE evaluations : la sortie
# est longue, max_tokens doit suivre la taille de batch.
MAX_TOKENS = int(os.environ.get("RECALLNET_VLLM_MAX_TOKENS",
                                str(150 * BATCH_SIZE)))

SYSTEM_PROMPT_PATH = PACKAGE_DIR / os.environ.get("RECALLNET_PROMPT", "prompt.txt")
PREDICATE_REG_PATH = PACKAGE_DIR / os.environ.get("RECALLNET_REGISTRY",
                                                  "predicate_registry.json")