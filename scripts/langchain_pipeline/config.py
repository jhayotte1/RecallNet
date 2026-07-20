import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

MODEL_NAME = os.environ.get("RECALLNET_MODEL", "llama3.1:8b")
MODEL_PROVIDER = os.environ.get("RECALLNET_PROVIDER", "ollama")
MODEL_NAME_LIGHT = os.environ.get("RECALLNET_MODEL_LIGHT", MODEL_NAME)
BATCH_SIZE = int(os.environ.get("RECALLNET_BATCH_SIZE", "10"))
MAX_CONCURRENCY = int(os.environ.get("RECALLNET_MAX_CONCURRENCY", "1"))
TEMPERATURE = float(os.environ.get("RECALLNET_TEMPERATURE", "0"))
SYSTEM_PROMPT_PATH = PACKAGE_DIR / os.environ.get("RECALLNET_PROMPT", "prompt.txt")
PREDICATE_REG_PATH = PACKAGE_DIR / os.environ.get("RECALLNET_REGISTRY", "predicate_registry.json")
