from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

MODEL_NAME = "llama3.3:70b"
MODEL_PROVIDER = "ollama"
MODEL_NAME_LIGHT = "llama3.3:70b"
BATCH_SIZE = 10
MAX_CONCURRENCY = 1
TEMPERATURE = 0
STRUCTURED_OUTPUT_METHOD = "json_mode"
SYSTEM_PROMPT_PATH = PACKAGE_DIR / "prompt_v2.txt"
PREDICATE_REG_PATH = PACKAGE_DIR / "predicate_registry.json"
