from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME_LIGHT = "llama3.1:8b"
BATCH_SIZE = 10
MAX_CONCURRENCY = 1
STRUCTURED_OUTPUT_METHOD = "json_mode"
SYSTEM_PROMPT_PATH = PACKAGE_DIR / "prompt.txt"
PREDICATE_REG_PATH = PACKAGE_DIR / "predicate_registry.json"
