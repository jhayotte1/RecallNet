from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

MODEL_NAME = "llama3.1:8b"
MODEL_PROVIDER = "ollama"
TEMPERATURE = 0
BATCH_SIZE = 10
MAX_CONCURRENCY = 1
STRUCTURED_OUTPUT_METHOD = "json_mode"
SYSTEM_PROMPT_PATH = PACKAGE_DIR / "prompt.txt"
