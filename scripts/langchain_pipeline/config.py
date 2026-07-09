from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

MODEL_NAME = str(Path("~/RecallNet/models/Llama-3.1-8B-Instruct").expanduser())
MODEL_NAME_LIGHT = "llama3.1:8b"
BATCH_SIZE = 10
MAX_CONCURRENCY = 1
STRUCTURED_OUTPUT_METHOD = "json_mode"
SYSTEM_PROMPT_PATH = PACKAGE_DIR / "prompt_v2.txt"
PREDICATE_REG_PATH = PACKAGE_DIR / "predicate_registry.json"
