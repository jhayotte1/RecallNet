from pathlib import Path
from langchain.chat_models import init_chat_model

from .output import BatchClassification
from .config import MODEL_NAME, MODEL_PROVIDER, TEMPERATURE, SYSTEM_PROMPT_PATH, MAX_CONCURRENCY
from .batching import format_batch

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()

def build_classifier():
    model = init_chat_model(
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=TEMPERATURE
    )
    return model.with_structured_output(BatchClassification)

def build_message(batch_str: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this batch:\n{batch_str}"}
    ]

def classify_batches(batches):
    classifier = build_classifier()
    all_messages = [build_message(format_batch(b)) for b in batches]
    return classifier.batch(all_messages, config={'max_concurrency': MAX_CONCURRENCY})