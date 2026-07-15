import json
import time
from pathlib import Path
from langchain.chat_models import init_chat_model

from .output import BatchEvaluation
from .config import MODEL_NAME, MODEL_PROVIDER, SYSTEM_PROMPT_PATH, MAX_CONCURRENCY, PREDICATE_REG_PATH, TEMPERATURE
from .batching import format_batch


SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()

with open(PREDICATE_REG_PATH, "r") as f:
    PREDICATE_REG = json.load(f)

def build_classifier():
    model_chat = init_chat_model(
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=TEMPERATURE
    )
    return model_chat.with_structured_output(BatchEvaluation, method="json_schema")

def format_scoring_examples(scoring_examples: list) -> str:
    lines = ["## Scoring examples for this predicate\n"]
    for i, ex in enumerate(scoring_examples):
        lines.append(f"**Example {i+1}**: {ex['triple']}")
        lines.append(f"- Meaningfulness: {ex['meaningfulness']}")
        lines.append(f"- Typicality: {ex['typicality']}")
        lines.append(f"- Saliency: {ex['saliency']}")
        lines.append(f"- Reasoning: {ex['reasoning']}\n")
    return "\n".join(lines)

def build_prompt(predicate: str):
    pred_dic = PREDICATE_REG[predicate]
    pred_def = pred_dic["definition"]
    pred_ex = pred_dic["example"]
    scoring_ex = pred_dic["scoring_examples"]
    sys_prompt = SYSTEM_PROMPT
    return sys_prompt.format(
        predicate_name = predicate,
        predicate_definition = pred_def,
        predicate_example = pred_ex,
        scoring_examples_block = format_scoring_examples(scoring_ex)
    )


def build_message(batch_str: str, sys_prompt: str):
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Evaluate this batch:\n{batch_str}"}
    ]

def classify_batches(batches, predicate: str):
    print("Building classifier")
    classifier = build_classifier()
    print("Building prompt")
    sys_prompt = build_prompt(predicate)
    all_messages = [build_message(format_batch(b), sys_prompt) for b in batches]
    start_time = time.time()
    print("Start inference")
    return classifier.batch(all_messages, config={'max_concurrency': MAX_CONCURRENCY}), start_time