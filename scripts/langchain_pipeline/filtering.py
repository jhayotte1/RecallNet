import json
import time
from pathlib import Path
from langchain.chat_models import init_chat_model

from .output import BatchFilterDecision
from .config import MODEL_NAME, MODEL_PROVIDER, SYSTEM_PROMPT_PATH, MAX_CONCURRENCY, PREDICATE_REG_PATH, TEMPERATURE
from .classify import build_message


FILTER_PROMPT = SYSTEM_PROMPT_PATH.read_text()

with open(PREDICATE_REG_PATH, "r") as f:
    PREDICATE_REG = json.load(f)

def format_filter_batch(batch):
    return"\n".join(
        f"{i}: ({s}, {p}, {o})" for i, (s, p, o, *_) in enumerate(batch)
    )

def build_reviewer():
    model_chat = init_chat_model(
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=TEMPERATURE
    )
    return model_chat.with_structured_output(BatchFilterDecision, method="json_schema")

def build_filtering_prompt(predicate: str):
    pred_dic = PREDICATE_REG[predicate]
    return FILTER_PROMPT.format(
        predicate_name=predicate,
        predicate_definition=pred_dic["definition"],
        predicate_scope=pred_dic["scope"],
        predicate_example=pred_dic["example"],
    )

def review_batches(batches, predicate: str):
    reviewer = build_reviewer()
    sys_prompt = build_filtering_prompt(predicate)
    all_messages = [
        build_message(format_filter_batch(b), sys_prompt)
        for b in batches
    ]
    start_time = time.time()
    return reviewer.batch(all_messages, config={"max_concurrency": MAX_CONCURRENCY}), start_time