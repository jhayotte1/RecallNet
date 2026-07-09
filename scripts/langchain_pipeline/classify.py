import json
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from .output import BatchEvaluation
from .config import MODEL_NAME, SYSTEM_PROMPT_PATH, MAX_CONCURRENCY, PREDICATE_REG_PATH
from .batching import format_batch

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()

with open(PREDICATE_REG_PATH, "r") as f:
    PREDICATE_REG = json.load(f)

def build_classifier():
    llm = HuggingFacePipeline.from_model_id(
        model_id=MODEL_NAME,
        task="text-generation",
        devide_map="auto",
        pipeline_kwargs=dict(
            max_new_tokens=2048,
            do_sample=False,
        ),
    )
    model = ChatHuggingFace(llm=llm)
    return model.with_structured_output(BatchEvaluation)

def build_prompt(predicate: str):
    pred_dic = PREDICATE_REG[predicate]
    pred_def = pred_dic["definition"]
    pred_ex = pred_dic["example"]
    scoring_ex_dic = pred_dic["scoring_examples"]
    sys_prompt = SYSTEM_PROMPT
    return sys_prompt.format(
        predicate_name = predicate,
        predicate_definition = pred_def,
        predicate_example = pred_ex,
        scoring_examples_block = ""
    )


def build_message(batch_str: str, sys_prompt: str):
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Evaluate this batch:\n{batch_str}"}
    ]

def classify_batches(batches, predicate: str):
    classifier = build_classifier()
    sys_prompt = build_prompt(predicate)
    all_messages = [build_message(format_batch(b), sys_prompt) for b in batches]
    return classifier.batch(all_messages, config={'max_concurrency': MAX_CONCURRENCY})