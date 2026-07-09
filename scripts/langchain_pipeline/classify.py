import json
import time
from pathlib import Path
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig, BitsAndBytesConfig

from .output import BatchEvaluation
from .config import MODEL_NAME, SYSTEM_PROMPT_PATH, MAX_CONCURRENCY, PREDICATE_REG_PATH
from .batching import format_batch

# transformers.logging.set_verbosity_error()

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()

with open(PREDICATE_REG_PATH, "r") as f:
    PREDICATE_REG = json.load(f)

def build_classifier():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="float16",
        device_map="auto",
    )
    model.generation_config = GenerationConfig(
        do_sample=False,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=2048
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model.with_structured_output(BatchEvaluation, method="json_mode")

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
    print("Building classifier")
    classifier = build_classifier()
    print("Building prompt")
    sys_prompt = build_prompt(predicate)
    all_messages = [build_message(format_batch(b), sys_prompt) for b in batches]
    start_time = time.time()
    print("Start inference")
    return classifier.batch(all_messages, config={'max_concurrency': MAX_CONCURRENCY}), start_time