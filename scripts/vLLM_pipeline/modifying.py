import json
import time

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from .output import BatchModification
from .batching import format_batch
from .config import (
    SYSTEM_PROMPT_PATH, PREDICATE_REG_PATH, TEMPERATURE,
    VLLM_MODEL, QUANTIZATION, GPU_UTIL, MAX_MODEL_LEN, MAX_TOKENS, MAX_NUM_BATCHED_TOKENS, MAX_NUM_SEQS
)

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()

with open(PREDICATE_REG_PATH, "r") as f:
    PREDICATE_REG = json.load(f)


def build_prompt(predicate: str):
    pred_dic = PREDICATE_REG[predicate]
    return SYSTEM_PROMPT.format(
        predicate_name=predicate,
        predicate_definition=pred_dic["definition"],
        predicate_scope=pred_dic["scope"],
        predicate_example=pred_dic["example"],
    )


def build_message(batch_str: str, sys_prompt: str):
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Evaluate this batch:\n{batch_str}"},
    ]


_LLM = None
_SAMPLING = None


def build_classifier():
    global _LLM, _SAMPLING
    if _LLM is None:
        print(f"Loading vLLM model {VLLM_MODEL} (quant={QUANTIZATION})...")
        _LLM = LLM(
            model=VLLM_MODEL,
            quantization=QUANTIZATION,
            gpu_memory_utilization=GPU_UTIL,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
        )
        _SAMPLING = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            structured_outputs=StructuredOutputsParams(
                json=BatchFilterDecision.model_json_schema()),
        )
    return _LLM, _SAMPLING


def classify_batches(batches, predicate: str):
    print("Building classifier")
    llm, sampling = build_classifier()
    print("Building prompt")
    sys_prompt = build_prompt(predicate)
    all_messages = [build_message(format_batch(b), sys_prompt) for b in batches]

    start_time = time.time()                
    print("Start inference")
    outputs = llm.chat(all_messages, sampling)

    results = []
    for o in outputs:
        try:
            results.append(
                BatchFilterDecision.model_validate_json(o.outputs[0].text))
        except Exception:
            results.append(None)
    return results, start_time