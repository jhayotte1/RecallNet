import ollama
import json
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
BATCH_SIZE = 10

SYSTEM_PROMPT = """You are a commonsense knowledge evaluator. Your task is to classify triplets from a commonsense knowledge graph as VALID, NOISY, or TOCHANGE.

## Label definitions

- **VALID**: The triplet is a factually correct commonsense statement. The subject, predicate, and object are all meaningful, and the predicate accurately describes the relationship between subject and object.
- **NOISY**: The triplet is false, contradictory, malformed, nonsensical, or too vague to be informative. Also includes tautologies (subject and object are identical or synonymous) and cases where a term is unrecognizable.
- **TOCHANGE**: The subject and object share a genuine, plausible relationship, but the predicate is incorrect. The triplet contains real signal but the relation needs to be fixed.

## Relation definitions

Here are the predicates used in the knowledge graph and what they mean:

- **AtLocation**: A is a typical location where B can be found. (butter, AtLocation, refrigerator)
- **CapableOf**: Something that A can typically do is B. (knife, CapableOf, cut)
- **Causes**: A typically causes B to happen. (exercise, Causes, sweat)
- **CausesDesire**: A makes someone want B. (having no food, CausesDesire, go to a store)
- **CreatedBy**: B is a process or agent that creates A. (cake, CreatedBy, bake)
- **DefinedAs**: A and B overlap in meaning, B is more explanatory. (peace, DefinedAs, absence of war)
- **Desires**: A is a conscious entity that typically wants B. (person, Desires, love)
- **DistinctFrom**: A and B are distinct members of a set; A is not B. (red, DistinctFrom, blue)
- **Entails**: If A is true, B must also be true. (breathe, Entails, be alive) [deprecated — reinterpret as HasPrerequisite or Causes if possible]
- **HasA**: B belongs to A, as an inherent part or possession. (bird, HasA, wing)
- **HasFirstSubevent**: A is an event that begins with subevent B. (sleep, HasFirstSubevent, close eyes)
- **HasLastSubevent**: A is an event that concludes with subevent B. (cook, HasLastSubevent, clean up)
- **HasPrerequisite**: In order for A to happen, B needs to happen. (dream, HasPrerequisite, sleep)
- **HasProperty**: A has B as a property; A can be described as B. (ice, HasProperty, cold)
- **HasSubevent**: A and B are events, B happens as part of A. (eating, HasSubevent, chewing)
- **InstanceOf**: A is a specific instance of B. (Albert Einstein, InstanceOf, physicist) [deprecated — reinterpret as IsA]
- **MadeOf**: A is made of B. (bottle, MadeOf, plastic)
- **MannerOf**: A is a specific way to do B. (sprint, MannerOf, run)
- **MotivatedByGoal**: The action A is done to achieve B. (study, MotivatedByGoal, pass exam)
- **PartOf**: A is a component of B. (wheel, PartOf, car)
- **ReceivesAction**: A undergoes the action B. (food, ReceivesAction, eat)
- **UsedFor**: A is used for the purpose B. (knife, UsedFor, cutting)

When a triplet uses the wrong predicate for a relationship that otherwise makes sense, classify as TOCHANGE.

## Reasoning guidelines

Think step by step before giving your label:
1. Are the subject and object real, meaningful concepts?
2. Does the predicate correctly describe their relationship?
3. If the predicate is wrong, do the subject and object still share a plausible connection?

Classify each triplet in the batch. Respond ONLY with a JSON object where each key is the triplet index and each value is an object with "reasoning" (one sentence max) and "label" (VALID, NOISY, or TOCHANGE).

Example response format:
{"0": {"reasoning": "Dogs do bark, correct use of CapableOf.", "label": "VALID"}, "1": {"reasoning": "Bananas have no connection to driving.", "label": "NOISY"}}
"""

SCORE = """
You are also given each triplet's confidence score (FinalScore), which aggregates source corpus frequency and generation rank.
Higher scores indicate stronger support from the original data.
This score is supplementary context only — always judge the triplet's semantic validity first based on the content alone.
"""


def build_batch_prompt(batch_rows, score=False):
    lines = []
    for i, row in enumerate(batch_rows):
        line = f"{i}. ({row['subject']}, {row['predicate']}, {row['object']})"
        if score:
            line += f" [FinalScore: {row['frequency']:.2f}]"
        lines.append(line)
    return "\n".join(lines)


def classify_batch(model, batch_rows, score=False):
    user_prompt = build_batch_prompt(batch_rows, score=score)
    system_prompt = SYSTEM_PROMPT
    if score:
        system_prompt += SCORE

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        format='json',
        options={'temperature': 0}
    )

    raw = response['message']['content']

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [("ERROR", f"JSON parse failed: {raw[:100]}")] * len(batch_rows)

    results = []
    for i in range(len(batch_rows)):
        entry = parsed.get(str(i), {})
        label = entry.get("label", "ERROR")
        reasoning = entry.get("reasoning", "")
        if label not in ("VALID", "NOISY", "TOCHANGE"):
            label = "ERROR"
        results.append((label, reasoning))

    return results


def run_experiment(df, model, experiment_name, experiment_desc, score=False, sample_size=100, batch_size=BATCH_SIZE):
    records = []
    rows = [row for _, row in df.iterrows()]
    n_batches = (len(rows) + batch_size - 1) // batch_size

    start_time = time.time()

    for b in tqdm(range(n_batches), desc=f"Batches (size={batch_size})"):
        batch = rows[b * batch_size : (b + 1) * batch_size]
        results = classify_batch(model, batch, score=score)

        for row, (label, reasoning) in zip(batch, results):
            record = {
                'subject': row['subject'],
                'predicate': row['predicate'],
                'object': row['object'],
                'label': label,
                'reasoning': reasoning,
            }
            if score:
                record['score'] = row['frequency']
            records.append(record)

    total_time = time.time() - start_time
    avg_time = total_time / len(df)

    results_df = pd.DataFrame(records)

    exp_dir = RESULTS_DIR / model / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(exp_dir / f"pred_{model}_{experiment_name}_{sample_size}.csv", index=False)

    with open(exp_dir / f"{experiment_name}_config.txt", "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Experiment description: {experiment_desc}\n")
        f.write(f"Sample size: {sample_size}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f}min)\n")
        f.write(f"Avg time per triplet: {avg_time:.2f}s\n")
        f.write(f"Distribution: {results_df['label'].value_counts().to_dict()}\n")
        f.write(f"Errors: {int((results_df['label'] == 'ERROR').sum())}\n")
        f.write(f"\n{'='*50}\n")
        if score:
            f.write(f"PROMPT:\n\n{SYSTEM_PROMPT + SCORE}\n")
        else:
            f.write(f"PROMPT:\n\n{SYSTEM_PROMPT}\n")

    print(f"\n=== {experiment_name} ({model}) ===")
    print(results_df['label'].value_counts())
    print(f"Total: {total_time:.1f}s | Avg: {avg_time:.2f}s/triplet")
    print(f"Saved to {exp_dir}")

    return results_df


if __name__ == "__main__":
    df_sample = pd.read_csv(DATA_DIR / "quasimodo_test_sample.csv")

    results = run_experiment(
        df=df_sample,
        model="llama3.1:8b",
        experiment_name="exp01_batched",
        experiment_desc="Strict VALID prompt, batched (15), ConceptNet Predicate Description",
        score=False,
        sample_size=100,
        batch_size=15
    )