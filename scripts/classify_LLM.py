import ollama
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

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

## Examples

(dog, CapableOf, barking)
Reasoning: Dogs do bark, correct use of CapableOf.
Label: VALID

(car, HasPrerequisite, fuel)
Reasoning: A car requires fuel to operate, correct use of HasPrerequisite.
Label: VALID

(banana, UsedFor, driving a car)
Reasoning: Bananas have no connection to driving.
Label: NOISY

(dog, IsA, dog)
Reasoning: Tautology, conveys no information.
Label: NOISY

(thing, HasProperty, quality)
Reasoning: Both subject and object are too vague to be meaningful.
Label: NOISY

(cat, IsA, dog)
Reasoning: Cat and dog are related (both pets/mammals), but a cat is not a type of dog — wrong predicate.
Label: TOCHANGE

(Paris, AtLocation, Germany)
Reasoning: Paris and Germany are geographically related, but Paris is not located in Germany.
Label: TOCHANGE

(piano, MadeOf, music)
Reasoning: Piano and music are related, but a piano is not composed of music — wrong predicate.
Label: TOCHANGE

Respond with ONLY two lines:
Reasoning: <Only One sentence maximum>
Label: <VALID|NOISY|TOCHANGE>
"""

SCORE = """
You are also given the triplet's confidence score (FinalScore), which aggregates source corpus frequency and generation rank.
Higher scores indicate stronger support from the original data.
This score is supplementary context only — always judge the triplet's semantic validity first based on the content alone.
"""



def classify(model, row, score=False):
    subject, predicate, obj = row['subject'], row['predicate'], row['object']
    user_prompt = f"({subject}, {predicate}, {obj})"
    system_prompt = SYSTEM_PROMPT
    if score:
        user_prompt += f" [FinalScore: {row['frequency']:.2f}]"
        system_prompt += SCORE

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        options={'temperature': 0}
    )

    raw = response['message']['content']

    label = "ERROR"
    reasoning = ""
    for line in raw.strip().split('\n'):
        if line.strip().startswith("Label:"):
            label = line.split("Label:")[-1].strip()
        elif line.strip().startswith("Reasoning:"):
            reasoning = line.split("Reasoning:")[-1].strip()

    if label not in ("VALID", "NOISY", "TOCHANGE"):
        label = "ERROR"

    return label, reasoning


def run_experiment(df, model, experiment_name, experiment_desc, score=False, sample_size=100):
    records = []
    start_time = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        label, reasoning = classify(model, row, score=score)
        if score:
            records.append({
                'subject': row['subject'],
                'predicate': row['predicate'],
                'object': row['object'],
                'label': label,
                'reasoning': reasoning,
                'score': row['frequency']
            })
        else:
            records.append({
                'subject': row['subject'],
                'predicate': row['predicate'],
                'object': row['object'],
                'label': label,
                'reasoning': reasoning,
            })

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
    df_sample = pd.read_csv(DATA_DIR / "quasi_test_100_sample.csv")

    results = run_experiment(
        df=df_sample,
        model="llama3.1:8b",
        experiment_name="exp01",
        experiment_desc="Strict VALID prompt, no batching, ConceptNet Predicate Description",
        score=False,
        sample_size=100
    )