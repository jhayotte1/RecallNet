import json
import time
import pandas as pd
from pathlib import Path

from langchain_pipeline.batching import make_batches
from langchain_pipeline.classify import classify_batches
from langchain_pipeline.config import BATCH_SIZE, MODEL_NAME, SYSTEM_PROMPT_PATH

SYSTEM_PROMPT = Path(SYSTEM_PROMPT_PATH).read_text()
DATA_DIR  = Path(__file__).parent.parent / "data"
RESULT_DIR = Path(__file__).parent.parent / "results"

def run_experiment(df: pd.DataFrame, experiment_name: str, experiment_desc: str, sample_size: int=100):
    triple_list = list(zip(df['subject'], df['predicate'], df['object']))

    batches = make_batches(triple_list, size=BATCH_SIZE)

    start = time.time()
    results = classify_batches(batches)
    total_time = time.time() - start
    avg_time = total_time / len(df)

    rows = []
    errors = 0
    for batch, result in zip(batches, results):
        if result is None:
            errors += len(batch)
            continue
        for idx_str, classification in result.classifications.items():
            idx = int(idx_str)
            s, p, o = batch[idx]
            rows.append({
                "subject": s,
                "predicate": p,
                "object": o,
                "label": classification.label,
                "reason": classification.reasoning,
            })
    out_df = pd.DataFrame(rows)

    out_dir = Path(f"{RESULT_DIR}/{experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / f"pred_{MODEL_NAME}.csv", index=False)

    distribution = out_df["label"].value_counts().to_dict()

    with open(out_dir / f"{experiment_name}_config.txt", "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Experiment description: {experiment_desc}\n")
        f.write(f"Sample size: {sample_size}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f}min)\n")
        f.write(f"Avg time per triplet: {avg_time:.2f}s\n")
        f.write(f"Distribution: {distribution}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"\n{'='*50}\n")
        f.write(f"PROMPT:\n\n{SYSTEM_PROMPT}\n")
    
    print(f"\n=== {experiment_name} ({MODEL_NAME}) ===")
    print(out_df['label'].value_counts())
    print(f"Total: {total_time:.1f}s | Avg: {avg_time:.2f}s/triplet")
    print(f"Saved to {out_dir}")

    return out_df

if __name__ == "__main__":
    df_sample = pd.read_csv(DATA_DIR / "quasi_test_100_sample.csv")

    results = run_experiment(
        df=df_sample,
        experiment_name="exp04_batched",
        experiment_desc="Langchain Pipeline, More detailed label description, batched 10, ConceptNet Predicate Description, Binary classification : VALID/NOISY",
        sample_size=100,
    )

