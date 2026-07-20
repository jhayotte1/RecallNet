import time
import argparse
import pandas as pd
from pathlib import Path

from langchain_pipeline.batching import make_batches
from langchain_pipeline.review import review_batches
from langchain_pipeline.config import BATCH_SIZE

RESULTS_DIR = Path(__file__).parent.parent / "results"

PRED_MAP = {
        "atlocation": "at location",
        "capableof": "capable of",
        "causes": "causes",
        "causedesire": "cause desire",
        "createdby": "created by",
        "definedas": "defined as",
        "desires": "desires",
        "distinctfrom": "distinct from",
        "hasa": "has a",
        "hassubevent": "has subevent",
        "hasfirstsubevent": "has first subevent",
        "haslastsubevent": "has last subevent",
        "hasprerequisite": "has prerequisite",
        "hasproperty": "has property",
        "madeof": "made of",
        "mannerof": "manner of",
        "motivatedbygoal": "motivated by goal",
        "partof": "part of",
        "receivesaction": "receives action",
        "usedfor": "used for"
    }

def parse_args():
    parser = argparse.ArgumentParser(description="RecallNet scoring reviewing for rule mining")
    parser.add_argument("--exp-name", type=str, default="exp00_LG", help="Experiment name")
    parser.add_argument("--model-reviewed", type=str, default="llama3.1:8b", help="Name of model used for global scoring")
    return parser.parse_args()

def load_scored_csvs(sampling_dir: Path, model_reviewed: str) -> dict[str, pd.DataFrame]:
    csvs = sampling_dir.glob("sample_pred_*.csv")
    result = {}
    for csv_path in csvs:
        pred = csv_path.stem.replace(f"sample_pred_{model_reviewed}_", "")
        result[pred] = pd.read_csv(csv_path)
    return result

def run_review(df: pd.DataFrame, pred: str, output_dir: Path):
    df.columns.str.strip()
    # Build tuples with scores
    triples_with_scores = list(zip(
        df["subject"], df["predicate"], df["object"],
        df["meaningfulness"], df["typicality"], df["saliency"], df["reason"],
    ))

    print("Making batches...")
    batches = make_batches(triples_with_scores, size=BATCH_SIZE)
    print("Reviewing batches...")
    results, start_time = review_batches(batches, predicate=pred)
    total_time = time.time() - start_time
    print("Reviewing done")

    rows = []
    errors = 0
    for batch, result in zip(batches, results):
        if result is None:
            errors += len(batch)
            continue
        for idx_str, review in result.reviews.items():
            idx = int(idx_str)
            s, p, o, m, t, sa, r = batch[idx]
            rows.append({
                "subject": s, "predicate": p, "object": o,
                "verdict": review.verdict,
                "reason_review": review.reasoning,
                "meaningfulness": m, "typicality": t, "saliency": sa,
                "reason_8b": r,
            })

    out_df = pd.DataFrame(rows)
    out_dir = output_dir / pred
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / f"review_{pred}.csv", index=False)

    # Stats
    verdicts = out_df["verdict"].value_counts()
    print(f"\n=== {pred} ===")
    print(f"Total time: {total_time:.1f}s ; {total_time:.1f}s")
    print(f"KEEP: {verdicts.get('KEEP', 0)} | REJECT: {verdicts.get('REJECT', 0)} | UNCERTAIN: {verdicts.get('UNCERTAIN', 0)}")
    print(f"Errors: {errors}")

    return out_df

if __name__=="__main__":
    args = parse_args()
    input_dir = Path(RESULTS_DIR, f"{args.model_reviewed}/scoring_exp/{args.exp_name}/0_SAMPLING")
    output_dir = Path(RESULTS_DIR, f"{args.model_reviewed}/scoring_exp/{args.exp_name}/1_REVIEW")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Load scored csvs...")
    scored_data = load_scored_csvs(input_dir, args.model_reviewed)
    for pred_key, df in scored_data.items():
        pred_name = PRED_MAP.get(pred_key, pred_key)
        try:
            print(f"Review for {pred_name}")
            run_review(df, pred_name, output_dir)
            print("\n")
        except Exception as e:
            print(f"Error on '{pred_name}': {e}")
            continue
