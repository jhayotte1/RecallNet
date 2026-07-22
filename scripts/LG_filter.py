import time
import argparse
import pandas as pd
from pathlib import Path

from langchain_pipeline.batching import make_batches
from langchain_pipeline.filtering import review_batches, build_filtering_prompt
from langchain_pipeline.config import BATCH_SIZE, SYSTEM_PROMPT_PATH, MODEL_NAME_LIGHT, MODEL_NAME

SYSTEM_PROMPT = Path(SYSTEM_PROMPT_PATH).read_text()
RESULT_DIR = Path(__file__).parent.parent / "results" / MODEL_NAME_LIGHT / "scoring_exp"


PREDICATE_LIST = [
        "at location",
        "capable of",
        "causes",
        "cause desire",
        "created by",
        "defined as",
        "desires",
        "distinct from",
        "has a",
        "has subevent",
        "has first subevent",
        "has last subevent",
        "has prerequisite",
        "has property",
        "made of",
        "manner of",
        "motivated by goal",
        "part of",
        "receives action",
        "used for"
    ]

def parse_args():
    parser = argparse.ArgumentParser(description="RecallNet second filtering pipeline")
    parser.add_argument("--split-dir", type=str, required=True, help="Directory containing the split of CSVs (ex: 0_SPLITED, 1_SPLITED, ...)")
    parser.add_argument("--exp-name", type=str, default="exp00", help="Experiment name, name of the directory")
    parser.add_argument("--exp-desc", type=str, default="Default filtering experiment", help="Experiment description")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--predicates", nargs="+", default=PREDICATE_LIST, help="Predicate list to be processed")
    return parser.parse_args()

def run_experiment(df: pd.DataFrame, experiment_name: str, experiment_desc: str, sample_size: int=None, pred: str="", in_dir: str=""):
    pred_parsed = pred.strip().replace(" ", "").lower()

    if df.empty:
        print(f"Empty dataframe for '{pred}', skipping")
        return None

    triple_list = list(zip(df['subject'], df['predicate'], df['object']))

    print("Making batches")
    batches = make_batches(triple_list, size=BATCH_SIZE)

    results, start_time = review_batches(batches, predicate=pred)
    total_time = time.time() - start_time
    print("Filtering finished")
    avg_time = total_time / len(df)

    rows = []
    errors = 0
    for batch, result in zip(batches, results):
        if result is None:
            errors += len(batch)
            continue
        for idx_str, decision in result.filterdecision.items():
            idx = int(idx_str)
            s, p, o, *_ = batch[idx]
            rows.append({
                "subject": s,
                "predicate": p,
                "object": o,
                "decision": decision.decision,
                "reason": decision.reasoning,
            })
    out_df = pd.DataFrame(rows)

    out_dir = RESULT_DIR / experiment_name / "0_FILTERED"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / f"filt_{MODEL_NAME_LIGHT}_{pred_parsed}.csv", index=False)

    decision_dist = out_df["decision"].value_counts().to_dict() if not out_df.empty else {}

    formatted_prompt = build_filtering_prompt(pred)

    with open(out_dir / f"{experiment_name}_config.txt", "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Experiment description: {experiment_desc}\n")
        f.write(f"From data dir : {in_dir}\n")
        f.write(f"Predicate evaluated: {pred}\n")
        f.write(f"Sample size: {sample_size if sample_size else 'full dataset'}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f}min)\n")
        f.write(f"Avg time per triplet: {avg_time:.2f}s\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"\nDecision distribution: {decision_dist}\n")
        f.write(f"\n{'='*50}\n")
        f.write(f"PROMPT:\n\n{formatted_prompt}\n")

    print(f"\n=== {experiment_name} ({MODEL_NAME}) ===")
    print(f"Total: {total_time:.1f}s | Avg: {avg_time:.2f}s/triplet")
    print(f"Decisions: {decision_dist}")
    print(f"Saved to {out_dir}\n\n")

    return out_df

if __name__ == "__main__":
    args = parse_args()
    predicate_list = args.predicates
    data_dir = Path(RESULT_DIR / args.exp_name / args.split_dir / "in_between")
    for pred in predicate_list:
        pred_parsed = pred.strip().replace(" ", "")
        try:
            print(f"Loading in_between_{pred_parsed}.csv")
            df_sample = pd.read_csv(f"{data_dir}/pred_{MODEL_NAME_LIGHT}_{pred_parsed}.csv")
            df_sample.columns = df_sample.columns.str.strip()

            results = run_experiment(
                df=df_sample,
                experiment_name=args.exp_name,
                experiment_desc=args.exp_desc,
                pred=pred,
                sample_size=args.sample_size,
                in_dir=data_dir,
            )
            del df_sample
        except Exception as e:
            print(f"Error on predicate '{pred}' : {e}\n")
            continue