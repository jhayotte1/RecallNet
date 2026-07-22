import time
import argparse
import pandas as pd
from pathlib import Path

from langchain_pipeline.batching import make_batches
from langchain_pipeline.classify import classify_batches
from langchain_pipeline.classify import build_prompt
from langchain_pipeline.config import BATCH_SIZE, SYSTEM_PROMPT_PATH, MODEL_NAME_LIGHT

SYSTEM_PROMPT = Path(SYSTEM_PROMPT_PATH).read_text()
DATA_DIR  = Path(__file__).parent.parent / "data"
RESULT_DIR = Path(__file__).parent.parent / "results" / MODEL_NAME_LIGHT / "final_process" 

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
    parser = argparse.ArgumentParser(description="RecallNet scoring pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory name")
    parser.add_argument("--exp-name", type=str, default="exp00_LG", help="Experiment name")
    parser.add_argument("--exp-desc", type=str, default="Default experiment", help="Experiment description")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--predicates", nargs="+", default=PREDICATE_LIST, help="Liste des prédicats à traiter")
    return parser.parse_args()

def run_experiment(df: pd.DataFrame, experiment_name: str, experiment_desc: str, sample_size: int=100, pred: str=""):
    pred_parsed = pred.strip().replace(" ", "").lower()
    
    triple_list = list(zip(df['subject'], df['predicate'], df['object']))

    print("Making batches")
    batches = make_batches(triple_list, size=BATCH_SIZE)

    results, start_time = classify_batches(batches, predicate=pred)
    total_time = time.time() - start_time
    print("Inference finished")
    avg_time = total_time / len(df)

    rows = []
    errors = 0
    for batch, result in zip(batches, results):
        if result is None:
            errors += len(batch)
            continue
        for idx_str, evaluation in result.evaluations.items():
            idx = int(idx_str)
            s, p, o = batch[idx]
            rows.append({
                "subject": s,
                "predicate": p,
                "object": o,
                "meaningfulness": evaluation.meaningfulness,
                "typicality": evaluation.typicality,
                "saliency": evaluation.saliency,
                "reason": evaluation.reasoning,
            })
    out_df = pd.DataFrame(rows)

    out_dir = RESULT_DIR/experiment_name/pred_parsed
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / f"pred_{MODEL_NAME_LIGHT}_{pred_parsed}.csv", index=False)

    metrics_summary = {}
    for metric in ["meaningfulness", "typicality", "saliency"]:
        metrics_summary[metric] = {
            "mean": out_df[metric].mean(),
            "distribution": out_df[metric].value_counts().sort_index().to_dict()
        }

    formatted_prompt = build_prompt(pred)

    with open(out_dir / f"{experiment_name}_config.txt", "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: {MODEL_NAME_LIGHT}\n")
        f.write(f"Experiment description: {experiment_desc}\n")
        f.write(f"Predicate evaluated: {pred}\n")
        f.write(f"Sample size: {sample_size if sample_size else 'full dataset'}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f}min)\n")
        f.write(f"Avg time per triplet: {avg_time:.2f}s\n")
        f.write(f"Errors: {errors}\n")
        for metric, stats in metrics_summary.items():
            f.write(f"\n{metric}:\n")
            f.write(f"  mean: {stats['mean']:.2f}\n")
            f.write(f"  distribution: {stats['distribution']}\n")        
        f.write(f"\n{'='*50}\n")
        f.write(f"PROMPT:\n\n{formatted_prompt}\n")
    
    print(f"\n=== {experiment_name} ({MODEL_NAME_LIGHT}) ===")
    print(f"Total: {total_time:.1f}s | Avg: {avg_time:.2f}s/triplet")
    print(f"Saved to {out_dir}\n\n")

    return out_df

if __name__ == "__main__":
    args = parse_args()
    predicate_list = args.predicates
    data_dir = Path(DATA_DIR / args.data_dir)
    for pred in predicate_list:
        pred_parsed = pred.strip().replace(" ", "")
        try: 
            print(f"Loading quasi_sample_{pred_parsed}.csv")
            df_sample = pd.read_csv(f"{data_dir}/quasi_top5000000_{pred_parsed}.csv")
            df_sample.columns = df_sample.columns.str.strip()

            results = run_experiment(
                df=df_sample,
                experiment_name=args.exp_name,
                experiment_desc=args.exp_desc,
                pred=pred,
                sample_size=args.sample_size,
            )
            del df_sample
        except Exception as e:
            print(f"Error on predicate'{pred} : {e}\n")
            continue