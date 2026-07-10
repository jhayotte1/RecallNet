import time
import pandas as pd
from pathlib import Path

from langchain_pipeline.batching import make_batches
from langchain_pipeline.classify import classify_batches
from langchain_pipeline.classify import build_prompt
from langchain_pipeline.config import BATCH_SIZE, SYSTEM_PROMPT_PATH, MODEL_NAME_LIGHT

SYSTEM_PROMPT = Path(SYSTEM_PROMPT_PATH).read_text()
DATA_DIR  = Path(__file__).parent.parent / "data" / "sample_data"
RESULT_DIR = Path(__file__).parent.parent / "results" / MODEL_NAME_LIGHT / "scoring_exp" 

#####

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
        for idx_str, evaluation in result["evaluations"].items():
            idx = int(idx_str)
            s, p, o = batch[idx]
            rows.append({
                "subject": s,
                "predicate": p,
                "object": o,
                "meaningfulness": evaluation["meaningfulness"],
                "typicality": evaluation["typicality"],
                "saliency": evaluation["saliency"],
                "reason": evaluation["reasoning"],
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
        f.write(f"Predicate evaluated: {pred}")
        f.write(f"Sample size: {sample_size}\n")
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
    print(f"Saved to {out_dir}")

    return out_df

if __name__ == "__main__":
    pred = "used for"
    pred_parsed = pred.strip().replace(" ", "")

    print(f"Loading quasi_sample_{pred_parsed}.csv")
    df_sample = pd.read_csv(DATA_DIR / f"quasi_sample_{pred_parsed}.csv")

    results = run_experiment(
        df=df_sample,
        experiment_name="exp01_LG",
        experiment_desc="Langchain Pipeline, Scoring 3 metrics : Meaningfulness/Typicality/Saliency, single predicate evaluation, 4 bits model loading",
        pred=pred,
        sample_size=100,
    )
