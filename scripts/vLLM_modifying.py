import time
import argparse
import pandas as pd
import re
import os
import traceback
from pathlib import Path

from vLLM_pipeline.batching import make_batches
from vLLM_pipeline.modifying import classify_batches
from vLLM_pipeline.modifying import build_prompt
from vLLM_pipeline.config import BATCH_SIZE, SYSTEM_PROMPT_PATH, MODEL_NAME_LIGHT

SYSTEM_PROMPT = Path(SYSTEM_PROMPT_PATH).read_text()
INPUT_DIR = Path("~/RecallNet/src/results/llama3.1:8b-fp8").expanduser()

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
    parser = argparse.ArgumentParser(description="RecallNet modifying triple pipeline (vLLM)")
    parser.add_argument("--dataset-prefix", type=str, default="q", help="Prefix for output files (q=quasimodo, a=ascent)")
    parser.add_argument("--predicates", nargs="+", default=PREDICATE_LIST, help="Predicates to process")
    return parser.parse_args()

def safe_mkdir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, OSError):
        os.stat(path.parent)
        path.mkdir(parents=True, exist_ok=True)

def mod_row(orig, predicate, edit):
    is_mod = edit.verdict == "MODIFIED" and edit.new is not None
    if is_mod:
        destination = "modified"
    elif edit.verdict == "MODIFIED":     # MODIFIED annoncé mais new manquant -> incohérent
        destination = "failed"
    else:                                 # KEEP / REJECT
        destination = edit.verdict.lower()
    return {
        "subject":          edit.new.subject if is_mod else orig["subject"],
        "predicate":        predicate,
        "object":           edit.new.object  if is_mod else orig["object"],
        "destination":      destination,
        "og_subject":       orig["subject"],
        "og_object":        orig["object"],
        "meaningfulness":   orig["meaningfulness"],
        "typicality":       orig["typicality"],
        "saliency":         orig["saliency"],
        "m_reason":         edit.reasoning,
    }

def run_chunk(df: pd.DataFrame, pred: str, pred_parsed: str, chunk_num: int, dataset_prefix: str, res_dir: Path):
    triple_list = list(zip(df['subject'], df['predicate'], df['object']))

    batches = make_batches(triple_list, size=BATCH_SIZE)
    results, start_time = classify_batches(batches, predicate=pred)
    total_time = time.time() - start_time
    avg_time = total_time / len(df)

    rows = []
    errors = 0
    row_idx = 0
    for batch, result in zip(batches, results):
        if result is None:
            errors += len(batch)
            row_idx += len(batch)
            continue

        sorted_evals = sorted(result.modifications.items(), key=lambda x: int(x[0]))

        if len(sorted_evals) == len(batch):
            for i, (_, edit) in enumerate(sorted_evals):
                orig = df.iloc[row_idx + i]
                rows.append(mod_row(orig, pred, edit))   # _mod_row gère verdict/new/provenance
        else:
            for idx_str, edit in result.modifications.items():
                idx = int(idx_str)
                if idx >= len(batch):
                    print(f"  Error: index {idx} for batch size {len(batch)} (pred={pred})")
                    errors += 1
                    continue
                orig = df.iloc[row_idx + idx]
                rows.append(mod_row(orig, pred, edit))
        row_idx += len(batch)
        
    out_df = pd.DataFrame(rows)

    out_dir = res_dir / pred_parsed
    safe_mkdir(out_dir)

    out_df.to_csv(out_dir / f"{dataset_prefix}mod_{pred_parsed}_{chunk_num}.csv", index=False)

    metrics_summary = out_df["destination"].value_counts().to_dict()

    formatted_prompt = build_prompt(pred)

    with open(out_dir / f"{dataset_prefix}cmod_{pred_parsed}_{chunk_num}.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME_LIGHT}\n")
        f.write(f"Predicate: {pred}\n") 
        f.write(f"Chunk: {chunk_num}\n")
        f.write(f"Triples: {len(df)}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f}min)\n")
        f.write(f"Avg time per triplet: {avg_time:.4f}s | {avg_time*1000:.2f}ms\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"\nDestination distribution:\n")
        for dest, count in metrics_summary.items():
            f.write(f"  {dest}: {count}\n")
        f.write(f"\n{'='*50}\n")
        f.write(f"PROMPT:\n\n{formatted_prompt}\n")

    print(f"  -> {dataset_prefix}mod_{pred_parsed}_{chunk_num}.csv | {len(df)} triples | {total_time:.1f}s | {avg_time*1000:.2f}ms/triple | {errors} errors")

    return out_df


if __name__ == "__main__":
    args = parse_args()
    dp = args.dataset_prefix
    data_dir = INPUT_DIR / f"{dp}_final_process" / "04_Filtering" / "MODIFY"
    output_dir = INPUT_DIR / f"{dp}_final_process" / "05_MODIFIED"
    output_dir.mkdir(parents=True, exist_ok=True)

    for pred in args.predicates:
        pred_parsed = pred.strip().replace(" ", "").lower()
        pred_dir = data_dir / pred_parsed

        if not pred_dir.exists():
            print(f"No directory for '{pred}' at {pred_dir}, skipping")
            continue

        chunk_files = sorted(pred_dir.glob(f"{dp}fm_{pred_parsed}_*.csv"))
        if not chunk_files:
            print(f"No chunk files for '{pred}' in {pred_dir}, skipping")
            continue

        print(f"Predicate: {pred} ({len(chunk_files)} chunks)")

        for chunk_file in chunk_files:
            chunk_num = int(re.search(r"_(\d+)\.csv$", chunk_file.name).group(1))

            result_file = output_dir / pred_parsed / f"{dp}mod_{pred_parsed}_{chunk_num}.csv"
            if result_file.exists():
                print(f"  Skipping chunk {chunk_num}, already done ({result_file.name})")
                continue

            print(f"  Processing chunk {chunk_num}: {chunk_file.name}")
            df = pd.read_csv(chunk_file)
            df.columns = df.columns.str.strip()

            try:
                run_chunk(df, pred=pred, pred_parsed=pred_parsed, chunk_num=chunk_num, dataset_prefix=dp, res_dir=output_dir)
            except Exception as e:
                print(f"  Error on chunk {chunk_num} of '{pred}': {e}")
                traceback.print_exc()
                continue
            del df

    print("\nDone")