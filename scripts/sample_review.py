import pandas as pd
from pathlib import Path
import json


# Predicates to skip (not actual predicate folders)
SKIP_DIRS = {"00_LOGS"}
DATA_DIR = Path("~/RecallNet/src/results/llama3.1:8b-fp8/q_final_process").expanduser()
CN_DATA = Path("~/RecallNet/src/results/llama3.1:8b-fp8/process_overlap_quasi_cn").expanduser()
OUTDIR = "00_sampling"
SAMPLE_SIZE = 100
RANDOM_STATE = 42


def load_predicate_triples(pred_dir: Path) -> pd.DataFrame:
    csv_files = sorted(pred_dir.glob("qp_*.csv"))
    if not csv_files:
        print(f"No qp_*.csv found in {pred_dir.name}, skipping")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  {pred_dir.name}: {len(combined)} triples from {len(csv_files)} file(s)")
    return combined


def sample_triples(df: pd.DataFrame, n=SAMPLE_SIZE, seed=RANDOM_STATE) -> pd.DataFrame:
    if len(df) <= n:
        print(f"    → only {len(df)} available, taking all")
        return df
    sampled = df.sample(n=n, random_state=seed)
    del df
    return sampled


def sampling_scored_data():
    pred_dirs = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    ])

    print(f"Found {len(pred_dirs)} predicate directories\n")

    for pred_dir in pred_dirs:
        triples = load_predicate_triples(pred_dir)
        if triples.empty:
            continue

        sampled = sample_triples(triples)
        pred = pred_dir.stem
        output = DATA_DIR / OUTDIR
        output.mkdir(parents=True, exist_ok=True)
        output_to = output / "to"
        output_to.mkdir(parents=True, exist_ok=True)
        sampled.to_csv(output / f"sampled_{pred}.csv", index=False)
        sampled[["subject", "predicate", "object"]].to_csv(output_to / f"sampled_{pred}_to.csv", index=False)
        print(f"Saved at {output}")

def convert_json_review_to_csv():
    samp_dir = DATA_DIR / "01_Sampling" / "ws"
    in_dir = DATA_DIR / "02_Review" / "json"
    out_dir = DATA_DIR / "02_Review" / "csvs"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsons = in_dir.glob("*.json")
    for jsn in jsons:
        pred = jsn.stem
        with open(jsn) as f:
            rev_df = pd.DataFrame(json.load(f))
        rev_df.rename(columns={"reasoning": "r_reason"}, inplace=True)

        ws_df = pd.read_csv(samp_dir / f"sampled_{pred}.csv")
        ws_df.rename(columns={"reason": "s_reason"}, inplace=True)

        merged = ws_df.merge(
            rev_df[["subject", "object", "verdict", "r_reason"]],
            on=["subject", "object"],
            how="left"
        )

        col_order = [
            "subject", "predicate", "object", "meaningfulness", "typicality", "saliency", "verdict", "source_file" 
        ]
        merged = merged[col_order]
        merged.to_csv(out_dir / f"rev_{pred}.csv", index=False)

def concat_rev():
    in_dir = DATA_DIR / "02_Review" / "csvs"
    out_dir = DATA_DIR / "02_Review"
    dfs = [pd.read_csv(f) for f in in_dir.glob("rev_*.csv")]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_dir / "rev_data.csv", index=False)

def count_rev():
    in_dir = DATA_DIR / "02_Review"
    df = pd.read_csv(in_dir / "rev_data_wcn.csv")
    print(df["verdict"].value_counts())

def concat_cn_scored():
    pred_dirs = sorted([
        d for d in CN_DATA.iterdir()
    ])
    dfs = []
    for pred_dir in pred_dirs:
        pred = pred_dir.stem
        df = pd.read_csv(pred_dir / f"qp_{pred}_1.csv")
        df["source_file"] = f"ov_cn_{pred}.csv"
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["verdict"] = "KEEP"
    col_list = [
        "subject", "predicate", "object",
        "meaningfulness", "typicality", "saliency",
        "verdict", "source_file",
    ]
    combined = combined[col_list]
    combined.to_csv(CN_DATA / "rev_cn_ov.csv", index=False)

def add_cn_triple_to_rev_data():
    df_cn = pd.read_csv(CN_DATA / "rev_cn_ov.csv")
    df_rev = pd.read_csv(DATA_DIR / "02_Review" / "rev_data.csv")
    df_cn_sample = df_cn.sample(n=800, random_state=RANDOM_STATE)
    combined = pd.concat([df_rev, df_cn_sample], ignore_index=True)
    combined.to_csv(DATA_DIR / "02_Review" / "rev_data_wcn.csv")

if __name__ == "__main__":
    count_rev()