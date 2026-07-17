import pandas as pd
from pathlib import Path
from fnmatch import fnmatch

from loading_dataset import load_ascent, load_quasimodo, load_conceptnet

SAMPLE_SIZE = 1000
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def sample_dataset(df: pd.DataFrame, size=SAMPLE_SIZE):
    return(df.sample(size))

def sample_predicate_top(df: pd.DataFrame, size=SAMPLE_SIZE, predicate: str="", top_n=100000):
    df_pred = df.loc[df["predicate"] == predicate].head(top_n)
    if len(df_pred) == 0:
        return None
    if len(df_pred) < size:
        print(f"Warning: only {len(df_pred)} triples available for '{predicate}' in top {top_n}")
        return df_pred
    return df_pred.sample(size)

def sample_predicate_size(df: pd.DataFrame, size=SAMPLE_SIZE, predicate: str=""):
    df_pred = df.loc[df["predicate"] == predicate]
    if len(df_pred) == 0:
        return None
    if len(df_pred) < size:
        print(f"Warning: only {len(df_pred)} triples available for '{predicate}'")
        return df_pred
    return df_pred.sample(size)

def split_top_by_predicate(df: pd.DataFrame, top_n: int, output_dir: str, predicate_list: list[str]):
    df_top = df.head(top_n)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for predicate in predicate_list:
        df_pred = df_top.loc[df_top["predicate"] == predicate]
        if len(df_pred) == 0:
            print(f"No triples for '{predicate}' in top {top_n}")
            continue
        pred_clean = predicate.strip().replace(" ", "")
        df_pred.to_csv(output_path / f"quasi_top{top_n}_{pred_clean}.csv", index=False)
        print(f"{predicate}: {len(df_pred)} triples saved")
        del df_pred

def sampling_results(size: int, model_name: str, exp_name: str):
    res_path = Path(RESULTS_DIR / model_name / "scoring_exp" / exp_name)
    out_path = res_path / "0_SAMPLING"
    out_path.mkdir(parents=True, exist_ok=True)

    EXCLUDE_PATTERNS = ["*SPLITED*", "*SAMPLING*"]
    csvs = [
        p for p in sorted(res_path.rglob("pred_*.csv"))
        if not any(fnmatch(part, pat) for part in p.parts for pat in EXCLUDE_PATTERNS)
    ]
    if not csvs:
        print(f"No scored CSVs found in {res_path}")
        exit(1)

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        df_sample = df.sample(min(size, len(df)))
        df_sample.to_csv(out_path / f"sample_{csv_path.name}", index=False)
        print(f"{csv_path.name} sampled")
        del df
        del df_sample
        

if __name__ == "__main__":
    df_quasi = load_quasimodo()
    #df_sampled = sample_dataset(df_quasi, size=SAMPLE_SIZE)
    predicate_list = [
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
    # for predicate in predicate_list:
    #     df_sampled = sample_predicate_top(df_quasi, size=SAMPLE_SIZE, predicate=predicate)
    #     if df_sampled is not None:
    #         predicate = predicate.strip().replace(" ", "")
    #         df_sampled.to_csv(f"{DATA_DIR}/sample_data_1k_top_100k/quasi_sample_{predicate}.csv", index=False)
    #         print(f"Quasimodo dataset sampled at sample_size = {SAMPLE_SIZE} and predicate = '{predicate}'")
    #         print(f"Saved at: {DATA_DIR}/sample_data_1k_top_100k/quasi_sample_{predicate}.csv")
    #         del df_sampled
    #     else:
    #         print(f"No triple with predicate = '{predicate}'")

    
    # split_top_by_predicate(
    #     df=df_quasi,
    #     top_n=5000000,
    #     output_dir=f"{DATA_DIR}/top_5M_by_predicate",
    #     predicate_list=predicate_list,
    # )

    sampling_results(
        size=100,
        model_name="llama3.1:8b",
        exp_name="exp05_LG_1k_top_100k"
    )

