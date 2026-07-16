import pandas as pd
from pathlib import Path
from loading_dataset import load_ascent, load_quasimodo, load_conceptnet

SAMPLE_SIZE = 1000
DATA_DIR = Path(__file__).parent.parent / "data"


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
    for predicate in predicate_list:
        df_sampled = sample_predicate_top(df_quasi, size=SAMPLE_SIZE, predicate=predicate)
        if df_sampled is not None:
            predicate = predicate.strip().replace(" ", "")
            df_sampled.to_csv(f"{DATA_DIR}/sample_data_1k_top_100k/quasi_sample_{predicate}.csv", index=False)
            print(f"Quasimodo dataset sampled at sample_size = {SAMPLE_SIZE} and predicate = '{predicate}'")
            print(f"Saved at: {DATA_DIR}/sample_data_1k_top_100k/quasi_sample_{predicate}.csv")
            del df_sampled
        else:
            print(f"No triple with predicate = '{predicate}'")