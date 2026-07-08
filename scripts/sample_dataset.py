import pandas as pd
from pathlib import Path
from loading_dataset import load_ascent, load_quasimodo, load_conceptnet

SAMPLE_SIZE = 100
DATA_DIR = Path(__file__).parent.parent / "data"


def sample_dataset(df: pd.DataFrame, size=SAMPLE_SIZE):
    return(df.sample(size))

def sample_predicate(df: pd.DataFrame, size=SAMPLE_SIZE, predicate: str=""):
    df_sampled = df[df["predicate"] == predicate].copy()
    if len(df_sampled) == 0:
        return None
    df_sampled = df_sampled.sample(size)
    return df_sampled


if __name__ == "__main__":
    df_quasi = load_quasimodo()
    #df_sampled = sample_dataset(df_quasi, size=SAMPLE_SIZE)
    predicate_list1 = ["causes", "causes desire", "created by", "defined as"]
    predicate_list2 = ["desires", "distinct from", "has a", "has subevent"]
    predicate_list3 = ["has first subevent", "has last subevent", "has prerequisite", "motivated by goal"]
    predicate_list4 = ["made of", "manner of", "part of", "receives action", "used for"]
    predicate_list5 = ["instance of", "entails"]
    for predicate in predicate_list5:
        df_sampled = sample_predicate(df_quasi, size=SAMPLE_SIZE, predicate=predicate)
        if df_sampled is not None:
            predicate = predicate.strip().replace(" ", "")
            df_sampled.to_csv(f"{DATA_DIR}/quasi_sample_{predicate}.csv", index=False)
            print(f"Quasimodo dataset sampled at sample_size = {SAMPLE_SIZE} and predicate = '{predicate}'")
            print(f"Saved at: {DATA_DIR}/quasi_sample_{predicate}.csv")
        else:
            print(f"No triple with predicate = '{predicate}'")

