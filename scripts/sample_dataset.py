import pandas as pd
from pathlib import Path
from loading_dataset import load_ascent, load_quasimodo, load_conceptnet

SAMPLE_SIZE = 100
DATA_DIR = Path(__file__).parent.parent / "data"


def sample_dataset(df: pd.DataFrame, size=SAMPLE_SIZE):
    return(df.sample(size))

if __name__ == "__main__":
    df_quasi = load_quasimodo()
    df_sampled = sample_dataset(df_quasi, size=SAMPLE_SIZE)
    df_sampled.to_csv(f"{DATA_DIR}/quasi_test_100_sample.csv")

