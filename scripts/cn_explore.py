import pandas as pd
from pathlib import Path
from loading_dataset import load_conceptnet

PREDICATES = [
    "AtLocation", "CapableOf", "Causes", "CauseDesire", "CreatedBy",
    "DefinedAs", "Desires", "DistinctFrom", "HasA", "HasSubevent",
    "HasFirstSubevent", "HasLastSubevent", "HasPrerequisite",
    "HasProperty", "MadeOf", "MannerOf", "MotivatedByGoal",
    "PartOf", "ReceivesAction", "UsedFor",
]

OUTPUT_PATH = Path("samples.csv")
SAMPLE_SIZE = 15


def sample_predicate(df: pd.DataFrame, predicate: str, size: int):
    df_pred = df.loc[df["predicate"] == predicate]
    return df_pred.sample(min(size, len(df_pred)))


if __name__ == "__main__":
    df_cn = load_conceptnet()
    samples = []

    for pred in PREDICATES:
        df_sample = sample_predicate(df_cn, pred, SAMPLE_SIZE)
        samples.append(df_sample)
        print(f"{pred}: {len(df_sample)} samples")

    pd.concat(samples).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {sum(len(s) for s in samples)} samples to {OUTPUT_PATH}")