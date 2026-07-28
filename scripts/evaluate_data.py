import pandas as pd
from pathlib import Path
import re
from loading_dataset import load_conceptnet, load_quasimodo

RES_DIR = Path(__file__).parent.parent / "results" / "llama3.1:8b" / "final_process"
DATA_DIR = Path(__file__).parent.parent / "data"

def to_quasimodo_format(df):
    for c in ['subject', 'object']:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df['predicate'] = (
        df['predicate'].astype(str)
        .str.replace(r'(?<!^)(?=[A-Z])', ' ', regex=True) 
        .str.strip()
        .str.lower()
    )
    return df

def compare_top5M_cn():
    df_cn = load_conceptnet().drop_duplicates()
    csvs = RES_DIR.rglob("*.csv")
    overlaps = []
    cols = ['subject', 'predicate', 'object']
    total = 0

    for csv in csvs:
        pred_n = csv.stem.strip().replace("pred_llama3.1:8b_", "")
        df_pred = pd.read_csv(csv)

        common = df_pred.merge(df_cn, on=cols, how="inner")
        overlaps.append(common)
        total += len(common)

        print(f"Total overlap with ConceptNet in '{pred_n}' : {len(common)}")

    df_overlap = pd.concat(overlaps, ignore_index=True) if overlaps else pd.DataFrame()
    print(f"Total number of triples overlapping with ConceptNet: {total}")

    out_dir = RES_DIR / "overlap_cn"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_overlap.to_csv(out_dir / "overlap_cn.csv", index=False)

def compare_tot_cn():
    df_quasi = load_quasimodo()
    df_cn = load_conceptnet()

    cols = ['subject', 'predicate', 'object']

    print("Normalize ConceptNet to Quasimodo format")
    df_cn = to_quasimodo_format(df_cn).drop_duplicates(subset=cols)

    df_quasi = df_quasi.reset_index(drop=True)
    df_quasi['rank'] = df_quasi.index + 1     # rang 1-based
    common = df_quasi.merge(df_cn, on=cols, how="inner")
    total = len(common)
    
    print("Doublons dans CN normalisé :", df_cn.duplicated(subset=cols).sum())
    print("Doublons dans overlap :", common.duplicated(subset=cols).sum())
    print("Triplets distincts en overlap :", common.drop_duplicates(subset=cols).shape[0])
    common = common.drop_duplicates(subset=cols)
    print(f"Total triple in Quasimodo overlap with ConceptNet : {total}")
    print(f"Best Quasimodo rank in overlap : rank={common['rank'].min()}")


    print("Saving overlap dataframe")
    common.to_csv(Path(DATA_DIR, "quasi_overlap_cn.csv"))


if __name__=="__main__":
    compare_tot_cn()