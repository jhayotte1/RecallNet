import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

if __name__=="__main__":
    in_dir = Path(DATA_DIR, "top_5M_by_predicate")
    csvs = in_dir.glob("*.csv")
    print("========")
    print("Nombre de triplet par prédicat dans le top 5M")
    for csv in csvs:
        pred = csv.stem.replace(f"quasi_top5000000_", "")
        pred_df = pd.read_csv(csv)
        print(f"Predicat : {pred} ; Nombre de ligne : {len(pred_df)}")
        del pred_df

