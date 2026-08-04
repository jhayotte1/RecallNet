import argparse
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

REV_DATA = Path("~/RecallNet/src/results/llama3.1:8b-fp8/q_final_process").expanduser()

def exhaustive_search():
    in_dir = REV_DATA / "02_Review"
    out_dir = REV_DATA / "02_Review" / "grid_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_dir / "rev_data_wcn.csv")
    y_true = (df["verdict"] == "KEEP").astype(int).values
    results = []
    for m_min in range(0,6):
        for t_min in range(0,6):
            for s_min in range(0,6):
                y_keep = (df["meaningfulness"] >= m_min) & (df["typicality"] >= t_min) & (df["saliency"] >= s_min)

                precision = precision_score(y_true, y_keep)
                recall = recall_score(y_true, y_keep)
                f1 = f1_score(y_true, y_keep)
                results.append({
                    "M_min": m_min,
                    "T_min": t_min,
                    "S_min": s_min,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                })
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("f1_score", ascending=False)
    res_df.to_csv(out_dir / "gs_rules_max_f1.csv")

if __name__ == "__main__":
    exhaustive_search()