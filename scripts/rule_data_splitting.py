import pandas as pd
from fnmatch import fnmatch
from pathlib import Path
from langchain_pipeline.config import MODEL_NAME_LIGHT

EXP_NAME = "exp05_LG_1k_top_100k"   
INPUT_DIR = Path(__file__).parent.parent / "results" / MODEL_NAME_LIGHT / "scoring_exp" / EXP_NAME
OUTPUT_DIR = Path(__file__).parent.parent / "results" / MODEL_NAME_LIGHT / "scoring_exp" / EXP_NAME / "1_SPLITED"

KEEP = {
    "m_min": 4, "m_max": 5,
    "t_min": 4, "t_max": 5,
    "s_min": 4, "s_max": 5,
}
IN_BETWEEN = {
    "m_min": 4, "m_max": 5,
    "t_min": 4, "t_max": 5,
    "s_min": 2, "s_max": 3,
}

def matches_rule(df: pd.DataFrame, rule: dict):
    return(
        df["meaningfulness"].between(rule["m_min"], rule["m_max"]) &
        df["typicality"].between(rule["t_min"], rule["t_max"]) &
        df["saliency"].between(rule["s_min"], rule["s_max"])
    )

def split_triples(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    is_keep = matches_rule(df, KEEP)
    is_between = matches_rule(df, IN_BETWEEN) & ~is_keep
    is_reject = ~is_keep & ~is_between

    return {
        "keep": df[is_keep],
        "in_between": df[is_between],
        "reject": df[is_reject],
    }

if __name__ == "__main__":
    EXCLUDE_PATTERN = "*_SPLITED"

    csvs = [
        p for p in sorted(INPUT_DIR.rglob("pred_*.csv"))
        if not any(fnmatch(part, EXCLUDE_PATTERN) for part in p.parts)
    ]

    if not csvs:
        print(f"No scored CSVs found in {INPUT_DIR}")
        exit(1)

    stats = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        splits = split_triples(df)

        for label, split_df in splits.items():
            out_path = OUTPUT_DIR / label
            out_path.mkdir(parents=True, exist_ok=True)
            split_df.to_csv(out_path / csv_path.name, index=False)

        total = len(df)
        stats.append({
            "file": csv_path.name,
            "total": total,
            "keep": len(splits["keep"]),
            "in_between": len(splits["in_between"]),
            "reject": len(splits["reject"]),
            "keep%": f"{len(splits['keep'])/total*100:.1f}",
            "in_between%": f"{len(splits['in_between'])/total*100:.1f}",
            "reject%": f"{len(splits['reject'])/total*100:.1f}",
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    total_all = stats_df["total"].sum()
    total_keep = stats_df["keep"].sum()
    total_between = stats_df["in_between"].sum()
    total_reject = stats_df["reject"].sum()

    with open(OUTPUT_DIR / "decision_rules.txt", "w") as f:
        f.write("Decision rules\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"KEEP:       M∈[{KEEP['m_min']},{KEEP['m_max']}] AND T∈[{KEEP['t_min']},{KEEP['t_max']}] AND S∈[{KEEP['s_min']},{KEEP['s_max']}]\n")
        f.write(f"IN_BETWEEN: M∈[{IN_BETWEEN['m_min']},{IN_BETWEEN['m_max']}] AND T∈[{IN_BETWEEN['t_min']},{IN_BETWEEN['t_max']}] AND S∈[{IN_BETWEEN['s_min']},{IN_BETWEEN['s_max']}]\n")
        f.write(f"REJECT:     everything else\n\n")
        f.write(f"Totals:\n")
        f.write(f"  KEEP:       {total_keep:>6} ({total_keep/total_all*100:.1f}%)\n")
        f.write(f"  IN_BETWEEN: {total_between:>6} ({total_between/total_all*100:.1f}%)\n")
        f.write(f"  REJECT:     {total_reject:>6} ({total_reject/total_all*100:.1f}%)\n")
        f.write(f"  TOTAL:      {total_all:>6}\n\n")
        f.write(stats_df.to_string(index=False))

    print(f"\nDecision rules:")
    print(f"  KEEP:       M∈[{KEEP['m_min']},{KEEP['m_max']}] AND T∈[{KEEP['t_min']},{KEEP['t_max']}] AND S∈[{KEEP['s_min']},{KEEP['s_max']}]")
    print(f"  IN_BETWEEN: M∈[{IN_BETWEEN['m_min']},{IN_BETWEEN['m_max']}] AND T∈[{IN_BETWEEN['t_min']},{IN_BETWEEN['t_max']}] AND S∈[{IN_BETWEEN['s_min']},{IN_BETWEEN['s_max']}]")
    print(f"  REJECT:     everything else\n")
    print(stats_df.to_string(index=False))
    print(f"\nTotals:")
    print(f"  KEEP:       {total_keep:>6} ({total_keep/total_all*100:.1f}%)")
    print(f"  IN_BETWEEN: {total_between:>6} ({total_between/total_all*100:.1f}%)")
    print(f"  REJECT:     {total_reject:>6} ({total_reject/total_all*100:.1f}%)")
    print(f"  TOTAL:      {total_all:>6}")