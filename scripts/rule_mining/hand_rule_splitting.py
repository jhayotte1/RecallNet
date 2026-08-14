import pandas as pd
from fnmatch import fnmatch
from pathlib import Path
from argparse import ArgumentParser

RES_DIR = Path("~/RecallNet/src/results/llama3.1:8b-fp8").expanduser()

KEEP = {
    "m_min": 4, "m_max": 5,
    "t_min": 4, "t_max": 5,
    "s_min": 4, "s_max": 5,
}
IN_BETWEEN = {
    "m_min": 4, "m_max": 5,
    "t_min": 3, "t_max": 5,
    "s_min": 1, "s_max": 3,
}

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset-prefix", type=str, required=True, help="q: Quasimodo ; a: Ascent")
    parser.add_argument("--mod", action="store_true", help="Split rescored (modified) triples from 06_RESCORING instead of the initial scoring")
    return parser.parse_args()

def matches_rule(df: pd.DataFrame, rule: dict):
    return(
        df["meaningfulness"].between(rule["m_min"], rule["m_max"]) &
        df["typicality"].between(rule["t_min"], rule["t_max"]) &
        df["saliency"].between(rule["s_min"], rule["s_max"])
    )

def split_triples(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    is_keep = matches_rule(df, KEEP)
    is_between = matches_rule(df, IN_BETWEEN) & ~is_keep

    return {
        "keep": df[is_keep],
        "in_between": df[is_between],
    }

def splitting_by_rule():
    args = arg_parser()
    dp = args.dataset_prefix
    base_dir = RES_DIR / f"{dp}_final_process"

    if args.mod:
        input_dir = base_dir / "06_RESCORING"
        output_dir = base_dir / "06_RESCORING" / "03_Splitting"
        csv_glob = f"{dp}rp_*.csv"
    else:
        input_dir = base_dir
        output_dir = base_dir / "03_Splitting"
        csv_glob = f"{dp}p_*.csv"

    pred_dirs = [
        p for p in sorted(input_dir.glob("*"))
        if p.is_dir() and not fnmatch(p.name, "0*")
    ]
    keep_dir = output_dir / "KEEP"
    inbet_dir = output_dir / "INBETWEEN"
    stats_dir = output_dir / "stats"
    stats_dir_predicate = stats_dir / "per_predicate"
    keep_dir.mkdir(parents=True, exist_ok=True)
    inbet_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_dir_predicate.mkdir(parents=True, exist_ok=True)
    stats_tot = []
    for pred_dir in pred_dirs:
        pred = pred_dir.stem
        print(f"Splitting predicate : {pred}")
        keep_pred_dir = keep_dir / pred
        inbet_pred_dir = inbet_dir / pred
        keep_pred_dir.mkdir(parents=True, exist_ok=True)
        inbet_pred_dir.mkdir(parents=True, exist_ok=True)

        stats_pred = []
        chunks = pred_dir.glob(csv_glob)
        for chunk in chunks:
            chunk_name = chunk.stem.split("_", 1)[1]
            df_chunk = pd.read_csv(chunk)
            splits = split_triples(df_chunk)
            splits["keep"].to_csv(keep_pred_dir / f"k_{chunk_name}.csv", index=False)
            splits["in_between"].to_csv(inbet_pred_dir / f"ib_{chunk_name}.csv", index=False)
            tot, tot_keep, tot_inbet = len(df_chunk), len(splits["keep"]), len(splits["in_between"])
            tot_rej = tot - tot_keep - tot_inbet
            stats_pred.append({
                "file": chunk.stem,
                "total": tot,
                "keep": tot_keep,
                "in_between": tot_inbet,
                "reject": tot_rej,
                "keep%": f"{tot_keep/tot*100:.1f}" if tot else "0.0",
                "in_between%": f"{tot_inbet/tot*100:.1f}" if tot else "0.0",
                "reject%": f"{tot_rej/tot*100:.1f}" if tot else "0.0",
            })
        if not stats_pred:
            continue
        df_stats_pred = pd.DataFrame(stats_pred)
        df_stats_pred.to_csv(stats_dir_predicate / f"s_{pred}.csv", index=False)
        pred_total = df_stats_pred["total"].sum()
        pred_keep = df_stats_pred["keep"].sum()
        pred_inbet = df_stats_pred["in_between"].sum()
        pred_rej = pred_total - pred_keep - pred_inbet
        stats_tot.append({
            "predicate": pred,
            "total": pred_total,
            "keep": pred_keep,
            "in_between": pred_inbet,
            "reject": pred_rej,
            "keep%": f"{pred_keep/pred_total*100:.1f}" if pred_total else "0.0",
            "in_between%": f"{pred_inbet/pred_total*100:.1f}" if pred_total else "0.0",
            "reject%": f"{pred_rej/pred_total*100:.1f}" if pred_total else "0.0",
        })

    df_stats_tot = pd.DataFrame(stats_tot)
    df_stats_tot.to_csv(stats_dir / "splitting_summary.csv", index=False)
    total_all = df_stats_tot["total"].sum()
    total_keep = df_stats_tot["keep"].sum()
    total_between = df_stats_tot["in_between"].sum()
    total_reject = total_all - total_keep - total_between

    with open(stats_dir / "decision_rules.txt", "w") as f:
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
        f.write(df_stats_tot.to_string(index=False))

if __name__=="__main__":
    splitting_by_rule()