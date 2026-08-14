import pandas as pd
import shutil
from fnmatch import fnmatch
from pathlib import Path
from argparse import ArgumentParser

RES_DIR = Path("~/RecallNet/src/results/llama3.1:8b-fp8").expanduser()

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset-prefix", type=str, required=True, help="Prefix dataset to be processed. q: Quasimodo ; a: Ascent")
    return parser.parse_args()

def split_dest_df(df: pd.DataFrame):
    keep = df[df["destination"]=="keep"]
    mod = df[df["destination"]=="modified"]
    rej = df[df["destination"]=="reject"]
    fail = df[df["destination"]=="failed"]
    return keep, mod, rej, fail

def split_dest_global():
    args = arg_parser()
    dp = args.dataset_prefix
    input_dir = RES_DIR / f"{dp}_final_process" / "05_MODIFIED"
    config_dir = input_dir / "0_config"
    keep_dir = input_dir / "KEEP"
    mod_dir = input_dir / "MODIFIED"
    rej_dir = input_dir / "REJECT"
    fail_dir = input_dir / "FAILED"
    stats_dir = input_dir / "0_stats"
    stats_dir_pred = stats_dir / "predicates"
    config_dir.mkdir(parents=True, exist_ok=True)
    keep_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)
    rej_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)
    stats_dir_pred.mkdir(parents=True, exist_ok=True)
    EXCLUDE = {"KEEP", "MODIFIED", "REJECT", "FAILED", "0_stats", "0_config", "0_LOGS"}
    pred_dirs = [
        p for p in sorted(input_dir.glob("*"))
        if p.is_dir() and not fnmatch(p.name, "0*") and p.name not in EXCLUDE
    ]
    stats = []
    for pred_dir in pred_dirs:
        pred = pred_dir.stem
        config_pred_dir = config_dir / f"{dp}cm_{pred}"
        config_pred_dir.mkdir(parents=True, exist_ok=True)
        for f in pred_dir.glob(f"{dp}cmod_{pred}_*.txt"):
            f.rename(config_pred_dir / f.name)
        keep_pred_dir = keep_dir / pred
        mod_pred_dir = mod_dir / pred
        rej_pred_dir = rej_dir / pred
        fail_pred_dir = fail_dir / pred
        keep_pred_dir.mkdir(parents=True, exist_ok=True)
        mod_pred_dir.mkdir(parents=True, exist_ok=True)
        rej_pred_dir.mkdir(parents=True, exist_ok=True)
        fail_pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"Splitting pred : {pred}")
        stats_pred = []
        chunks = pred_dir.glob(f"{dp}mod_*.csv")
        for chunk in chunks:
            chunk_name = chunk.stem.split("_", 1)[1]
            df = pd.read_csv(chunk)
            keep, mod, rej, fail = split_dest_df(df)
            keep.to_csv(keep_pred_dir / f"{dp}mk_{chunk_name}.csv", index=False)
            mod.to_csv(mod_pred_dir / f"{dp}mm_{chunk_name}.csv", index=False)
            rej.to_csv(rej_pred_dir / f"{dp}mr_{chunk_name}.csv", index=False)
            fail.to_csv(fail_pred_dir / f"{dp}mf_{chunk_name}.csv", index=False)
            tot, tot_keep, tot_mod, tot_rej, tot_fail = len(df), len(keep), len(mod), len(rej), len(fail)
            stats_pred.append({
                "file": chunk_name,
                "total": tot,
                "keep": tot_keep,
                "modified": tot_mod,
                "reject": tot_rej,
                "failed": tot_fail,
                "keep%": f"{tot_keep/tot*100:.1f}" if tot else "0.0",
                "modified%": f"{tot_mod/tot*100:.1f}" if tot else "0.0",
                "reject%": f"{tot_rej/tot*100:.1f}" if tot else "0.0",
                "failed%": f"{tot_fail/tot*100:.1f}" if tot else "0.0",
            })
        if not stats_pred:
            continue
        df_stats_pred = pd.DataFrame(stats_pred)
        df_stats_pred.to_csv(stats_dir_pred / f"s_{pred}.csv", index=False)
        pred_total = df_stats_pred["total"].sum()
        pred_keep = df_stats_pred["keep"].sum()
        pred_mod = df_stats_pred["modified"].sum()
        pred_rej = df_stats_pred["reject"].sum()
        pred_fail = df_stats_pred["failed"].sum()
        stats.append({
            "predicate": pred,
            "total": pred_total,
            "keep": pred_keep,
            "modified": pred_mod,
            "reject": pred_rej,
            "failed": pred_fail,
            "keep%": f"{pred_keep/pred_total*100:.1f}" if pred_total else "0.0",
            "modified%": f"{pred_mod/pred_total*100:.1f}" if pred_total else "0.0",
            "reject%": f"{pred_rej/pred_total*100:.1f}" if pred_total else "0.0",
            "failed%": f"{pred_fail/pred_total*100:.1f}" if pred_total else "0.0",
        })
    df_stats_tot = pd.DataFrame(stats)
    df_stats_tot.to_csv(stats_dir / "modifying_summary.csv", index=False)
    total_all = df_stats_tot["total"].sum()
    total_keep = df_stats_tot["keep"].sum()
    total_mod = df_stats_tot["modified"].sum()
    total_reject = df_stats_tot["reject"].sum()
    total_fail = df_stats_tot["failed"].sum()

    with open(stats_dir / "modifying.txt", "w") as f:
        f.write("Modifying Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Description : Modification of the triples labeled 'MODIFY' by the filtering step\n")
        f.write(f"Model used for modifying : llama3.1:8b-fp8\n\n")
        f.write(f"Totals:\n")
        f.write(f"  KEEP:     {total_keep:>6} ({total_keep/total_all*100:.1f}%)\n")
        f.write(f"  MODIFIED: {total_mod:>6} ({total_mod/total_all*100:.1f}%)\n")
        f.write(f"  REJECT:   {total_reject:>6} ({total_reject/total_all*100:.1f}%)\n")
        f.write(f"  FAILED:   {total_fail:>6} ({total_fail/total_all*100:.1f}%)\n")
        f.write(f"  TOTAL:    {total_all:>6}\n\n")
        f.write(df_stats_tot.to_string(index=False))
    return None

def remove_old():
    args = arg_parser()
    dp = args.dataset_prefix
    input_dir = RES_DIR / f"{dp}_final_process" / "05_MODIFIED"
    EXCLUDE = {"KEEP", "MODIFIED", "REJECT", "FAILED", "0_stats", "0_config", "0_LOGS"}
    pred_dirs = [
        p for p in sorted(input_dir.glob("*"))
        if p.is_dir() and not fnmatch(p.name, "0*") and p.name not in EXCLUDE
    ]
    for pred_dir in pred_dirs:
        print(f"Removing {pred_dir.name}")
        shutil.rmtree(pred_dir)

if __name__=="__main__":
    remove_old()