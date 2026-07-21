import json
import argparse
import pandas as pd
from fnmatch import fnmatch
from pathlib import Path

RESULTS_DIR=Path(__file__).parent.parent.parent / "results"

PRED_MAP = {
        "atlocation": "at location",
        "capableof": "capable of",
        "causes": "causes",
        "causedesire": "cause desire",
        "createdby": "created by",
        "definedas": "defined as",
        "desires": "desires",
        "distinctfrom": "distinct from",
        "hasa": "has a",
        "hassubevent": "has subevent",
        "hasfirstsubevent": "has first subevent",
        "haslastsubevent": "has last subevent",
        "hasprerequisite": "has prerequisite",
        "hasproperty": "has property",
        "madeof": "made of",
        "mannerof": "manner of",
        "motivatedbygoal": "motivated by goal",
        "partof": "part of",
        "receivesaction": "receives action",
        "usedfor": "used for"
    }

def arg_parse():
    parser = argparse.ArgumentParser(description="Script for splitting the data following a mined rule")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of experiment directory")
    parser.add_argument("--model-reviewed", type=str, default="llama3.1:8b", help="Name of the model used for scoring")
    parser.add_argument("--rule-dir", type=str, default="0_RULE", help="Directory for json rules")
    parser.add_argument("--outdir", type=str, default="0_SPLITED", help="Output Directory for your splited data")
    args = parser.parse_args()
    return args

def parse_conditions(cond: str):
    parts = cond.split()
    return parts[0], parts[1], float(parts[2])

def matches_rule(row, rule):
    for cond in rule["conditions"]:
        feat, op, thresh = parse_conditions(cond)
        if op=='<=' and not (row[feat] <= thresh):
            return False
        if op=='>' and not (row[feat] > thresh):
            return False
    return True

def classify_row(row, keep_rules, reject_rules):
    for rule in keep_rules:
        if matches_rule(row, rule):
            return "KEEP"
    for rule in reject_rules:
        if matches_rule(row, rule):
            return "REJECT"
    return "INBETWEEN"


if __name__=="__main__":
    args = arg_parse()
    in_dir = Path(RESULTS_DIR, f"{args.model_reviewed}/scoring_exp/{args.exp_name}")
    rule_dir = Path(RESULTS_DIR, f"{args.model_reviewed}/scoring_exp/{args.exp_name}/{args.rule_dir}")
    out_dir = Path(RESULTS_DIR, f"{args.model_reviewed}/scoring_exp/{args.exp_name}/{args.outdir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    keepdir = Path(out_dir, "keep")
    inbetweendir = Path(out_dir, "inbetween")
    rejectdir = Path(out_dir, "reject")

    keepdir.mkdir(parents=True, exist_ok=True)
    rejectdir.mkdir(parents=True, exist_ok=True)
    inbetweendir.mkdir(parents=True, exist_ok=True)

    verdict = ["KEEP", "REJECT", "INBETWEEN"]

    with open(Path(rule_dir, "decision_rules.json")) as f:
        rules_data = json.load(f)

    stats = []

    for pred in PRED_MAP.keys():
        pred.strip()
        
        if pred not in rules_data["rules_by_predicate"]:
            print(f"No rule for {pred}, skipping")
            continue

        df_pred = pd.read_csv(Path(in_dir, f"{pred}/pred_{args.model_reviewed}_{pred}.csv"))
        df_pred.columns.str.strip()

        pred_rules = rules_data["rules_by_predicate"][pred]
        keep_rules = pred_rules["keep"]
        reject_rules = pred_rules["reject"]
        
        df_pred["split"] = df_pred.apply(lambda row: classify_row(row, keep_rules, reject_rules), axis=1)


        for elt in verdict:
            df_elt = df_pred.loc[df_pred["split"]==elt]
            if elt=="KEEP":
                df_elt.to_csv(keepdir / f"{elt}_{args.model_reviewed}_{pred}.csv", index=False)
                ver = elt
            if elt=="REJECT":
                df_elt.to_csv(rejectdir / f"{elt}_{args.model_reviewed}_{pred}.csv", index=False)
                ver = elt
            if elt=="INBETWEEN":
                df_elt.to_csv(inbetweendir/ f"{elt}_{args.model_reviewed}_{pred}.csv", index=False)
                ver = elt
            del df_elt

        total = len(df_pred)
        len_keep = len(df_pred.loc[df_pred["split"]=="KEEP"])
        len_reject = len(df_pred.loc[df_pred["split"]=="REJECT"])
        len_inbet = len(df_pred.loc[df_pred["split"]=="INBETWEEN"])
        stats.append({
            "file": f"{ver}/pred_{args.model_reviewed}_{pred}.csv",
            "total": total,
            "keep": len_keep,
            "in_between": len_inbet,
            "reject": len_reject,
            "keep%": f"{len_keep/total*100:.1f}",
            "in_between%": f"{len_inbet/total*100:.1f}",
            "reject%": f"{len_reject/total*100:.1f}",
        })
        del df_pred
    
    stats_df = pd.DataFrame(stats)    
    total_all = stats_df["total"].sum()
    total_keep = stats_df["keep"].sum()
    total_between = stats_df["in_between"].sum()
    total_reject = stats_df["reject"].sum()
    total_keep_per = f"{total_keep/total_all*100:.1f}"
    total_between_per = f"{total_between/total_all*100:.1f}"
    total_reject_per = f"{total_reject/total_all*100:.1f}"

    new_row = pd.DataFrame([{"file": "total", "total": total_all, "keep": total_keep, "in_between": total_between, "reject": total_reject, "keep%": total_keep_per, "in_between%": total_between_per, "reject%": total_reject_per}])
    stats_df = pd.concat([stats_df, new_row], ignore_index=True)
    stats_df.to_csv(Path(out_dir, "split_summary.csv"))