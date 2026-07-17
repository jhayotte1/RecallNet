import pandas as pd
from fnmatch import fnmatch
from pathlib import Path
import json
import re
import argparse



def parse_condition(cond: str) -> tuple[str, str, float]:
    m = re.match(r"(\w+)\s*(<=|>)\s*([\d.]+)", cond)
    if not m:
        raise ValueError(f"Cannot parse condition: {cond}")
    return m.group(1), m.group(2), float(m.group(3))


def matches_rule(row: pd.Series, conditions: list[str]) -> bool:
    for cond in conditions:
        feat, op, thresh = parse_condition(cond)
        val = row[feat]
        if op == "<=" and not (val <= thresh):
            return False
        if op == ">" and not (val > thresh):
            return False
    return True



def split_triples(df: pd.DataFrame, keep_rules: list, reject_rules: list) -> dict[str, pd.DataFrame]:
    """
    Apply mined rules. Output only keep and in_between.
    Everything not KEEP = in_between (includes REJECT and UNCERTAIN zones).
    """
    is_keep = pd.Series(False, index=df.index)

    for rule in keep_rules:
        mask = df.apply(lambda row: matches_rule(row, rule["conditions"]), axis=1)
        is_keep = is_keep | mask

    return {
        "keep": df[is_keep],
        "in_between": df[~is_keep],
    }

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
    "usedfor": "used for",
}


def extract_predicate(csv_path: Path, model_name: str) -> str | None:
    stem = csv_path.stem
    for prefix in [f"pred_{model_name}_", "pred_"]:
        stem = stem.replace(prefix, "")
    return PRED_MAP.get(stem, stem)


def parse_args():
    parser = argparse.ArgumentParser(description="Split scored dataset using mined rules")
    parser.add_argument("--rules", type=str, required=True, help="Path to decision_rules.json")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Model name used for scoring")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    INPUT_DIR = Path(__file__).parent.parent / "results" / args.model / "scoring_exp" / args.exp_name
    OUTPUT_DIR = INPUT_DIR / "2_SPLITED"
    EXCLUDE_PATTERNS = ["*_SPLITED", "*_REVIEW", "*_SAMPLING"]

    with open(args.rules) as f:
        rules_data = json.load(f)

    rules_by_pred = rules_data["rules_by_predicate"]
    print(f"Loaded rules for {len(rules_by_pred)} predicates from {args.rules}")
    print(f"Input: {INPUT_DIR}\n")

    csvs = [
        p for p in sorted(INPUT_DIR.rglob("pred_*.csv"))
        if not any(fnmatch(part, pat) for part in p.parts for pat in EXCLUDE_PATTERNS)
    ]

    if not csvs:
        print(f"No scored CSVs found in {INPUT_DIR}")
        exit(1)

    stats = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        pred = extract_predicate(csv_path, args.model)

        if pred not in rules_by_pred or rules_by_pred[pred]["status"] != "ok":
            print(f"  ⚠ No rules for '{pred}', skipping {csv_path.name}")
            continue

        pred_rules = rules_by_pred[pred]
        splits = split_triples(df, pred_rules.get("keep", []), pred_rules.get("reject", []))

        for label, split_df in splits.items():
            out_path = OUTPUT_DIR / label
            out_path.mkdir(parents=True, exist_ok=True)
            split_df.to_csv(out_path / csv_path.name, index=False)

        total = len(df)
        stats.append({
            "predicate": pred,
            "file": csv_path.name,
            "total": total,
            "keep": len(splits["keep"]),
            "in_between": len(splits["in_between"]),
            "keep%": f"{len(splits['keep'])/total*100:.1f}",
            "in_between%": f"{len(splits['in_between'])/total*100:.1f}",
        })

    stats_df = pd.DataFrame(stats)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    total_all = stats_df["total"].sum()
    total_keep = stats_df["keep"].sum()
    total_between = stats_df["in_between"].sum()

    print("\n" + stats_df.to_string(index=False))
    print(f"\nTotals:")
    print(f"  KEEP:       {total_keep:>6} ({total_keep/total_all*100:.1f}%)")
    print(f"  IN_BETWEEN: {total_between:>6} ({total_between/total_all*100:.1f}%)")
    print(f"  TOTAL:      {total_all:>6}")