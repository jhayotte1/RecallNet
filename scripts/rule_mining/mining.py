"""
RecallNet Rule Mining
---------------------
Loads review CSVs (scores + verdicts from the large model),
fits a shallow DecisionTree per predicate to derive explicit
filtering rules, and exports them as JSON + readable .txt.

Usage:
    python rule_mining.py --review-dir results/llama3.1:8b/scoring_exp/exp00_LG/0_REVIEW
    python rule_mining.py --review-dir <path> --max-depth 3 --out-dir rules/
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

FEATURES = ["meaningfulness", "typicality", "saliency"]
LABELS = ["KEEP", "UNCERTAIN", "REJECT"]



def load_reviews(review_dir: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for csv_path in sorted(review_dir.rglob("review_*.csv")):
        pred = csv_path.stem.replace("review_", "")
        df = pd.read_csv(csv_path)
        df["verdict"] = df["verdict"].str.strip().str.upper()
        df = df[df["verdict"].isin(LABELS)]
        if len(df) > 0:
            data[pred] = df
    return data



def extract_rules(tree: DecisionTreeClassifier, feature_names: list[str], class_names: list[str]) -> list[dict]:
    """Walk the fitted tree and extract every leaf as a rule dict."""
    tree_ = tree.tree_
    rules = []

    def recurse(node, conditions):
        if tree_.feature[node] == -2:  # leaf
            counts = tree_.value[node][0]
            total = counts.sum()
            majority_idx = int(np.argmax(counts))
            majority_class = class_names[majority_idx]
            confidence = counts[majority_idx] / total if total > 0 else 0
            distribution = {class_names[i]: int(counts[i]) for i in range(len(class_names))}
            rules.append({
                "conditions": list(conditions),
                "verdict": majority_class,
                "confidence": round(confidence, 3),
                "support": int(total),
                "distribution": distribution,
            })
            return
        feat = feature_names[tree_.feature[node]]
        threshold = round(tree_.threshold[node], 1)
        recurse(tree_.children_left[node], conditions + [f"{feat} <= {threshold}"])
        recurse(tree_.children_right[node], conditions + [f"{feat} > {threshold}"])

    recurse(0, [])
    return rules


def mine_rules_for_predicate(
    df: pd.DataFrame,
    pred: str,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
) -> dict:
    """Fit a decision tree and return all rules."""
    X = df[FEATURES].values
    y = df["verdict"].values

    unique = np.unique(y)
    if len(unique) < 2:
        return {
            "predicate": pred,
            "status": "skipped",
            "reason": f"Only one class present: {unique[0]}",
            "n_samples": len(df),
        }

    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    dt.fit(X, y)

    tree_text = export_text(dt, feature_names=FEATURES, show_weights=True)
    present_classes = sorted(set(y))
    rules = extract_rules(dt, FEATURES, present_classes)

    y_pred = dt.predict(X)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    dist = df["verdict"].value_counts().to_dict()

    # Group rules by verdict
    keep_rules = [r for r in rules if r["verdict"] == "KEEP"]
    reject_rules = [r for r in rules if r["verdict"] == "REJECT"]
    uncertain_rules = [r for r in rules if r["verdict"] == "UNCERTAIN"]

    return {
        "predicate": pred,
        "status": "ok",
        "n_samples": len(df),
        "distribution": dist,
        "tree_text": tree_text,
        "rules": rules,
        "keep": keep_rules,
        "reject": reject_rules,
        "uncertain": uncertain_rules,
        "accuracy": round(report["accuracy"], 3),
        "classification_report": {
            k: v for k, v in report.items()
            if k in LABELS
        },
    }


def write_rules_txt(all_results: dict, out_path: Path, args):
    with open(out_path, "w") as f:
        f.write("RecallNet — Mined Decision Rules\n")
        f.write(f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}\n")
        f.write("=" * 60 + "\n\n")

        for pred, result in sorted(all_results.items()):
            f.write(f"── {pred} ({result['n_samples']} samples) ──\n")

            if result["status"] != "ok":
                f.write(f"  SKIPPED: {result['reason']}\n\n")
                continue

            f.write(f"  Distribution: {result['distribution']}\n")
            f.write(f"  Accuracy: {result['accuracy']}\n\n")

            for rule in result["rules"]:
                conds = " AND ".join(rule["conditions"]) if rule["conditions"] else "(root)"
                f.write(f"  {rule['verdict']:>10}  |  {conds}\n")
                f.write(f"             |  conf={rule['confidence']}  n={rule['support']}  {rule['distribution']}\n")

            f.write("\n")

    print(f"✓ Readable rules written to {out_path}")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="RecallNet rule mining from review data")
    parser.add_argument("--review-dir", type=str, required=True, help="Directory with review CSVs")
    parser.add_argument("--max-depth", type=int, default=3, help="Max tree depth (default: 3)")
    parser.add_argument("--min-samples-leaf", type=int, default=5, help="Min samples per leaf (default: 5)")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for rules files")
    args = parser.parse_args()

    review_dir = Path(args.review_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reviews from {review_dir}...\n")
    data = load_reviews(review_dir)

    if not data:
        print("No review CSVs found!")

    all_results = {}

    for pred, df in sorted(data.items()):
        print(f"\n{'='*60}")
        print(f"  {pred.upper()}")
        print(f"{'='*60}")

        result = mine_rules_for_predicate(df, pred, args.max_depth, args.min_samples_leaf)
        all_results[pred] = result

        if result["status"] == "ok":
            print(f"\nDistribution: {result['distribution']}")
            print(f"Accuracy: {result['accuracy']}")
            print(f"\nDecision Tree:\n{result['tree_text']}")

            print(f"\n→ KEEP rules: {len(result['keep'])}")
            for r in result["keep"]:
                print(f"    {' AND '.join(r['conditions'])}  (conf={r['confidence']}, n={r['support']})")
            print(f"→ REJECT rules: {len(result['reject'])}")
            for r in result["reject"]:
                print(f"    {' AND '.join(r['conditions'])}  (conf={r['confidence']}, n={r['support']})")
            print(f"→ UNCERTAIN rules: {len(result['uncertain'])}")
            for r in result["uncertain"]:
                print(f"    {' AND '.join(r['conditions'])}  (conf={r['confidence']}, n={r['support']})")
        else:
            print(f"  Skipped: {result['reason']}")

    export = {
        "metadata": {
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "features": FEATURES,
            "labels": LABELS,
        },
        "rules_by_predicate": all_results,
    }

    json_path = out_dir / "decision_rules.json"
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON rules exported to {json_path}")

    txt_path = out_dir / "decision_rules.txt"
    write_rules_txt(all_results, txt_path, args)
