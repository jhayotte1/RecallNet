import pandas as pd
import argparse
import json

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
FEATURES = ["meaningfulness", "typicality", "saliency"]


def arg_parser():
    parser = argparse.ArgumentParser(description="RecallNet rule mining from review data")
    parser.add_argument("--exp-name", type=str, required=True, help="Directory of experiment name")
    parser.add_argument("--review-dir", type=str, required=True, help="Directory with review CSVs")
    parser.add_argument("--model-reviewed", type=str, default="llama3.1:8b", help="Name of the model being reviewed")
    parser.add_argument("--max-depth", type=int, default=3, help="Max tree depth (default: 3)")
    parser.add_argument("--min-samples-leaf", type=int, default=5, help="Min samples per leaf (default: 5)")
    parser.add_argument("--out-dir", type=str, default="0_RULE", help="Output directory for rules files")
    args = parser.parse_args()
    return args


def load_csvs(in_dir: Path):
    csvs = in_dir.rglob("*.csv")
    data={}
    for csv_path in csvs:
        pred = csv_path.stem.replace("review_", "").replace(" ", "")
        df_pred = pd.read_csv(csv_path)
        df_pred.columns = df_pred.columns.str.strip()
        data[pred] = df_pred
    return data

def build_tree_for_pred(pred: str, df: pd.DataFrame, max_depth: int = 3, min_samples_leaf: int = 5, random_state:int = 42):
    X = df[FEATURES]
    y = df["verdict"].str.upper()

    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    dt.fit(X, y)

    print(export_text(dt, feature_names=FEATURES))
    print(f"Classes: {list(dt.classes_)}")
    print(f"Accuracy: {dt.score(X, y):.3f}")

    tree = dt.tree_
    classes = list(dt.classes_)

    for i in range(tree.node_count):
        is_leaf = tree.feature[i]==-2
        n = tree.n_node_samples[i]
        proportions=tree.value[i][0]
        counts = (proportions * n).round().astype(int)

        if is_leaf:
            majority = classes[counts.argmax()]
            dist = {c: int(counts[j]) for j, c in enumerate(classes)}
            print(f"Node {i}: LEAF -> {majority} (n={n}, {dist})")
        else:
            feat = FEATURES[tree.feature[i]]
            thresh = tree.threshold[i]
            print(f"Node {i}: {feat} <= {thresh:.1f} ? (n={n})")        
    return dt

def extract_rules(dt: DecisionTreeClassifier):
    tree = dt.tree_
    classes = dt.classes_
    rules=[]
    def rec_walk(node, conditions):
        if tree.feature[node]==-2:
            n = tree.n_node_samples[node]
            counts = (tree.value[node][0] * n).round().astype(int)
            majority = classes[counts.argmax()]
            dist = {c: int(counts[j]) for j, c in enumerate(classes)}
            rules.append({
                "conditions": list(conditions),
                "verdict": majority,
                "confidence": round(int(counts.max()) / n, 3),
                "support": int(n),
                "distribution": dist,
            })
            return
        feat = FEATURES[tree.feature[node]]
        thresh = round(tree.threshold[node], 1)
        rec_walk(tree.children_left[node], conditions + [f"{feat} <= {thresh}"])
        rec_walk(tree.children_right[node], conditions + [f"{feat} > {thresh}"])
    
    rec_walk(0, [])
    return rules


if __name__=="__main__":
    args = arg_parser()
    in_dir = Path(RESULTS_DIR, f"{args.model_reviewed}/scoring_exp/{args.exp_name}/{args.review_dir}")
    out_dir = Path(RESULTS_DIR, f"{args.model_reviewed}/scoring_exp/{args.exp_name}/{args.out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_csvs(in_dir=in_dir)
    all_results = {}
    for pred, df_pred in data.items():
        print(f"Rule mining for predicate : {pred}")
        dt = build_tree_for_pred(pred=pred, df=df_pred, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
        rules = extract_rules(dt=dt)
        all_results[pred] = {
            "predicate": pred,
            "status": "ok",
            "n_samples": len(df_pred),
            "distribution": df_pred["verdict"].value_counts().to_dict(),
            "accuracy": round(dt.score(df_pred[FEATURES], df_pred["verdict"].str.upper()), 3),
            "tree_text": export_text(dt, feature_names=FEATURES),
            "rules": rules,
            "keep": [r for r in rules if r["verdict"] == "KEEP"],
            "reject": [r for r in rules if r["verdict"] == "REJECT"],
            "uncertain": [r for r in rules if r["verdict"] == "UNCERTAIN"],
        }
    
     # Export JSON
    export = {
        "metadata": {
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "features": FEATURES,
        },
        "rules_by_predicate": all_results,
    }
    json_path = out_dir / "decision_rules.json"
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"\nJSON exported to {json_path}")

    # Export TXT lisible
    txt_path = out_dir / "decision_rules.txt"
    with open(txt_path, "w") as f:
        for pred, result in sorted(all_results.items()):
            f.write(f"── {pred} ({result['n_samples']} samples, acc={result['accuracy']}) ──\n")
            f.write(f"  Distribution: {result['distribution']}\n\n")
            for rule in result["rules"]:
                conds = " AND ".join(rule["conditions"]) if rule["conditions"] else "(root)"
                f.write(f"  {rule['verdict']:>10}  |  {conds}\n")
                f.write(f"             |  conf={rule['confidence']}  n={rule['support']}  {rule['distribution']}\n")
            f.write("\n")
    print(f"TXT exported to {txt_path}")