import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path

DATA_DIR = Path("~/RecallNet/src/results/llama3.1:8b-fp8/q_final_process/02_Review").expanduser()

def train_logreg():
    df = pd.read_csv(DATA_DIR / "rev_data_wcn.csv")

    df["label"] = (df["verdict"] == "KEEP").astype(int)

    X = df[["meaningfulness", "typicality", "saliency"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Test Report")
    print(classification_report(y_test, y_pred, target_names=["REJECT", "KEEP"]))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Coefficients")
    for feat, coef in zip(X.columns, model.coef_[0]):
        print(f"  {feat}: {coef:.4f}")
    print(f"  intercept: {model.intercept_[0]:.4f}")

    df["proba_keep"] = model.predict_proba(X)[:, 1]
    df = df.sort_values("proba_keep", ascending=False)

    df["decision"] = pd.cut(
        df["proba_keep"],
        bins=[0.0, 0.3, 0.7, 1.0],
        labels=["REJECT", "INBETWEEN", "KEEP"],
        include_lowest=True
    )

    print("Distribution with thresholds (0.3 / 0.7)")
    print(df["decision"].value_counts())

    col_order = [
            "subject", "predicate", "object", "meaningfulness", "typicality", "saliency", "verdict", "proba_keep", "decision", "source_file" 
    ]
    df = df[col_order]

    out_dir = DATA_DIR / "logreg"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ranked_rev_data.csv", index=False)
    joblib.dump(model, out_dir / "logreg_model.pkl")

    return model, df

def test_on_conceptnet_overlap(model):
    cn_dir = DATA_DIR / "conceptnet_overlap" 
    dfs = [pd.read_csv(f) for f in cn_dir.rglob("*.csv")]
    cn_triples = pd.concat(dfs, ignore_index=True)

    X = cn_triples[["meaningfulness", "typicality", "saliency"]]
    cn_triples["proba_keep"] = model.predict_proba(X)[:, 1]

    cn_triples = cn_triples.sort_values("proba_keep", ascending=False).reset_index(drop=True)
    cn_triples["rank"] = cn_triples.index + 1

    n = len(cn_triples)

    # Moyenne des 1/rang
    mean_inv_rank = (1 / cn_triples["rank"]).mean()
    ideal_mean = sum(1 / i for i in range(1, n + 1)) / n
    normalized_score = mean_inv_rank / ideal_mean

    print(f"=== Test ConceptNet ===")
    print(f"Triples : {n}")
    print(f"Moyenne 1/rang      : {mean_inv_rank:.6f}")
    print(f"Moyenne idéale      : {ideal_mean:.6f}")
    print(f"Score normalisé     : {normalized_score:.4f}")

    cn_triples["decision"] = pd.cut(
        cn_triples["proba_keep"],
        bins=[0.0, 0.3, 0.7, 1.0],
        labels=["REJECT", "INBETWEEN", "KEEP"],
        include_lowest=True
    )
    print(f"\n=== Distribution ===")
    print(cn_triples["decision"].value_counts())

    return cn_triples

if __name__=="__main__":
    train_logreg()