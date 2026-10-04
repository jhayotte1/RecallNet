import pandas as pd
from pathlib import Path

RES_DIR = Path(__file__).resolve().parent.parent / "results" / "Final_results"
OUTPUT_FILE = RES_DIR / "RecallNet.csv"
SUMMARY_FILE = RES_DIR / "RecallNet_summary.txt"

# For each dataset, the subfolders holding triples that are FINAL keeps
# (see README "Core Steps of the Cleaning Process" for the rationale):
#   2_Splitting/KEEP             -> kept right away at the first split
#   3_Filtering/KEEP             -> kept after filtering the INBETWEEN triples
#   4_Modifying/KEEP             -> re-confirmed KEEP while modifying (no change needed)
#   6_Splitting_Rescored/KEEP    -> kept after the second split (on modified+rescored triples)
#   7_Filtering_rescored/KEEP    -> kept after filtering the rescored INBETWEEN triples
DATASETS = ["Quasimodo", "Ascent"]
KEEP_STEPS = [
    "2_Splitting",
    "3_Filtering",
    "4_Modifying",
    "6_Splitting_Rescored",
    "7_Filtering_rescored",
]

KEPT_COLUMNS = ["subject", "predicate", "object", "meaningfulness", "typicality", "saliency"]

# All categories a step was split into (for the summary counts), in the order
# they should be reported. Only "KEEP" ends up in RecallNet.csv, the others
# are either dropped (REJECT/FAILED) or carried on to the next step.
STEP_CATEGORIES = {
    "2_Splitting": ["KEEP", "INBETWEEN"],
    "3_Filtering": ["KEEP", "MODIFY", "REJECT"],
    "4_Modifying": ["KEEP", "MODIFIED", "REJECT", "FAILED"],
    "6_Splitting_Rescored": ["KEEP", "INBETWEEN"],
    "7_Filtering_rescored": ["KEEP", "MODIFY", "REJECT"],
}


def count_category(dataset: str, step: str, category: str) -> int:
    cat_dir = RES_DIR / dataset / step / category
    total = 0
    for csv_file in cat_dir.glob("*/*.csv"):
        with open(csv_file, "rb") as f:
            total += sum(1 for _ in f) - 1  # minus header
    return total


def collect_step(dataset: str, step: str) -> pd.DataFrame:
    keep_dir = RES_DIR / dataset / step / "KEEP"
    rows = []
    for csv_file in sorted(keep_dir.glob("*/*.csv")):
        df = pd.read_csv(csv_file)
        df = df[KEPT_COLUMNS]
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=KEPT_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def build_recallnet() -> pd.DataFrame:
    parts = []
    for dataset in DATASETS:
        for step in KEEP_STEPS:
            df = collect_step(dataset, step)
            df["source_dataset"] = dataset
            df["source_step"] = step
            print(f"{dataset:10s} {step:25s} {len(df):>7} triples")
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_summary(recallnet: pd.DataFrame) -> str:
    lines = ["RecallNet - triple counts per step and category", "=" * 50, ""]
    for step in KEEP_STEPS:
        categories = STEP_CATEGORIES[step]
        lines.append(f"Step {step}")
        for dataset in DATASETS:
            counts = {cat: count_category(dataset, step, cat) for cat in categories}
            total = sum(counts.values())
            detail = "  ".join(
                f"{cat}={counts[cat]} ({counts[cat]/total*100:.1f}%)" if total else f"{cat}=0"
                for cat in categories
            )
            lines.append(f"  {dataset:10s}: {detail}")
        lines.append("")

    lines.append("Final RecallNet totals (KEEP triples only)")
    for dataset in DATASETS:
        total = len(recallnet[recallnet["source_dataset"] == dataset])
        lines.append(f"  {dataset:10s}: {total}")
    lines.append(f"  {'TOTAL':10s}: {len(recallnet)}")

    return "\n".join(lines)


if __name__ == "__main__":
    recallnet = build_recallnet()
    print(f"TOTAL: {len(recallnet)} triples")
    recallnet.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

    summary = build_summary(recallnet)
    SUMMARY_FILE.write_text(summary + "\n")
    print(summary)
    print(f"Saved to {SUMMARY_FILE}")
