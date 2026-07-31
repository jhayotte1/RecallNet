import pandas as pd
from pathlib import Path

from loading_dataset import load_quasimodo, load_ascent

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "ascent_chunked"

TOP_N_ALREADY_DONE = 5_000_000
CHUNK_SIZE = 400_000
TRIPLE_COLS = ["subject", "predicate", "object"]


CONCEPTNET_PREDICATES = {
    "at location", "capable of", "causes", "cause desire",
    "created by", "defined as", "desires", "distinct from",
    "has a", "has subevent", "has first subevent", "has last subevent",
    "has prerequisite", "has property", "made of", "manner of",
    "motivated by goal", "part of", "receives action", "used for",
}


def parse_predicate(pred: str) -> str:
    return pred.strip().lower().replace(" ", "")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    for col in TRIPLE_COLS:
        df[col] = df[col].str.strip().str.lower()
    return df


def chunk_quasimodo():
    df = load_quasimodo()
    print(f"Total: {len(df)} rows")

    df_remaining = df.iloc[TOP_N_ALREADY_DONE:]
    print(f"After skipping top {TOP_N_ALREADY_DONE}: {len(df_remaining)} rows")
    del df

    skipped = {}
    for pred, group in df_remaining.groupby("predicate"):
        if pred not in CONCEPTNET_PREDICATES:
            skipped[pred] = len(group)
            continue

        pred_parsed = parse_predicate(pred)
        pred_dir = OUT_DIR / pred_parsed
        pred_dir.mkdir(parents=True, exist_ok=True)

        n_chunks = (len(group) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for i, start in enumerate(range(0, len(group), CHUNK_SIZE), 1):
            chunk = group.iloc[start:start + CHUNK_SIZE]
            out_path = pred_dir / f"quasi_{pred_parsed}_{i}.csv"
            chunk.to_csv(out_path, index=False)
            print(f"  {out_path.name}: {len(chunk)} rows")

        print(f"{pred} -> {pred_parsed}/: {len(group)} rows, {n_chunks} chunks\n")

    if skipped:
        print(f"\n=== Skipped {len(skipped)} non-ConceptNet predicates ===")
        for pred, count in sorted(skipped.items(), key=lambda x: -x[1]):
            print(f"  {pred}: {count} rows")
        print(f"Total skipped: {sum(skipped.values())} rows")

    print("\nDone")


def chunk_ascent():
    df_quasi = normalize(load_quasimodo())
    print(f"Quasimodo: {len(df_quasi)} rows")

    # Garder uniquement les clés uniques pour le dedup
    quasi_keys = df_quasi[TRIPLE_COLS].drop_duplicates()
    del df_quasi
    print(f"Quasimodo unique triples: {len(quasi_keys)}")

    print("\nLoading Ascent...")
    df_ascent = normalize(load_ascent())
    print(f"Ascent: {len(df_ascent)} rows")

    # Dedup via merge anti-join
    before = len(df_ascent)
    df_ascent = df_ascent.merge(quasi_keys, on=TRIPLE_COLS, how="left", indicator=True)
    df_ascent = df_ascent[df_ascent["_merge"] == "left_only"].drop(columns="_merge")
    del quasi_keys
    print(f"Removed {before - len(df_ascent)} overlapping triples")
    print(f"Ascent after dedup: {len(df_ascent)} rows")

    skipped = {}
    for pred, group in df_ascent.groupby("predicate"):
        if pred not in CONCEPTNET_PREDICATES:
            skipped[pred] = len(group)
            continue

        pred_parsed = parse_predicate(pred)
        pred_dir = OUT_DIR / pred_parsed
        pred_dir.mkdir(parents=True, exist_ok=True)

        n_chunks = (len(group) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for i, start in enumerate(range(0, len(group), CHUNK_SIZE), 1):
            chunk = group.iloc[start : start + CHUNK_SIZE]
            out_path = pred_dir / f"ascent_{pred_parsed}_{i}.csv"
            chunk.to_csv(out_path, index=False)
            print(f"  {out_path.name}: {len(chunk)} rows")

        print(f"{pred} -> {pred_parsed}/: {len(group)} rows, {n_chunks} chunks\n")

    if skipped:
        print(f"\n=== Skipped {len(skipped)} non-ConceptNet predicates ===")
        for pred, count in sorted(skipped.items(), key=lambda x: -x[1]):
            print(f"  {pred}: {count} rows")
        print(f"Total skipped: {sum(skipped.values())} rows")

    print("\nDone")


if __name__=="__main__":
    chunk_ascent()