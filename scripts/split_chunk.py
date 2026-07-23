import pandas as pd
from pathlib import Path

CHUNK_SIZE = 280000

DATA_DIR = Path(__file__).parent.parent / "data" / "top_5M_by_predicate"  
PRED_TO_CHUNK = ["hasproperty", "partof", "atlocation", "capableof"]

if __name__=="__main__":
    for pred in sorted(PRED_TO_CHUNK):
        print(f"Chunk for {pred}")
        df = pd.read_csv(f"{DATA_DIR}/quasi_top5000000_{pred}.csv")
        
        n_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for i, start in enumerate(range(0, len(df), CHUNK_SIZE), 1):
            chunk = df.iloc[start:start + CHUNK_SIZE]
            out = DATA_DIR / f"quasi_top5000000_{pred}_{i}.csv"
            chunk.to_csv(out, index=False)
            print(f"  {out.name}: {len(chunk)} rows")
        
        print(f"{pred}: {len(df)} rows -> {n_chunks} chunks")