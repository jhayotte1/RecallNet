import pandas as pd
from pathlib import Path
 
DATA_DIR = Path(__file__).parent.parent / "data"

def parse_gent(input_path, output_path=None):
    df_raw = pd.read_csv(input_path, sep="\t")
 
    df = df_raw['triple'].str.split(',', n=2, expand=True)
    df.columns = ['subject', 'predicate', 'object']
    df['frequency'] = df_raw['frequency']
 
    df = df.dropna(subset=['subject', 'predicate', 'object'])

    for col in ['subject', 'predicate', 'object']:
        df[col] = df[col].str.strip()
 
    df = df.sort_values('frequency', ascending=False).reset_index(drop=True)
 
    if output_path:
        df.to_csv(output_path, sep="\t", index=False)
    return df
 
 
def load_quasimodo(raw_path=f"{DATA_DIR}/quasimodo_gent_lm_based_inv_top10.tsv",
                   parsed_path=f"{DATA_DIR}/quasimodo_parsed.tsv"):
    parsed = Path(parsed_path)
    if parsed.exists():
        print("Loading Quasimodo...")
        return pd.read_csv(parsed, sep="\t")
    print("Parsing Quasimodo from raw TSV...")
    return parse_gent(raw_path, parsed_path)
 
 
def load_ascent(raw_path=f"{DATA_DIR}/ascent_gent_lm_based_inv_top10.tsv",
                parsed_path=f"{DATA_DIR}/ascent_parsed.tsv"):
    parsed = Path(parsed_path)
    if parsed.exists():
        print("Loading Ascent...")
        return pd.read_csv(parsed, sep="\t")
    print("Parsing Ascent from raw TSV...")
    return parse_gent(raw_path, parsed_path)
 
 
def load_conceptnet(path=f"{DATA_DIR}/conceptnet_csk_spor.tsv"):
    print("Loading ConceptNet...")
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["subject", "predicate", "object", "score"]
    return df
 
 
if __name__ == "__main__":
    df_quasi = load_quasimodo() 
    df_ascent = load_ascent()
 
    df_cn = load_conceptnet()
