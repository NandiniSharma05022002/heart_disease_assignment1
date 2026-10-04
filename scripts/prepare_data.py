"""Convert UCI Cleveland data into a clean, model-ready CSV.

The original UCI target is 0..4. The assignment asks for binary presence/absence,
so values 1..4 are mapped to 1 and 0 remains 0.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "processed.cleveland.data"
OUT_FILE = ROOT / "data" / "processed" / "heart_disease.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def prepare(input_path: Path = RAW_FILE, output_path: Path = OUT_FILE) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run scripts/download_data.py first.")
    df = pd.read_csv(input_path, header=None, names=COLUMNS, na_values="?")
    numeric = [c for c in COLUMNS if c != "target"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = (df["target"] > 0).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    print(df.isna().sum())
    return df


if __name__ == "__main__":
    prepare()
