"""Generate professional EDA plots from the processed dataset."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "heart_disease.csv"
OUT = ROOT / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    sns.set_theme(style="whitegrid")

    numeric = df.select_dtypes(include="number").columns
    df[numeric].hist(figsize=(14, 10), bins=20)
    plt.suptitle("Heart Disease — Feature Distributions")
    plt.tight_layout()
    plt.savefig(OUT / "feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    plt.figure(figsize=(6, 4))
    df["target"].value_counts().sort_index().plot(kind="bar")
    plt.title("Class Balance")
    plt.xlabel("Heart disease (0 = absent, 1 = present)")
    plt.tight_layout()
    plt.savefig(OUT / "class_balance.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), cmap="vlag", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"EDA figures written to {OUT}")


if __name__ == "__main__":
    main()
