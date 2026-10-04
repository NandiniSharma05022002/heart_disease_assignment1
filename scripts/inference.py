"""Run a local prediction using the packaged model artifact."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heart_disease_mlops.schema import FEATURES

MODEL_PATH = ROOT / "artifacts" / "model.pkl"

DEFAULT_INPUT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 2, "ca": 0, "thal": 3,
}


def predict(payload: dict) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}. Run training first.")
    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    row = pd.DataFrame([payload], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "label": "heart_disease_present" if prediction else "heart_disease_absent",
        "confidence": round(max(probability, 1 - probability), 6),
        "probability_positive": round(probability, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="JSON file containing one patient record")
    args = parser.parse_args()
    payload = DEFAULT_INPUT if args.input is None else json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(predict(payload), indent=2))


if __name__ == "__main__":
    main()
