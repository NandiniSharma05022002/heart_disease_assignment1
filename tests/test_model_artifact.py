import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.pkl"


def test_model_pickle_exists_and_loads():
    assert MODEL_PATH.exists(), "Expected artifacts/model.pkl to exist"
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
