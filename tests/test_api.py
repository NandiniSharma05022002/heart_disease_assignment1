from fastapi.testclient import TestClient

from src.heart_disease_mlops import api


class DummyModel:
    def predict_proba(self, X):
        import numpy as np

        return np.array([[0.25, 0.75]])


def payload():
    return {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }


def test_predict(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: DummyModel())
    client = TestClient(api.app)
    response = client.post("/predict", json=payload())

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["probability_positive"] == 0.75


def test_predict_rejects_unknown_category():
    client = TestClient(api.app)
    invalid = payload() | {"thal": 1}
    response = client.post("/predict", json=invalid)

    assert response.status_code == 422


def test_health(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: DummyModel())
    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_metrics_endpoint(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: DummyModel())
    client = TestClient(api.app)
    client.post("/predict", json=payload())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "model_predictions_total" in response.text
