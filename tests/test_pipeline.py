import pandas as pd

from src.heart_disease_mlops.pipeline import make_logistic_regression, make_random_forest
from src.heart_disease_mlops.schema import FEATURES


def sample_data():
    return pd.DataFrame([
        [63,1,1,145,233,1,2,150,0,2.3,3,0,6],
        [45,0,2,120,220,0,1,170,0,0.2,1,0,3],
        [58,1,4,140,250,0,2,130,1,2.0,2,1,7],
        [39,0,3,110,190,0,0,180,0,0.0,1,0,3],
    ], columns=FEATURES), pd.Series([1,0,1,0])


def test_logistic_pipeline_fits_and_predicts():
    X, y = sample_data()
    model = make_logistic_regression()
    model.fit(X, y)
    assert model.predict(X).shape == (4,)
    assert model.predict_proba(X).shape == (4, 2)


def test_random_forest_pipeline_fits_and_predicts():
    X, y = sample_data()
    model = make_random_forest(n_estimators=10)
    model.fit(X, y)
    assert model.predict(X).shape == (4,)
    assert model.predict_proba(X).shape == (4, 2)
