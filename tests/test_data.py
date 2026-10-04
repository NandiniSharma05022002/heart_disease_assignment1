from pathlib import Path

import pandas as pd

from scripts.prepare_data import COLUMNS, prepare


def test_prepare_maps_target_to_binary(tmp_path):
    raw = tmp_path / "processed.cleveland.data"
    raw.write_text(
        "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n"
        "67,1,4,160,286,0,2,108,1,1.5,2,3,3,4\n"
    )
    out = tmp_path / "heart.csv"
    df = prepare(raw, out)
    assert list(df.columns) == COLUMNS
    assert set(df.target.unique()) == {0, 1}
    assert len(df) == 2


def test_processed_schema_if_present():
    path = Path("data/processed/heart_disease.csv")
    if path.exists():
        df = pd.read_csv(path)
        assert set(df.target.unique()).issubset({0, 1})
        assert len(df.columns) == 14
