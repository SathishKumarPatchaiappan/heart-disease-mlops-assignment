import pandas as pd

from src.data import COLUMN_NAMES, clean_data, load_raw_data


def test_load_raw_data_has_expected_shape():
    df = load_raw_data("data/raw/processed.cleveland.data")
    assert df.shape == (303, 14)
    assert list(df.columns) == COLUMN_NAMES


def test_clean_data_resolves_missing_values_and_binarizes_target():
    raw = load_raw_data("data/raw/processed.cleveland.data")
    cleaned = clean_data(raw)
    assert not cleaned.isna().any().any()
    assert set(cleaned["target"].unique()) == {0, 1}
    assert all(pd.api.types.is_numeric_dtype(cleaned[col]) for col in cleaned.columns)
