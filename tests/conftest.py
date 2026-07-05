from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, clean_data, load_raw_data
from src.features import build_pipeline


@pytest.fixture(scope="session")
def cleaned_df():
    return clean_data(load_raw_data("data/raw/processed.cleveland.data"))


@pytest.fixture(scope="session")
def test_model_path(tmp_path_factory, cleaned_df):
    path = Path(tmp_path_factory.mktemp("models")) / "test_pipeline.joblib"
    model = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    model.fit(cleaned_df[FEATURE_COLUMNS], cleaned_df[TARGET_COLUMN])
    joblib.dump(model, path)
    return path
