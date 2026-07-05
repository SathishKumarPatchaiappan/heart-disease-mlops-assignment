from src.data import FEATURE_COLUMNS


def test_model_predicts_class_and_probability(test_model_path, cleaned_df):
    import joblib

    model = joblib.load(test_model_path)
    sample = cleaned_df[FEATURE_COLUMNS].head(3)
    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)
    assert len(predictions) == 3
    assert probabilities.shape == (3, 2)
    assert set(predictions).issubset({0, 1})
