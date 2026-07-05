"""Dataset loading and cleaning utilities for the UCI Cleveland heart dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

FEATURE_COLUMNS = COLUMN_NAMES[:-1]
TARGET_COLUMN = "target"


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the processed Cleveland file using its documented column order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, names=COLUMN_NAMES, na_values="?")
    if df.empty:
        raise ValueError("Dataset is empty")
    if list(df.columns) != COLUMN_NAMES:
        raise ValueError("Unexpected dataset schema")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert fields to numeric, impute known missing values, and binarize target."""
    cleaned = df.copy()
    for column in COLUMN_NAMES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if cleaned[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing or invalid values")

    # UCI target 0 means no disease; values 1-4 indicate disease presence.
    cleaned[TARGET_COLUMN] = (cleaned[TARGET_COLUMN] > 0).astype(int)

    # Keep preprocessing reproducible and explicit before model-level transformations.
    numeric_medians = cleaned[FEATURE_COLUMNS].median(numeric_only=True)
    cleaned[FEATURE_COLUMNS] = cleaned[FEATURE_COLUMNS].fillna(numeric_medians)

    if cleaned.isna().any().any():
        missing = cleaned.isna().sum()
        raise ValueError(f"Unresolved missing values: {missing[missing > 0].to_dict()}")
    if not set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary after cleaning")
    return cleaned


def save_clean_data(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Load, clean, validate, and persist a CSV copy."""
    cleaned = clean_data(load_raw_data(input_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the UCI Cleveland heart dataset")
    parser.add_argument("--input", default="data/raw/processed.cleveland.data")
    parser.add_argument("--output", default="data/processed/heart_disease_clean.csv")
    args = parser.parse_args()
    cleaned = save_clean_data(args.input, args.output)
    print(f"Saved {len(cleaned)} rows to {args.output}")
    print(f"Class distribution: {cleaned[TARGET_COLUMN].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
