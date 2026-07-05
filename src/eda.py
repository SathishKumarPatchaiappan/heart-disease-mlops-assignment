"""Generate assignment-ready exploratory data-analysis figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.data import TARGET_COLUMN, clean_data, load_raw_data

RAW_PATH = Path("data/raw/processed.cleveland.data")
OUTPUT_DIR = Path("reports/figures")


def run_eda(raw_path: Path = RAW_PATH, output_dir: Path = OUTPUT_DIR) -> None:
    raw = load_raw_data(raw_path)
    cleaned = clean_data(raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Missing value analysis from raw data.
    missing = raw.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(missing.index, missing.values)
    plt.title("Missing Values by Feature")
    plt.ylabel("Missing value count")
    plt.xlabel("Feature")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=160)
    plt.close()

    # Class balance.
    plt.figure(figsize=(6, 4))
    class_counts = cleaned[TARGET_COLUMN].value_counts().sort_index()
    plt.bar(class_counts.index.astype(str), class_counts.values)
    plt.title("Heart Disease Class Distribution")
    plt.xlabel("Target (0 = No disease, 1 = Disease)")
    plt.ylabel("Patient count")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close()

    # Histograms.
    cleaned[["age", "trestbps", "chol", "thalach", "oldpeak"]].hist(
        bins=20, figsize=(12, 8)
    )
    plt.suptitle("Numerical Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_histograms.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Correlation heatmap.
    plt.figure(figsize=(12, 9))
    correlation = cleaned.corr(numeric_only=True)
    image = plt.imshow(correlation, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(image, label="Correlation")
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.index)), correlation.index)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    # Feature relationship example.
    plt.figure(figsize=(8, 5))
    groups = [
        cleaned.loc[cleaned[TARGET_COLUMN] == class_value, "thalach"]
        for class_value in sorted(cleaned[TARGET_COLUMN].unique())
    ]
    plt.boxplot(groups, tick_labels=["No disease", "Disease"])
    plt.title("Maximum Heart Rate by Disease Class")
    plt.xlabel("Target")
    plt.ylabel("Maximum heart rate achieved")
    plt.tight_layout()
    plt.savefig(output_dir / "thalach_by_target.png", dpi=160)
    plt.close()

    print(f"EDA figures saved to {output_dir}")


if __name__ == "__main__":
    run_eda()
