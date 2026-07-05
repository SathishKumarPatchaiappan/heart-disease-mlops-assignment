"""Download the official UCI Heart Disease archive and extract Cleveland data."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def download_dataset(output_dir: str | Path = "data/raw", url: str = UCI_ARCHIVE_URL) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "processed.cleveland.data"

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        urllib.request.urlretrieve(url, temp_path)
        with zipfile.ZipFile(temp_path) as archive:
            with archive.open("processed.cleveland.data") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Dataset saved to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    download_dataset(args.output_dir)


if __name__ == "__main__":
    main()
