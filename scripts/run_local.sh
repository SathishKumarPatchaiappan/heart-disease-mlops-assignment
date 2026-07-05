#!/usr/bin/env bash
set -euo pipefail

python -m src.data
python -m src.eda
python -m src.train
uvicorn app.main:app --host 0.0.0.0 --port 8000
