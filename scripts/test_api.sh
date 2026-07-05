#!/usr/bin/env bash
set -euo pipefail
curl -sS -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data @sample_request.json
printf "\n"
