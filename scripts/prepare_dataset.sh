#!/usr/bin/env bash
# Linux/Colab equivalent of prepare_dataset.ps1
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

SOURCE_DIR="${SOURCE_DIR:-data/raw/tmc_sources}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"

echo ">> Building dataset from ${SOURCE_DIR} -> ${OUTPUT_DIR}"
python -m tmc_llm.dataset_builder \
  --source-dir "$SOURCE_DIR" \
  --output-dir "$OUTPUT_DIR"