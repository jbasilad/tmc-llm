#!/usr/bin/env bash
# Linux/Colab equivalent of merge_lora.ps1
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

BASE_MODEL="${BASE_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
ADAPTER_DIR="${ADAPTER_DIR:-models/adapters/tmc-lm-tinyllama-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-models/merged/tmc-lm-tinyllama}"

echo ">> Merging LoRA adapter ${ADAPTER_DIR} -> ${OUTPUT_DIR}"
python -m tmc_llm.merge_lora \
  --base-model "$BASE_MODEL" \
  --adapter-dir "$ADAPTER_DIR" \
  --output-dir "$OUTPUT_DIR"