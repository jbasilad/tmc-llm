#!/usr/bin/env bash
# Linux/Colab equivalent of train_lora.ps1
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

CONFIG_PATH="${CONFIG_PATH:-configs/train_lora.yaml}"

echo ">> Training LoRA with config ${CONFIG_PATH}"
python -m tmc_llm.train_lora --config "$CONFIG_PATH"