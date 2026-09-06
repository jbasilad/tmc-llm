#!/usr/bin/env bash
# Linux/Colab equivalent of convert_to_gguf.ps1
#
# Replaces the Docker llama.cpp step: clones llama.cpp, builds only the
# llama-quantize binary (CPU-only for maximum Colab compatibility), converts
# the merged HF model to F16 GGUF, then quantizes it (default Q4_K_M).
#
# Env overrides:
#   LLAMACPP_DIR, MERGED_MODEL_DIR, OUTPUT_DIR, QUANTIZATION, BUILD_JOBS
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

LLAMACPP_DIR="${LLAMACPP_DIR:-external/llama.cpp}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-models/merged/tmc-lm-tinyllama}"
OUTPUT_DIR="${OUTPUT_DIR:-models/gguf}"
QUANTIZATION="${QUANTIZATION:-Q4_K_M}"
BUILD_JOBS="${BUILD_JOBS:-2}"

MODEL_NAME="tmc-lm-tinyllama"
F16_OUT="$OUTPUT_DIR/${MODEL_NAME}-f16.gguf"
QUANT_SUFFIX="$(printf '%s' "${QUANTIZATION,,}")"
QUANT_OUT="$OUTPUT_DIR/${MODEL_NAME}-${QUANT_SUFFIX}.gguf"

if [ ! -f "$MERGED_MODEL_DIR/config.json" ] || [ ! -f "$MERGED_MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: merged model not found in $MERGED_MODEL_DIR" >&2
    echo "Run scripts/merge_lora.sh first." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 1. Tools: cmake and the gguf package needed by convert_hf_to_gguf.py
if ! command -v cmake >/dev/null 2>&1; then
    echo ">> cmake not found, installing via pip"
    python -m pip install -q cmake
fi
if ! python -c "import gguf" >/dev/null 2>&1; then
    echo ">> Installing gguf package"
    python -m pip install -q gguf
fi

# 2. Ensure llama.cpp source exists
if [ ! -d "$LLAMACPP_DIR/.git" ]; then
    echo ">> Cloning llama.cpp into $LLAMACPP_DIR"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMACPP_DIR"
fi

# 3. Build llama-quantize (CPU-only build; quantization is fast on CPU)
QUANTIZE_BIN="$LLAMACPP_DIR/build/bin/llama-quantize"
if [ ! -x "$QUANTIZE_BIN" ]; then
    echo ">> Building llama-quantize (this takes a few minutes)"
    cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=OFF \
      -DGGML_NATIVE=OFF
    cmake --build "$LLAMACPP_DIR/build" --config Release --target llama-quantize -j"$BUILD_JOBS"
fi

# 4. Convert merged HF model -> F16 GGUF
if [ ! -f "$F16_OUT" ]; then
    echo ">> Converting HF model to F16 GGUF"
    python "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$MERGED_MODEL_DIR" \
      --outfile "$F16_OUT" \
      --outtype f16
else
    echo ">> $F16_OUT already exists, skipping conversion (delete it to redo)"
fi

# 5. Quantize
if [ ! -f "$QUANT_OUT" ]; then
    echo ">> Quantizing to $QUANTIZATION"
    "$QUANTIZE_BIN" "$F16_OUT" "$QUANT_OUT" "$QUANTIZATION"
else
    echo ">> $QUANT_OUT already exists, skipping quantization (delete it to redo)"
fi

# 6. Verify
echo ">> Verifying GGUF files"
python -m tmc_llm.gguf_check --path "$QUANT_OUT"
echo ">> Done: $QUANT_OUT"