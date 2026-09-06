from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def ensure_valid_config(model, output_dir: Path) -> None:
    ConfigClass = getattr(model, "config", None)
    if ConfigClass is None:
        return
    cfg = json.loads(ConfigClass.to_json_string())
    cfg.setdefault("model_type", "llama")
    (output_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def merge_lora(base_model: str, adapter_dir: Path, output_dir: Path) -> None:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)

    ensure_valid_config(merged, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into TinyLlama.")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/adapters/tmc-lm-tinyllama-lora"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/merged/tmc-lm-tinyllama"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_lora(args.base_model, args.adapter_dir, args.output_dir)


if __name__ == "__main__":
    main()

