#!/usr/bin/env python3
"""
Extract and repack Qwen3-30B-A3B safetensors into pocket-moe format.

Separates weights into:
  - resident.bin:  non-expert weights (embeddings, attention, norms, shared experts)
  - experts.bin:   expert weights packed contiguously per (layer, expert)
  - config.json:   model config + weight layout metadata

Usage:
  python extract_weights.py --model-dir /path/to/Qwen3-30B-A3B --output-dir ./models/qwen3-30b
  python extract_weights.py --model-dir /path/to/Qwen3-30B-A3B --output-dir ./models/qwen3-30b --quant q4
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract weights for pocket-moe")
    parser.add_argument(
        "--model-dir", required=True, help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for pocket-moe weights"
    )
    parser.add_argument(
        "--quant",
        choices=["fp16", "q8", "q4"],
        default="q4",
        help="Quantization level (default: q4)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        print(f"Error: model directory {model_dir} does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    with open(config_path) as f:
        hf_config = json.load(f)

    print(f"Model: {hf_config.get('model_type', 'unknown')}")
    print(f"Hidden size: {hf_config.get('hidden_size', '?')}")
    print(f"Num layers: {hf_config.get('num_hidden_layers', '?')}")
    print(
        f"Num experts: {hf_config.get('num_experts', hf_config.get('num_local_experts', '?'))}"
    )
    print(f"Active experts: {hf_config.get('num_experts_per_tok', '?')}")
    print(f"Quantization: {args.quant}")
    print()

    # TODO: Implement weight extraction
    # 1. Load safetensors index
    # 2. Classify each tensor as resident vs expert
    # 3. Pack resident weights into resident.bin (mmap-friendly layout)
    # 4. Pack expert weights into experts.bin (contiguous per layer,expert)
    # 5. Quantize if needed
    # 6. Write config.json with offsets and sizes

    print("Weight extraction not yet implemented.")
    print("This script will:")
    print("  1. Separate expert weights from non-expert weights")
    print("  2. Pack experts contiguously: experts[layer][expert] = gate + up + down")
    print(f"  3. Quantize to {args.quant}")
    print(f"  4. Write to {output_dir}/")
    print()

    # Write pocket-moe config
    pocket_config = {
        "model_type": hf_config.get("model_type", "qwen3_moe"),
        "vocab_size": hf_config.get("vocab_size"),
        "hidden_size": hf_config.get("hidden_size"),
        "num_layers": hf_config.get("num_hidden_layers"),
        "num_attention_heads": hf_config.get("num_attention_heads"),
        "num_kv_heads": hf_config.get("num_key_value_heads"),
        "intermediate_size": hf_config.get("intermediate_size"),
        "max_seq_len": hf_config.get("max_position_embeddings", 4096),
        "num_experts": hf_config.get("num_experts", hf_config.get("num_local_experts")),
        "num_active_experts": hf_config.get("num_experts_per_tok"),
        "num_shared_experts": hf_config.get("num_shared_experts", 0),
        "expert_hidden_size": hf_config.get(
            "moe_intermediate_size", hf_config.get("intermediate_size")
        ),
        "quant_bits": {"fp16": 16, "q8": 8, "q4": 4}[args.quant],
        "group_size": 128,
        "weight_files": {
            "resident": "resident.bin",
            "experts": "experts.bin",
        },
    }

    config_out = output_dir / "config.json"
    with open(config_out, "w") as f:
        json.dump(pocket_config, f, indent=2)
    print(f"Wrote config to {config_out}")


if __name__ == "__main__":
    main()
