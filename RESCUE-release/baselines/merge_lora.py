#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge a PEFT LoRA adapter into Qwen3-8B and save a standalone model."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:
    raise SystemExit(
        "This script requires peft. Install it in the training environment, "
        "for example: pip install peft"
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Qwen3-8B LoRA adapter into the base model."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter-path",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("Merging LoRA adapter into Qwen3-8B")
    print("=" * 100)
    print(f"BASE_MODEL         = {args.base_model}")
    print(f"ADAPTER_PATH       = {args.adapter_path}")
    print(f"OUTPUT_DIR         = {args.output_dir}")
    print(f"TORCH_DTYPE        = {args.torch_dtype}")
    print(f"MAX_SHARD_SIZE     = {args.max_shard_size}")
    print(f"SAFE_SERIALIZATION = {args.safe_serialization}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=resolve_dtype(args.torch_dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        is_trainable=False,
    )
    model.eval()

    print("[Merge] merge_and_unload()")
    merged_model = model.merge_and_unload()
    merged_model.eval()

    print("[Save] writing merged full model")
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)
    print("=" * 100)
    print(f"Finished. Merged model saved to: {args.output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
