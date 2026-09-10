#!/usr/bin/env python3
"""Bake an exact binary MLP-neuron mask into a Hugging Face causal LM."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--binary-mask-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def resolve_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def validate_binary_mask(name, raw, output_rows):
    mask = raw.detach().float().flatten()
    if mask.numel() != output_rows:
        raise ValueError(f"{name}: mask has {mask.numel()} values, expected {output_rows}")
    if not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError(f"{name}: expected an exact binary 0/1 mask")
    return mask


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.binary_mask_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("binary mask checkpoint must be a dictionary")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=resolve_dtype(args.dtype),
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=False, trust_remote_code=True
    )

    report = {"base_model": args.base_model, "mask_path": args.binary_mask_path, "modules": []}
    used_keys = set()
    total = deleted = 0
    for name, module in model.named_modules():
        if not name.endswith(MLP_PROJECTIONS) or name not in state:
            continue
        mask = validate_binary_mask(name, state[name], module.weight.shape[0])
        keep = mask.to(module.weight.dtype)
        module.weight.data.mul_(keep[:, None])
        if module.bias is not None:
            module.bias.data.mul_(keep)
        module_deleted = int((mask == 0).sum().item())
        total += mask.numel()
        deleted += module_deleted
        used_keys.add(name)
        report["modules"].append({"name": name, "neurons": mask.numel(), "deleted": module_deleted})

    if not used_keys:
        raise ValueError("no MLP projection masks matched the model")
    unknown = sorted(set(state) - used_keys)
    report.update({
        "matched_modules": len(used_keys),
        "total_neurons": total,
        "deleted_neurons": deleted,
        "deleted_ratio": deleted / max(total, 1),
        "ignored_mask_keys": unknown,
    })
    (output_dir / "bake_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "modules"}, indent=2))


if __name__ == "__main__":
    main()
