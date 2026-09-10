#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-LoRA supervised repair baseline for the Qwen3-8B repair set.

This baseline fits the corrected reasoning targets directly with LoRA adapters
on all transformer projection matrices, then evaluates both repair accuracy and
clean accuracy after every epoch.  The default split is flat:

  train: first 150 rows from gsm8k_qwen_repair.json
  eval:  next 50 rows from gsm8k_qwen_repair.json

The default is a pure repair SFT baseline, with clean protection disabled, so
clean-accuracy degradation is visible.  Add --clean-protection to train a
regularized variant using clean NLL replay.
"""

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:
    raise SystemExit(
        "This script requires peft. Install it in the training environment, "
        "for example: pip install peft"
    ) from exc


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
MLP_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


@dataclass
class RepairItem:
    qid: Any
    question: str
    ground_truth: str
    target_text: str
    type_name: str = ""


@dataclass
class CleanItem:
    qid: Any
    question: str
    target_text: str
    ground_truth: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a full-LoRA mixed repair baseline."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--repair-data-path",
        required=True,
        help="Repair json. Default split uses the first 150 rows for train and next 50 for eval.",
    )
    parser.add_argument(
        "--clean-data-path",
        required=True,
        help="Clean-set json used to monitor clean accuracy.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument("--train-repair-samples", type=int, default=150)
    parser.add_argument("--eval-repair-samples", type=int, default=50)
    parser.add_argument("--train-clean-samples", type=int, default=80)
    parser.add_argument("--eval-clean-samples", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-total-len", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--train-micro-batch-size",
        type=int,
        default=2,
        help="Increase this if memory allows; lower to 1 if Qwen3-8B OOMs.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help=(
            "Comma-separated LoRA module names, or all-linear to target every "
            "nn.Linear leaf except lm_head by default."
        ),
    )
    parser.add_argument(
        "--mlp-only",
        action="store_true",
        help="Shortcut for --target-modules gate_proj,up_proj,down_proj.",
    )
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--clean-protection", action="store_true")
    parser.add_argument("--clean-nll-weight", type=float, default=0.1)
    parser.add_argument("--eval-train-repair", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-epoch-adapters",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="bf16",
    )
    args = parser.parse_args()
    if args.train_micro_batch_size <= 0:
        parser.error("--train-micro-batch-size must be positive.")
    if args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be positive.")
    if args.max_total_len <= 1:
        parser.error("--max-total-len must be > 1.")
    if args.lora_r <= 0:
        parser.error("--lora-r must be positive.")
    if args.clean_nll_weight < 0:
        parser.error("--clean-nll-weight must be non-negative.")
    if args.train_repair_samples <= 0:
        parser.error("--train-repair-samples must be positive.")
    if args.eval_repair_samples <= 0:
        parser.error("--eval-repair-samples must be positive.")
    return args


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_auto(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported json format: {path}")


def safe_get(item: Dict[str, Any], keys: Sequence[str], default=None):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def normalize_text(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_answer_str(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    match = _NUMBER_RE.search(text)
    if not match:
        return text.lower().strip()
    val = match.group(0).replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        number = float(val)
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return ("%.10f" % number).rstrip("0").rstrip(".")
    except Exception:
        return val


def strip_think_blocks(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_generation_for_scoring(text: str) -> str:
    text = strip_think_blocks(text)
    for token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        text = text.replace(token, "")
    cut_position = len(text)
    for marker in [
        "\nQuestion:",
        "\nHuman:",
        "\nComment:",
        "\nInstruction:",
        "\nUser:",
        "\nAssistant:",
        "\n[INST]",
    ]:
        position = text.find(marker)
        if position != -1:
            cut_position = min(cut_position, position)
    return text[:cut_position].strip()


def _last_number(text: str) -> str:
    numbers = _NUMBER_RE.findall(text or "")
    return normalize_answer_str(numbers[-1]) if numbers else ""


def parse_final_answer_strict(text: str) -> str:
    text = clean_generation_for_scoring(text)
    if not text:
        return ""
    patterns = [
        r"Final\s+[Aa]nswer\s*[:：]\s*([^\n\r]+)",
        r"####\s*([^\n\r]+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            answer_line = matches[-1].group(1).strip()
            if "=" in answer_line:
                rhs_number = _last_number(answer_line.split("=")[-1])
                if rhs_number:
                    return rhs_number
            answer_number = _last_number(answer_line)
            return answer_number if answer_number else answer_line
    return ""


def build_chat_prompt(question: str, tokenizer) -> str:
    system_prompt = (
        "Solve the following grade-school math problem.\n"
        "Use concise reasoning. Do not write an introduction. Do not use Markdown headings.\n"
        "Track the changing quantities carefully.\n"
        "Your response must follow this exact format:\n\n"
        "Reasoning:\n"
        "1. <short calculation>\n"
        "2. <short calculation>\n"
        "3. <short calculation>\n"
        "4. <short calculation>\n"
        "Final Answer: <only one number>\n\n"
        "Stop immediately after the Final Answer line."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return (
        "<s>[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n\n"
        f"{question.strip()} [/INST]"
    )


def select_repair_target(row: Dict[str, Any]) -> str:
    target = normalize_text(
        safe_get(
            row,
            ["corrected_reasoning", "corrected_response", "solution", "target"],
        )
    )
    answer = normalize_text(safe_get(row, ["ground_truth", "answer", "label"]))
    if not target:
        return f"Final Answer: {answer}" if answer else ""
    if parse_final_answer_strict(target) == "" and answer:
        target = target.rstrip() + f"\nFinal Answer: {answer}"
    return target


def select_clean_target(row: Dict[str, Any]) -> str:
    response = normalize_text(
        safe_get(
            row,
            [
                "clean_response",
                "model_response",
                "response",
                "generation",
                "solution",
                "corrected_reasoning",
                "reasoning",
                "target",
            ],
        )
    )
    if response:
        return response
    answer = normalize_text(safe_get(row, ["ground_truth", "answer", "label"]))
    return f"Final Answer: {answer}" if answer else ""


def build_repair_items(rows: Sequence[Dict[str, Any]], type_name: str) -> List[RepairItem]:
    items = []
    for index, row in enumerate(rows):
        question = normalize_text(safe_get(row, ["question", "prompt"]))
        ground_truth = normalize_text(safe_get(row, ["ground_truth", "answer", "label"]))
        target = select_repair_target(row)
        if question and ground_truth and target:
            items.append(
                RepairItem(
                    qid=safe_get(row, ["id", "qid"], index),
                    question=question,
                    ground_truth=ground_truth,
                    target_text=target,
                    type_name=type_name,
                )
            )
    return items


def build_clean_items(rows: Sequence[Dict[str, Any]]) -> List[CleanItem]:
    items = []
    for index, row in enumerate(rows):
        question = normalize_text(safe_get(row, ["question", "prompt"]))
        ground_truth = normalize_text(safe_get(row, ["ground_truth", "answer", "label"]))
        target = select_clean_target(row)
        if question and ground_truth and target:
            items.append(
                CleanItem(
                    qid=safe_get(row, ["id", "qid"], index),
                    question=question,
                    target_text=target,
                    ground_truth=ground_truth,
                )
            )
    return items


def build_flat_repair_splits(
    data: Sequence[Dict[str, Any]],
    train_count: int,
    eval_count: int,
) -> Tuple[List[RepairItem], List[RepairItem]]:
    expected = train_count + eval_count
    if len(data) < expected:
        raise ValueError(f"Expected at least {expected} rows, found {len(data)}.")
    train_rows = data[:train_count]
    eval_rows = data[train_count:train_count + eval_count]
    train_items = build_repair_items(train_rows, "repair_train")
    eval_items = build_repair_items(eval_rows, "repair_eval")
    if len(train_items) != train_count:
        raise ValueError(f"Invalid train repair rows: built {len(train_items)}/{train_count}.")
    if len(eval_items) != eval_count:
        raise ValueError(f"Invalid eval repair rows: built {len(eval_items)}/{eval_count}.")
    return train_items, eval_items


def make_repair_batches(
    train_items: Sequence[RepairItem],
    batch_size: int,
) -> List[List[RepairItem]]:
    shuffled = list(train_items)
    random.shuffle(shuffled)
    return [list(batch) for batch in chunks(shuffled, batch_size)]


def chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def get_qwen_eos_ids(tokenizer):
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
    return eos_ids if eos_ids else None


def get_model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def tokenize_supervised_batch(
    tokenizer,
    items: Sequence[Any],
    max_prompt_len: int,
    max_total_len: int,
    device,
) -> Dict[str, torch.Tensor]:
    input_rows, label_rows = [], []
    eos = tokenizer.eos_token or ""
    for item in items:
        prompt = build_chat_prompt(item.question, tokenizer)
        target = normalize_text(item.target_text)
        if eos and not target.endswith(eos):
            target = target + eos
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_len,
        )["input_ids"]
        max_target_len = max(1, max_total_len - len(prompt_ids))
        target_ids = tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_len,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        input_rows.append(input_ids)
        label_rows.append(labels)

    pad_id = tokenizer.pad_token_id
    max_len = max(len(row) for row in input_rows)
    padded_inputs, padded_labels, attention_masks = [], [], []
    for input_ids, labels in zip(input_rows, label_rows):
        pad_len = max_len - len(input_ids)
        padded_inputs.append(input_ids + [pad_id] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
        attention_masks.append([1] * len(input_ids) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
    }


def compute_sft_loss(
    model,
    tokenizer,
    items: Sequence[Any],
    max_prompt_len: int,
    max_total_len: int,
) -> torch.Tensor:
    batch = tokenize_supervised_batch(
        tokenizer=tokenizer,
        items=items,
        max_prompt_len=max_prompt_len,
        max_total_len=max_total_len,
        device=get_model_input_device(model),
    )
    return model(**batch).loss


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    items: Sequence[Any],
    max_prompt_len: int,
    max_new_tokens: int,
    batch_size: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    model.eval()
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rows, correct = [], 0
    for batch_items in chunks(list(items), batch_size):
        prompts = [build_chat_prompt(item.question, tokenizer) for item in batch_items]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=False,
        ).to(get_model_input_device(model))
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=get_qwen_eos_ids(tokenizer),
        )
        prompt_len = encoded["input_ids"].shape[1]
        for index, item in enumerate(batch_items):
            raw_text = tokenizer.decode(
                generated[index][prompt_len:],
                skip_special_tokens=True,
            )
            clean_text = clean_generation_for_scoring(raw_text)
            prediction = normalize_answer_str(parse_final_answer_strict(clean_text))
            target = normalize_answer_str(item.ground_truth)
            is_correct = int(prediction != "" and prediction == target)
            correct += is_correct
            rows.append(
                {
                    "qid": item.qid,
                    "type": getattr(item, "type_name", ""),
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "generation_text": clean_text,
                }
            )
    tokenizer.padding_side = old_padding_side
    return correct / max(len(items), 1), rows


def evaluate_repair_flat(
    model,
    tokenizer,
    items: Sequence[RepairItem],
    max_prompt_len: int,
    max_new_tokens: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    acc, rows = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        items=items,
        max_prompt_len=max_prompt_len,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    no_final = sum(int(row["prediction"] == "") for row in rows)
    return {
        "repair_generation_acc": acc,
        "repair_no_final_answer_rate": no_final / max(len(rows), 1),
        "repair_generation_total": len(items),
    }, rows


def evaluate_clean_accuracy(
    model,
    tokenizer,
    clean_items: Sequence[CleanItem],
    max_prompt_len: int,
    max_new_tokens: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    acc, rows = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        items=clean_items,
        max_prompt_len=max_prompt_len,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    no_final = sum(int(row["prediction"] == "") for row in rows)
    return {
        "clean_eval_acc": acc,
        "clean_eval_total": len(clean_items),
        "clean_no_final_answer_rate": no_final / max(len(rows), 1),
    }, rows


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def discover_all_linear_target_modules(model, include_lm_head: bool) -> List[str]:
    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf == "lm_head" and not include_lm_head:
            continue
        names.add(leaf)
    if not names:
        raise ValueError("No nn.Linear modules found for LoRA targeting.")
    return sorted(names)


def resolve_target_modules(
    model,
    target_modules_arg: str,
    include_lm_head: bool,
    mlp_only: bool = False,
) -> List[str]:
    if mlp_only:
        modules = list(MLP_TARGET_MODULES)
        if include_lm_head:
            modules.append("lm_head")
        return modules
    value = target_modules_arg.strip()
    if value == "all-linear":
        return discover_all_linear_target_modules(model, include_lm_head)
    modules = [part.strip() for part in value.split(",") if part.strip()]
    if include_lm_head and "lm_head" not in modules:
        modules.append("lm_head")
    if not modules:
        raise ValueError("--target-modules resolved to an empty list.")
    return modules


def summarize_trainable_parameters(model) -> Dict[str, Any]:
    trainable, total = 0, 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / max(total, 1),
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_random_seed(args.seed)
    print("=" * 100)
    print("Full-LoRA supervised repair baseline")
    print("=" * 100)
    print(f"MODEL_PATH       = {args.model_path}")
    print(f"REPAIR_DATA_PATH = {args.repair_data_path}")
    print(f"CLEAN_DATA_PATH  = {args.clean_data_path}")
    print(f"OUTPUT_DIR       = {args.output_dir}")
    print(f"REPAIR_SPLIT     = train {args.train_repair_samples}, eval {args.eval_repair_samples}")
    print(f"LORA             = r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"LR               = {args.lr}")
    effective_clean_weight = args.clean_nll_weight if args.clean_protection else 0.0
    print(f"CLEAN_PROTECTION = {args.clean_protection}")
    print(f"CLEAN_NLL_WEIGHT = {effective_clean_weight}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=resolve_torch_dtype(args.torch_dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = resolve_target_modules(
        model,
        args.target_modules,
        args.include_lm_head,
        args.mlp_only,
    )
    print(f"TARGET_MODULES   = {target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    param_summary = summarize_trainable_parameters(model)

    repair_rows = load_json_auto(args.repair_data_path)
    train_repair_items, eval_repair_items = build_flat_repair_splits(
        repair_rows,
        args.train_repair_samples,
        args.eval_repair_samples,
    )
    print(f"[Data] repair train={len(train_repair_items)}, repair eval={len(eval_repair_items)}")

    clean_rows = load_json_auto(args.clean_data_path)
    clean_items = build_clean_items(clean_rows)
    train_clean_items = clean_items[:args.train_clean_samples]
    eval_clean_items = clean_items[
        args.train_clean_samples:args.train_clean_samples + args.eval_clean_samples
    ]
    if effective_clean_weight > 0 and not train_clean_items:
        raise ValueError("--clean-protection is enabled but no train clean items were built.")
    print(f"[Data] clean train={len(train_clean_items)}, clean eval={len(eval_clean_items)}")

    config_row = vars(args).copy()
    config_row.update({
        "target_modules_resolved": target_modules,
        **param_summary,
    })
    write_json(os.path.join(args.output_dir, "run_config.json"), config_row)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    step_log_path = os.path.join(args.output_dir, "train_step_logs.jsonl")
    epoch_log_path = os.path.join(args.output_dir, "epoch_metrics.jsonl")
    best_dir = os.path.join(args.output_dir, "best_adapter")
    best_score = (-math.inf, -math.inf)

    print("\n[Init Evaluation] base model + zero LoRA adapter")
    init_repair_metrics, init_repair_generations = evaluate_repair_flat(
        model=model,
        tokenizer=tokenizer,
        items=eval_repair_items,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
    )
    init_clean_metrics, init_clean_generations = evaluate_clean_accuracy(
        model=model,
        tokenizer=tokenizer,
        clean_items=eval_clean_items,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
    )
    init_row = {
        "epoch": 0,
        **init_repair_metrics,
        **init_clean_metrics,
        "clean_acc_drop_from_init": 0.0,
        "best_saved": True,
    }
    write_json(os.path.join(args.output_dir, "init_eval_metrics.json"), init_row)
    write_json(os.path.join(args.output_dir, "init_repair_generations.json"), init_repair_generations)
    write_json(os.path.join(args.output_dir, "init_clean_generations.json"), init_clean_generations)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    best_score = (
        init_repair_metrics["repair_generation_acc"],
        init_clean_metrics["clean_eval_acc"],
    )
    base_clean_acc = init_clean_metrics["clean_eval_acc"]
    print(
        f"  repair_acc={init_repair_metrics['repair_generation_acc']:.6f}, "
        f"clean_acc={base_clean_acc:.6f}"
    )
    append_jsonl(epoch_log_path, init_row)

    global_step = 0
    clean_cursor = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        repair_batches = make_repair_batches(train_repair_items, args.train_micro_batch_size)
        epoch_losses, epoch_repair_losses, epoch_clean_losses = [], [], []
        print(f"\n[Epoch {epoch}/{args.num_epochs}] repair batches={len(repair_batches)}")

        for step_index, repair_batch in enumerate(repair_batches):
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            repair_loss = compute_sft_loss(
                model=model,
                tokenizer=tokenizer,
                items=repair_batch,
                max_prompt_len=args.max_prompt_len,
                max_total_len=args.max_total_len,
            )
            repair_loss.backward()

            clean_loss_value = 0.0
            if effective_clean_weight > 0:
                clean_batch = []
                for _ in range(len(repair_batch)):
                    clean_batch.append(train_clean_items[clean_cursor % len(train_clean_items)])
                    clean_cursor += 1
                clean_loss = compute_sft_loss(
                    model=model,
                    tokenizer=tokenizer,
                    items=clean_batch,
                    max_prompt_len=args.max_prompt_len,
                    max_total_len=args.max_total_len,
                )
                (effective_clean_weight * clean_loss).backward()
                clean_loss_value = float(clean_loss.detach().item())

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            repair_loss_value = float(repair_loss.detach().item())
            total_loss_value = repair_loss_value + effective_clean_weight * clean_loss_value
            epoch_repair_losses.append(repair_loss_value)
            epoch_clean_losses.append(clean_loss_value)
            epoch_losses.append(total_loss_value)
            step_row = {
                "global_step": global_step,
                "epoch": epoch,
                "repair_batch_index": step_index,
                "repair_loss": repair_loss_value,
                "clean_nll": clean_loss_value,
                "clean_protection": args.clean_protection,
                "clean_nll_weight": effective_clean_weight,
                "total_loss": total_loss_value,
                "num_repair_items": len(repair_batch),
            }
            append_jsonl(step_log_path, step_row)
            if step_index == 0 or (step_index + 1) == len(repair_batches):
                print(
                    f"  step {step_index + 1}/{len(repair_batches)}: "
                    f"repair_loss={repair_loss_value:.6f}, "
                    f"clean_nll={clean_loss_value:.6f}"
                )

        repair_metrics, repair_generations = evaluate_repair_flat(
            model=model,
            tokenizer=tokenizer,
            items=eval_repair_items,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size,
        )
        clean_metrics, clean_generations = evaluate_clean_accuracy(
            model=model,
            tokenizer=tokenizer,
            clean_items=eval_clean_items,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size,
        )
        train_repair_metrics = {}
        if args.eval_train_repair:
            train_metrics, train_generations = evaluate_repair_flat(
                model=model,
                tokenizer=tokenizer,
                items=train_repair_items,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
            )
            train_repair_metrics = {
                f"train_{key}": value for key, value in train_metrics.items()
            }
            write_json(
                os.path.join(args.output_dir, f"epoch_{epoch:02d}_train_repair_generations.json"),
                train_generations,
            )

        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch:02d}_adapter")
        if args.save_epoch_adapters:
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

        current_score = (
            repair_metrics["repair_generation_acc"],
            clean_metrics["clean_eval_acc"],
        )
        best_saved = False
        if current_score > best_score:
            best_score = current_score
            best_saved = True
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

        epoch_row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(epoch_losses)),
            "train_repair_loss": float(np.mean(epoch_repair_losses)),
            "train_clean_nll": float(np.mean(epoch_clean_losses)),
            "clean_protection": args.clean_protection,
            "clean_nll_weight": effective_clean_weight,
            **repair_metrics,
            **clean_metrics,
            **train_repair_metrics,
            "clean_acc_drop_from_init": base_clean_acc - clean_metrics["clean_eval_acc"],
            "best_saved": best_saved,
            "adapter_path": epoch_dir if args.save_epoch_adapters else "",
            "best_adapter_path": best_dir if best_saved else "",
        }
        append_jsonl(epoch_log_path, epoch_row)
        write_json(
            os.path.join(args.output_dir, f"epoch_{epoch:02d}_repair_generations.json"),
            repair_generations,
        )
        write_json(
            os.path.join(args.output_dir, f"epoch_{epoch:02d}_clean_generations.json"),
            clean_generations,
        )
        write_json(os.path.join(args.output_dir, "latest_metrics.json"), epoch_row)
        print(
            f"[Epoch {epoch}] repair_acc={repair_metrics['repair_generation_acc']:.6f}, "
            f"clean_acc={clean_metrics['clean_eval_acc']:.6f}, "
            f"clean_drop={epoch_row['clean_acc_drop_from_init']:.6f}, "
            f"best_saved={best_saved}"
        )

    print("=" * 100)
    print("Finished full-LoRA mixed repair baseline.")
    print(f"Metrics: {epoch_log_path}")
    print(f"Best adapter: {best_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
