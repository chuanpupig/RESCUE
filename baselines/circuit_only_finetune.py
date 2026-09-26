#!/usr/bin/env python3


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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


TRAIN_REPAIR_SAMPLES = 150
EVAL_REPAIR_SAMPLES = 50
TRAIN_CLEAN_SAMPLES = 100
EVAL_CLEAN_SAMPLES = 100


TRAIN_MICRO_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 16


CLEAN_PROTECTION_DEFAULT = False
NUM_EPOCHS_DEFAULT = 10
LEARNING_RATE_DEFAULT = 2e-4


FINAL_ANSWER_WEIGHT = 3.0


DEFAULT_MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]
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


class RowSubsetLinear(nn.Module):


    def __init__(self, original_layer: nn.Linear, active_indices: torch.Tensor, train_bias: bool = True):
        super().__init__()
        if active_indices.numel() <= 0:
            raise ValueError("RowSubsetLinear requires at least one active row.")
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        device = original_layer.weight.device
        active_indices = active_indices.detach().long().cpu().unique(sorted=True)
        self.register_buffer("active_indices", active_indices.to(device), persistent=True)
        self.delta_weight = nn.Parameter(
            torch.zeros(
                active_indices.numel(),
                original_layer.in_features,
                dtype=torch.float32,
                device=device,
            )
        )
        if train_bias and original_layer.bias is not None:
            self.delta_bias = nn.Parameter(torch.zeros(active_indices.numel(), dtype=torch.float32, device=device))
        else:
            self.delta_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original_layer(x)
        delta = F.linear(
            x.to(self.delta_weight.dtype),
            self.delta_weight,
            self.delta_bias,
        ).to(base.dtype)
        idx = self.active_indices.to(base.device)
        out = base.clone()
        out[..., idx] = out[..., idx] + delta
        return out

    def merge(self) -> nn.Linear:
        idx = self.active_indices.to(self.original_layer.weight.device)
        self.original_layer.weight.data[idx] += self.delta_weight.detach().to(
            self.original_layer.weight.device,
            dtype=self.original_layer.weight.dtype,
        )
        if self.delta_bias is not None and self.original_layer.bias is not None:
            self.original_layer.bias.data[idx] += self.delta_bias.detach().to(
                self.original_layer.bias.device,
                dtype=self.original_layer.bias.dtype,
            )
        return self.original_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune only a located RESCUE circuit.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--mask-path",
        required=True,
    )
    parser.add_argument("--repair-data-path", required=True)
    parser.add_argument("--clean-data-path", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument("--train-repair-samples", type=int, default=TRAIN_REPAIR_SAMPLES)
    parser.add_argument("--eval-repair-samples", type=int, default=EVAL_REPAIR_SAMPLES)
    parser.add_argument("--train-clean-samples", type=int, default=TRAIN_CLEAN_SAMPLES)
    parser.add_argument("--eval-clean-samples", type=int, default=EVAL_CLEAN_SAMPLES)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS_DEFAULT)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE_DEFAULT)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-total-len", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--final-answer-weight", type=float, default=FINAL_ANSWER_WEIGHT)
    parser.add_argument("--train-micro-batch-size", type=int, default=TRAIN_MICRO_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--circuit-from",
        choices=["auto", "active", "closed"],
        default="auto",
        help=(
            "auto treats hard SFT/RL binary masks as 0=located circuit. "
            "active trains values > threshold; closed trains values <= threshold."
        ),
    )
    parser.add_argument(
        "--mask-value-type",
        choices=["auto", "binary", "probability", "logits"],
        default="auto",
    )
    parser.add_argument("--mask-init-value", type=float, default=0.2)
    parser.add_argument("--mask-temperature", type=float, default=1.0)
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=0.96,
        help="RESCUE hard threshold used only when a logits/probability mask has no binary companion.",
    )
    parser.add_argument(
        "--max-delete-ratio",
        type=float,
        default=1.0,
        help="RESCUE deletion budget used only when hardifying logits/probability masks without a binary companion.",
    )
    parser.add_argument("--target-modules", default=",".join(DEFAULT_MLP_TARGETS))
    parser.add_argument("--clean-protection", action=argparse.BooleanOptionalAction, default=CLEAN_PROTECTION_DEFAULT)
    parser.add_argument("--clean-nll-weight", type=float, default=0.1)
    parser.add_argument("--eval-train-repair", action="store_true")
    parser.add_argument(
        "--save-epoch-deltas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save lightweight RowSubsetLinear delta checkpoints after each epoch.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()
    if args.train_micro_batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("batch sizes must be positive.")
    if args.train_repair_samples <= 0 or args.eval_repair_samples <= 0:
        parser.error("repair split sizes must be positive.")
    if args.clean_nll_weight < 0:
        parser.error("--clean-nll-weight must be non-negative.")
    if not os.path.exists(args.mask_path):
        parser.error(f"--mask-path does not exist: {args.mask_path}")
    if not 0.0 < args.hard_threshold < 1.0:
        parser.error("--hard-threshold must be in (0, 1).")
    if not 0.0 <= args.max_delete_ratio <= 1.0:
        parser.error("--max-delete-ratio must be in [0, 1].")
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
    return "" if value is None else str(value).strip()


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
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        "<s>[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n\n"
        f"{question.strip()} [/INST]"
    )


def select_repair_target(row: Dict[str, Any]) -> str:
    target = normalize_text(safe_get(row, ["corrected_reasoning", "corrected_response", "solution", "target"]))
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
            items.append(RepairItem(safe_get(row, ["id", "qid"], index), question, ground_truth, target, type_name))
    return items


def build_clean_items(rows: Sequence[Dict[str, Any]]) -> List[CleanItem]:
    items = []
    for index, row in enumerate(rows):
        question = normalize_text(safe_get(row, ["question", "prompt"]))
        ground_truth = normalize_text(safe_get(row, ["ground_truth", "answer", "label"]))
        target = select_clean_target(row)
        if question and ground_truth and target:
            items.append(CleanItem(safe_get(row, ["id", "qid"], index), question, target, ground_truth))
    return items


def build_flat_repair_splits(
    data: Sequence[Dict[str, Any]],
    train_count: int,
    eval_count: int,
) -> Tuple[List[RepairItem], List[RepairItem]]:
    expected = train_count + eval_count
    if len(data) < expected:
        raise ValueError(f"Expected at least {expected} rows, found {len(data)}.")
    train_items = build_repair_items(data[:train_count], "repair_train")
    eval_items = build_repair_items(data[train_count:train_count + eval_count], "repair_eval")
    if len(train_items) != train_count:
        raise ValueError(f"Invalid train repair rows: built {len(train_items)}/{train_count}.")
    if len(eval_items) != eval_count:
        raise ValueError(f"Invalid eval repair rows: built {len(eval_items)}/{eval_count}.")
    return train_items, eval_items


def chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def make_repair_batches(train_items: Sequence[RepairItem], batch_size: int) -> List[List[RepairItem]]:
    shuffled = list(train_items)
    random.shuffle(shuffled)
    return [list(batch) for batch in chunks(shuffled, batch_size)]


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
    return device if device is not None else next(model.parameters()).device


def tokenize_supervised_batch(
    tokenizer,
    items: Sequence[Any],
    max_prompt_len: int,
    max_total_len: int,
    final_answer_weight: float,
    device,
) -> Dict[str, torch.Tensor]:
    input_rows, label_rows, weight_rows = [], [], []
    eos = tokenizer.eos_token or ""
    for item in items:
        prompt = build_chat_prompt(item.question, tokenizer)
        target = normalize_text(item.target_text)
        if eos and not target.endswith(eos):
            target = target + eos
        target_for_final = normalize_text(item.target_text)
        final_match = re.search(r"Final\s+[Aa]nswer\s*[:：]", target_for_final)
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
        target_weights = [1.0] * len(target_ids)
        if final_match is not None and final_answer_weight > 1.0:
            final_prefix_ids = tokenizer(
                target_for_final[:final_match.start()],
                add_special_tokens=False,
            )["input_ids"]
            final_start = min(len(final_prefix_ids), len(target_weights))
            for index in range(final_start, len(target_weights)):
                target_weights[index] = final_answer_weight
        input_rows.append(prompt_ids + target_ids)
        label_rows.append([-100] * len(prompt_ids) + target_ids)
        weight_rows.append([0.0] * len(prompt_ids) + target_weights)

    pad_id = tokenizer.pad_token_id
    max_len = max(len(row) for row in input_rows)
    padded_inputs, padded_labels, padded_weights, attention_masks = [], [], [], []
    for input_ids, labels, weights in zip(input_rows, label_rows, weight_rows):
        pad_len = max_len - len(input_ids)
        padded_inputs.append(input_ids + [pad_id] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
        padded_weights.append(weights + [0.0] * pad_len)
        attention_masks.append([1] * len(input_ids) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
        "loss_weights": torch.tensor(padded_weights, dtype=torch.float32, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
    }


def weighted_token_nll(logits: torch.Tensor, labels: torch.Tensor, loss_weights: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].to(shift_logits.device, dtype=shift_logits.dtype)
    valid = shift_labels.ne(-100)
    shift_weights = shift_weights * valid
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    return (per_token * shift_weights).sum() / shift_weights.sum().clamp_min(1.0)


def compute_sft_loss(
    model,
    tokenizer,
    items: Sequence[Any],
    max_prompt_len: int,
    max_total_len: int,
    final_answer_weight: float,
) -> torch.Tensor:
    batch = tokenize_supervised_batch(
        tokenizer=tokenizer,
        items=items,
        max_prompt_len=max_prompt_len,
        max_total_len=max_total_len,
        final_answer_weight=final_answer_weight,
        device=get_model_input_device(model),
    )
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    return weighted_token_nll(outputs.logits, batch["labels"], batch["loss_weights"])


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
            raw_text = tokenizer.decode(generated[index][prompt_len:], skip_special_tokens=True)
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


def evaluate_repair_flat(model, tokenizer, items, max_prompt_len, max_new_tokens, batch_size):
    acc, rows = evaluate_generation(model, tokenizer, items, max_prompt_len, max_new_tokens, batch_size)
    no_final = sum(int(row["prediction"] == "") for row in rows)
    return {
        "repair_generation_acc": acc,
        "repair_no_final_answer_rate": no_final / max(len(rows), 1),
        "repair_generation_total": len(items),
    }, rows


def evaluate_clean_accuracy(model, tokenizer, clean_items, max_prompt_len, max_new_tokens, batch_size):
    acc, rows = evaluate_generation(model, tokenizer, clean_items, max_prompt_len, max_new_tokens, batch_size)
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


def effective_soft_mask(logits: torch.Tensor, init_value: float, temperature: float) -> torch.Tensor:
    delta = (init_value - logits.float()) / max(temperature, 1e-6)
    close_raw = 2.0 * (torch.sigmoid(delta) - 0.5)
    close = close_raw.clamp(min=0.0, max=1.0)
    return 1.0 - close


def hard_mask_from_soft(mask_soft: torch.Tensor, hard_threshold: float, max_delete_ratio: float) -> torch.Tensor:

    candidate_close = mask_soft <= hard_threshold
    max_delete = int(mask_soft.numel() * max(0.0, min(1.0, max_delete_ratio)))
    if max_delete <= 0:
        return torch.ones_like(mask_soft)
    candidate_count = int(candidate_close.sum().item())
    if candidate_count <= max_delete:
        return (~candidate_close).to(mask_soft.dtype)
    flat = mask_soft.flatten()
    _, idx = torch.topk(-flat, k=max_delete)
    close_flat = torch.zeros_like(flat, dtype=torch.bool)
    close_flat[idx] = True
    close_flat = close_flat & candidate_close.flatten()
    return (~close_flat).view_as(mask_soft).to(mask_soft.dtype)


def infer_mask_value_type(path: str, tensor: torch.Tensor, configured: str) -> str:
    if configured != "auto":
        return configured
    lower_path = path.lower()
    x = tensor.detach().float()
    if "binary" in lower_path or tensor.dtype in {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        return "binary"
    if "probability" in lower_path or "prob_masks" in lower_path:
        return "probability"
    if x.numel() > 0:
        xmin = float(x.min().item())
        xmax = float(x.max().item())
        if xmin >= 0.0 and xmax <= 1.0:
            is_binary = bool(((x == 0.0) | (x == 1.0)).all().item())
            return "binary" if is_binary else "probability"
    return "logits"


def should_train_closed_circuit(path: str, args: argparse.Namespace, value_type: str) -> bool:
    if args.circuit_from == "closed":
        return True
    if args.circuit_from == "active":
        return False
    lower_path = path.lower()
    if value_type == "binary" and (
        "hard" in lower_path
        or "binary_masks" in lower_path
        or "sft" in lower_path
        or "rl" in lower_path
    ):
        return True
    if value_type == "logits" and "hard" in lower_path:
        return True
    return False


def tensor_to_active_indices(
    raw: torch.Tensor,
    path: str,
    args: argparse.Namespace,
    expected_rows: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    value_type = infer_mask_value_type(path, raw, args.mask_value_type)
    if value_type == "binary":
        keep = (raw.float() > 0.5).float()
    elif value_type == "probability":
        keep = raw.float().clamp(0.0, 1.0)
    elif value_type == "logits":
        keep = effective_soft_mask(raw, args.mask_init_value, args.mask_temperature)
    else:
        raise ValueError(f"Unsupported mask value type: {value_type}")
    keep = keep.flatten()
    if keep.numel() != expected_rows:
        raise ValueError(f"mask has {keep.numel()} values, expected {expected_rows}.")

    train_closed = should_train_closed_circuit(path, args, value_type)
    if value_type == "binary":
        hard_keep = keep
    else:
        hard_keep = hard_mask_from_soft(keep, args.hard_threshold, args.max_delete_ratio)

    if train_closed:
        active = hard_keep <= args.mask_threshold
    else:
        active = hard_keep > args.mask_threshold
    indices = active.nonzero(as_tuple=False).flatten().long()
    report = {
        "value_type": value_type,
        "train_closed_circuit": bool(train_closed),
        "mask_values": int(keep.numel()),
        "active_rows": int(indices.numel()),
        "active_ratio": float(indices.numel() / max(keep.numel(), 1)),
        "hard_threshold": float(args.hard_threshold),
        "max_delete_ratio": float(args.max_delete_ratio),
        "hard_keep_rows": int((hard_keep > args.mask_threshold).sum().item()),
        "hard_closed_rows": int((hard_keep <= args.mask_threshold).sum().item()),
        "keep_min": float(keep.min().item()) if keep.numel() else 0.0,
        "keep_max": float(keep.max().item()) if keep.numel() else 0.0,
        "keep_mean": float(keep.mean().item()) if keep.numel() else 0.0,
    }
    return indices, report


def get_parent_and_attr(model: nn.Module, module_name: str):
    parent_name, attr_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr_name


def wrap_linear_rows(model: nn.Module, module_name: str, indices: torch.Tensor) -> Dict[str, Any]:
    parent, attr_name = get_parent_and_attr(model, module_name)
    module = getattr(parent, attr_name)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"{module_name} is not nn.Linear: {type(module)}")
    wrapped = RowSubsetLinear(module, indices)
    setattr(parent, attr_name, wrapped)
    return {
        "module": module_name,
        "kind": "linear_rows",
        "active_rows": int(indices.numel()),
        "out_features": int(module.out_features),
        "in_features": int(module.in_features),
        "trainable_params": int(indices.numel() * module.in_features + (indices.numel() if module.bias is not None else 0)),
    }


def apply_pt_mask(model: nn.Module, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[nn.Parameter]]:
    state = torch.load(args.mask_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected mask state dict in {args.mask_path}, got {type(state)}.")

    target_suffixes = tuple(part.strip() for part in args.target_modules.split(",") if part.strip())
    wrapped_reports: List[Dict[str, Any]] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not name.endswith(target_suffixes):
            continue
        if name not in state:
            continue
        indices, mask_report = tensor_to_active_indices(state[name], args.mask_path, args, module.out_features)
        if indices.numel() == 0:
            print(f"[Mask] skip {name}: no active rows after threshold.")
            continue
        report = wrap_linear_rows(model, name, indices)
        report.update(mask_report)
        wrapped_reports.append(report)
        print(f"[SaCirT] wrapped {name}: active_rows={indices.numel()}/{module.out_features}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    return wrapped_reports, trainable


def apply_circuit_tuners(model: nn.Module, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[nn.Parameter]]:
    for param in model.parameters():
        param.requires_grad = False
    return apply_pt_mask(model, args)


def merge_row_subset_modules(model: nn.Module) -> int:
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, RowSubsetLinear):
            continue
        parent, attr_name = get_parent_and_attr(model, name)
        setattr(parent, attr_name, module.merge())
        merged += 1
    return merged


def summarize_trainable_parameters(trainable_params: Sequence[nn.Parameter], model: nn.Module) -> Dict[str, Any]:
    trainable = sum(p.numel() for p in trainable_params)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_ratio": float(trainable / max(total, 1)),
    }


def capture_delta_state(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, RowSubsetLinear):
            row = {
                "active_indices": module.active_indices.detach().cpu(),
                "delta_weight": module.delta_weight.detach().cpu().clone(),
            }
            if module.delta_bias is not None:
                row["delta_bias"] = module.delta_bias.detach().cpu().clone()
            state[name] = row
    return state


def restore_delta_state(model: nn.Module, state: Dict[str, Dict[str, torch.Tensor]]):
    restored = 0
    for name, module in model.named_modules():
        if not isinstance(module, RowSubsetLinear) or name not in state:
            continue
        row = state[name]
        module.delta_weight.data.copy_(row["delta_weight"].to(module.delta_weight.device))
        if module.delta_bias is not None and "delta_bias" in row:
            module.delta_bias.data.copy_(row["delta_bias"].to(module.delta_bias.device))
        restored += 1
    print(f"[Best] restored delta state for {restored} modules")


def save_merged_full_model(model, tokenizer, output_dir: str):
    ensure_dir(output_dir)
    merged = merge_row_subset_modules(model)
    print(f"[Save] merged {merged} RowSubsetLinear modules into full model: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_random_seed(args.seed)

    print("=" * 100)
    print("SaCirT circuit-only supervised repair")
    print("=" * 100)
    print(f"MODEL_PATH       = {args.model_path}")
    print(f"MASK_PATH        = {args.mask_path}")
    print(f"OUTPUT_DIR       = {args.output_dir}")
    print(f"REPAIR_DATA_PATH = {args.repair_data_path}")
    print(f"CLEAN_DATA_PATH  = {args.clean_data_path}")
    print(f"REPAIR_SPLIT     = train {args.train_repair_samples}, eval {args.eval_repair_samples}")
    print(f"SaCirT           = epochs={args.num_epochs}, lr={args.lr}, circuit_from={args.circuit_from}")
    print(f"FINAL_ANS_WEIGHT = {args.final_answer_weight}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    model.config.use_cache = False

    wrapped_reports, trainable_params = apply_circuit_tuners(model, args)
    if not trainable_params:
        raise RuntimeError("No trainable circuit parameters were created. Check mask path, mask polarity, and target modules.")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    param_summary = summarize_trainable_parameters(trainable_params, model)
    print(
        f"[Trainable] {param_summary['trainable_params']:,}/"
        f"{param_summary['total_params']:,} ({param_summary['trainable_ratio'] * 100:.6f}%)"
    )
    write_json(
        os.path.join(args.output_dir, "run_config.json"),
        {
            **vars(args),
            **param_summary,
            "wrapped_modules": wrapped_reports,
        },
    )

    repair_rows = load_json_auto(args.repair_data_path)
    train_repair_items, eval_repair_items = build_flat_repair_splits(
        repair_rows,
        args.train_repair_samples,
        args.eval_repair_samples,
    )
    clean_rows = load_json_auto(args.clean_data_path)
    clean_items = build_clean_items(clean_rows)
    train_clean_items = clean_items[:args.train_clean_samples]
    eval_clean_items = clean_items[args.train_clean_samples:args.train_clean_samples + args.eval_clean_samples]
    effective_clean_weight = args.clean_nll_weight if args.clean_protection else 0.0
    if effective_clean_weight > 0 and not train_clean_items:
        raise ValueError("--clean-protection is enabled but no clean training items were built.")
    print(f"[Data] repair train={len(train_repair_items)}, repair eval={len(eval_repair_items)}")
    print(f"[Data] clean train={len(train_clean_items)}, clean eval={len(eval_clean_items)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    step_log_path = os.path.join(args.output_dir, "train_step_logs.jsonl")
    epoch_log_path = os.path.join(args.output_dir, "epoch_metrics.jsonl")
    best_score = (-math.inf, -math.inf)
    best_delta_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    best_epoch = -1

    print("\n[Init Evaluation] base model + zero circuit deltas")
    init_repair_metrics, init_repair_generations = evaluate_repair_flat(
        model, tokenizer, eval_repair_items, args.max_prompt_len, args.max_new_tokens, args.eval_batch_size
    )
    init_clean_metrics, init_clean_generations = evaluate_clean_accuracy(
        model, tokenizer, eval_clean_items, args.max_prompt_len, args.max_new_tokens, args.eval_batch_size
    )
    init_row = {
        "epoch": 0,
        **init_repair_metrics,
        **init_clean_metrics,
        "clean_acc_drop_from_init": 0.0,
        "best_saved": False,
    }
    write_json(os.path.join(args.output_dir, "init_eval_metrics.json"), init_row)
    write_json(os.path.join(args.output_dir, "init_repair_generations.json"), init_repair_generations)
    write_json(os.path.join(args.output_dir, "init_clean_generations.json"), init_clean_generations)
    append_jsonl(epoch_log_path, init_row)
    base_clean_acc = init_clean_metrics["clean_eval_acc"]
    print(f"  repair_acc={init_repair_metrics['repair_generation_acc']:.6f}, clean_acc={base_clean_acc:.6f}")

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
                model,
                tokenizer,
                repair_batch,
                args.max_prompt_len,
                args.max_total_len,
                args.final_answer_weight,
            )
            repair_loss.backward()

            clean_loss_value = 0.0
            if effective_clean_weight > 0:
                clean_batch = []
                for _ in range(len(repair_batch)):
                    clean_batch.append(train_clean_items[clean_cursor % len(train_clean_items)])
                    clean_cursor += 1
                clean_loss = compute_sft_loss(
                    model,
                    tokenizer,
                    clean_batch,
                    args.max_prompt_len,
                    args.max_total_len,
                    args.final_answer_weight,
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
            append_jsonl(
                step_log_path,
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "repair_batch_index": step_index,
                    "repair_loss": repair_loss_value,
                    "clean_nll": clean_loss_value,
                    "clean_protection": args.clean_protection,
                    "clean_nll_weight": effective_clean_weight,
                    "total_loss": total_loss_value,
                    "num_repair_items": len(repair_batch),
                },
            )
            if step_index == 0 or (step_index + 1) == len(repair_batches):
                print(
                    f"  step {step_index + 1}/{len(repair_batches)}: "
                    f"repair_loss={repair_loss_value:.6f}, clean_nll={clean_loss_value:.6f}"
                )

        repair_metrics, repair_generations = evaluate_repair_flat(
            model, tokenizer, eval_repair_items, args.max_prompt_len, args.max_new_tokens, args.eval_batch_size
        )
        clean_metrics, clean_generations = evaluate_clean_accuracy(
            model, tokenizer, eval_clean_items, args.max_prompt_len, args.max_new_tokens, args.eval_batch_size
        )
        train_repair_metrics = {}
        if args.eval_train_repair:
            train_metrics, train_generations = evaluate_repair_flat(
                model, tokenizer, train_repair_items, args.max_prompt_len, args.max_new_tokens, args.eval_batch_size
            )
            train_repair_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
            write_json(os.path.join(args.output_dir, f"epoch_{epoch:02d}_train_repair_generations.json"), train_generations)

        if args.save_epoch_deltas:
            torch.save(capture_delta_state(model), os.path.join(args.output_dir, f"epoch_{epoch:02d}_deltas.pt"))

        current_score = (repair_metrics["repair_generation_acc"], clean_metrics["clean_eval_acc"])
        best_saved = False
        if current_score > best_score:
            best_score = current_score
            best_saved = True
            best_epoch = epoch
            best_delta_state = capture_delta_state(model)
            torch.save(best_delta_state, os.path.join(args.output_dir, "best_deltas.pt"))
            write_json(
                os.path.join(args.output_dir, "best_delta_metrics.json"),
                {
                    "epoch": epoch,
                    "repair_generation_acc": current_score[0],
                    "clean_eval_acc": current_score[1],
                },
            )

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
            "best_epoch": best_epoch,
        }
        append_jsonl(epoch_log_path, epoch_row)
        write_json(os.path.join(args.output_dir, f"epoch_{epoch:02d}_repair_generations.json"), repair_generations)
        write_json(os.path.join(args.output_dir, f"epoch_{epoch:02d}_clean_generations.json"), clean_generations)
        write_json(os.path.join(args.output_dir, "latest_metrics.json"), epoch_row)
        print(
            f"[Epoch {epoch}] repair_acc={repair_metrics['repair_generation_acc']:.6f}, "
            f"clean_acc={clean_metrics['clean_eval_acc']:.6f}, "
            f"clean_drop={epoch_row['clean_acc_drop_from_init']:.6f}, "
            f"best_seen={best_saved}"
        )

    final_dir = os.path.join(args.output_dir, "final_model")
    if best_delta_state is not None:
        restore_delta_state(model, best_delta_state)
    save_merged_full_model(model, tokenizer, final_dir)
    write_json(
        os.path.join(args.output_dir, "final_summary.json"),
        {
            "final_model": final_dir,
            "best_epoch": best_epoch,
            "best_score_seen": {
                "repair_generation_acc": best_score[0],
                "clean_eval_acc": best_score[1],
            },
            **param_summary,
        },
    )
    print("=" * 100)
    print("Finished SaCirT circuit-only repair.")
    print(f"Metrics    : {epoch_log_path}")
    print(f"Final model: {final_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
