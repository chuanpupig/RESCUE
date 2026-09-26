import argparse
import os
import csv
import json
import random
import re
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json_auto(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported json format: {path}")


def safe_get(item: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in item and item[k] is not None:
            return item[k]
    return default


def normalize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def normalize_numeric_string(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    m = _NUMBER_RE.search(s)
    if not m:
        return s.lower().strip()

    val = m.group(0).replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        f = float(val)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return ("%.10f" % f).rstrip("0").rstrip(".")
    except Exception:
        return val


def _last_number(text: str) -> str:
    nums = _NUMBER_RE.findall(text or "")
    return normalize_numeric_string(nums[-1]) if nums else ""


def parse_final_answer_strict(text: str) -> str:

    text = (text or "").strip()
    if not text:
        return ""

    matches = list(re.finditer(r"Final\s+[Aa]nswer\s*[:：]\s*([^\n\r]+)", text, flags=re.IGNORECASE))
    if matches:
        ans_line = matches[-1].group(1).strip()
        ans_num = _last_number(ans_line)
        return ans_num if ans_num else ans_line

    matches = list(re.finditer(r"####\s*([^\n\r]+)", text))
    if matches:
        ans_line = matches[-1].group(1).strip()
        ans_num = _last_number(ans_line)
        return ans_num if ans_num else ans_line

    return ""


def clean_generation_for_scoring(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"</?think>", "", s, flags=re.IGNORECASE)
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        s = s.replace(tok, "")


    stop_markers = [
        "\nQuestion:", "\nHuman:", "\nComment:",
        "\nInstruction:", "\nUser:", "\nAssistant:", "\n[INST]",
    ]
    cut_pos = len(s)
    for marker in stop_markers:
        pos = s.find(marker)
        if pos != -1:
            cut_pos = min(cut_pos, pos)
    return s[:cut_pos].strip()


def get_qwen_eos_ids(tokenizer):
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
    return eos_ids if eos_ids else None


def disable_generation_max_length_warning(model):

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and hasattr(gen_cfg, "max_length"):
        gen_cfg.max_length = None
    return model


class AblatedLinear(nn.Module):
    enabled = True
    mask_temperature = 1.0
    mask_init_value = 0.2
    sensitivity_probe_offset = 0.0

    def __init__(self, original_layer: nn.Linear, init_value: float = 0.2):
        super().__init__()
        self.original_layer = original_layer
        for p in self.original_layer.parameters():
            p.requires_grad = False

        out_features = original_layer.out_features
        device = original_layer.weight.device
        self.init_value = init_value


        self.logits = nn.Parameter(
            torch.ones(out_features, device=device, dtype=torch.float32) * init_value
        )

    def get_mask(self, hard: Optional[bool] = None):
        surrogate = effective_mask_prob(self.logits)
        mask_hard = (self.logits >= self.init_value).to(surrogate.dtype)
        mask = mask_hard - surrogate.detach() + surrogate
        return mask.to(self.original_layer.weight.dtype)

    def forward(self, x):
        out = self.original_layer(x)
        if not AblatedLinear.enabled:
            return out
        mask = self.get_mask()
        return out * mask

    @staticmethod
    def save_masks(model):
        state = {}
        for name, module in model.named_modules():
            if isinstance(module, AblatedLinear):
                state[name] = module.logits.detach().cpu()
        return state

    @staticmethod
    def load_masks(model, state):
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, AblatedLinear) and name in state:
                module.logits.data.copy_(state[name].to(module.logits.device))
                count += 1
        return count


def save_all_masks(model, path: str):

    torch.save(AblatedLinear.save_masks(model), path)
    print(f"[Save] neuron masks -> {path}")


def configure_ste(temperature: float = 1.0, init_value: float = 0.2):

    if temperature <= 0:
        raise ValueError("mask temperature must be > 0")
    AblatedLinear.mask_temperature = temperature
    AblatedLinear.mask_init_value = init_value


def set_mask_probe_offset(offset: float):
    AblatedLinear.sensitivity_probe_offset = offset


def effective_mask_prob(logits: torch.Tensor) -> torch.Tensor:

    delta = (
        AblatedLinear.mask_init_value
        - logits
        + AblatedLinear.sensitivity_probe_offset
    ) / max(AblatedLinear.mask_temperature, 1e-6)
    close_raw = 2.0 * (torch.sigmoid(delta) - 0.5)
    close_forward = close_raw.clamp(min=0.0, max=1.0)
    close = close_forward.detach() - close_raw.detach() + close_raw
    return 1.0 - close


def patch_model(model):

    mlp_targets = ("gate_proj", "up_proj", "down_proj")
    mask_params = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith(mlp_targets):
            parent_name, attr_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            wrapped = AblatedLinear(module, init_value=AblatedLinear.mask_init_value)
            setattr(parent, attr_name, wrapped)
            mask_params.append(wrapped.logits)
    print(f"[Patch] patched {len(mask_params)} neuron-mask projections")
    return mask_params


def build_prompt(question: str, tokenizer) -> str:

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

def build_target(corrected_reasoning: str, ground_truth: Optional[str] = None) -> str:
    text = corrected_reasoning.strip()
    text = re.sub(r"^\s*Reasoning\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n", text)
    target = "Reasoning: " + text if not text.startswith("Reasoning:") else text


    gt = normalize_text(ground_truth)
    final = parse_final_answer_strict(target) or gt
    target = re.sub(r"\n?\s*Final\s+answer\s*[:：].*$", "", target, flags=re.IGNORECASE | re.DOTALL).rstrip()
    if final:
        target = target.rstrip() + f"\nFinal Answer: {final}"
    return target


class RESCUERepairDataset(Dataset):
    def __init__(
        self,
        data_list,
        tokenizer,
        max_length: int = 512,
        final_answer_weight: float = 6.0,
        target_head_tokens_when_truncated: int = 128,
        max_reasoning_tokens_before_final: int = 160,
    ):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.final_answer_weight = final_answer_weight
        self.target_head_tokens_when_truncated = target_head_tokens_when_truncated
        self.max_reasoning_tokens_before_final = max_reasoning_tokens_before_final
        self.samples = []
        self.num_target_truncated = 0
        self.num_final_missing_after_encode = 0

        for i, item in enumerate(data_list):
            question = normalize_text(safe_get(item, ["question", "prompt"]))
            corrected_reasoning = normalize_text(safe_get(item, ["corrected_reasoning"]))
            ground_truth = normalize_text(safe_get(item, ["ground_truth", "answer", "label"]))

            if not question:
                raise ValueError(f"sample idx={i} missing question/prompt")
            if not corrected_reasoning:
                raise ValueError(f"sample idx={i} missing corrected_reasoning")

            prompt = build_prompt(question, self.tokenizer)
            target = build_target(corrected_reasoning, ground_truth)

            self.samples.append(self._encode(prompt, target))

        print(
            "[RESCUERepairDataset] "
            f"samples={len(self.samples)}, "
            f"target_truncated={self.num_target_truncated}, "
            f"final_missing_after_encode={self.num_final_missing_after_encode}"
        )

    def _encode(self, prompt: str, target: str):
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_text = target + self.tokenizer.eos_token
        target_ids_full = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        final_match = re.search(r"Final\s+answer\s*[:：]", target, flags=re.IGNORECASE)
        final_prefix_ids = self.tokenizer(target[:final_match.start()], add_special_tokens=False)["input_ids"] if final_match else []

        if final_match and len(final_prefix_ids) > self.max_reasoning_tokens_before_final:
            final_and_after = target[final_match.start():] + self.tokenizer.eos_token
            final_ids = self.tokenizer(final_and_after, add_special_tokens=False)["input_ids"]
            target_ids_full = final_prefix_ids[:self.max_reasoning_tokens_before_final] + final_ids
            final_prefix_ids = final_prefix_ids[:self.max_reasoning_tokens_before_final]

        max_target_len = max(self.max_length - len(prompt_ids), 1)
        target_ids = target_ids_full
        final_start_in_target = len(final_prefix_ids) if final_match else None

        if len(target_ids_full) > max_target_len:
            self.num_target_truncated += 1
            head_len = min(self.target_head_tokens_when_truncated, max_target_len // 2)
            tail_len = max_target_len - head_len
            target_ids = target_ids_full[:head_len] + target_ids_full[-tail_len:]
            if final_match:
                tail_start = len(target_ids_full) - tail_len
                final_start_in_target = head_len + (len(final_prefix_ids) - tail_start) if len(final_prefix_ids) >= tail_start else None

        input_ids_list = prompt_ids + target_ids
        attention_mask_list = [1] * len(input_ids_list)
        labels_list = [-100] * len(prompt_ids) + target_ids.copy()
        loss_weights_list = [0.0] * len(prompt_ids) + [1.0] * len(target_ids)

        if final_start_in_target is not None and 0 <= final_start_in_target < len(target_ids):
            final_start = len(prompt_ids) + final_start_in_target
            for idx in range(final_start, len(loss_weights_list)):
                loss_weights_list[idx] = self.final_answer_weight
        else:
            self.num_final_missing_after_encode += 1

        pad_len = self.max_length - len(input_ids_list)
        if pad_len > 0:
            input_ids_list += [self.tokenizer.pad_token_id] * pad_len
            attention_mask_list += [0] * pad_len
            labels_list += [-100] * pad_len
            loss_weights_list += [0.0] * pad_len

        input_ids = torch.tensor(input_ids_list[:self.max_length], dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list[:self.max_length], dtype=torch.long)
        labels = torch.tensor(labels_list[:self.max_length], dtype=torch.long)
        loss_weights = torch.tensor(loss_weights_list[:self.max_length], dtype=torch.float32)

        prompt_len = min(len(prompt_ids), input_ids.shape[0])
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        loss_weights[labels == -100] = 0.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def select_clean_target(item: Dict[str, Any]) -> str:

    response = normalize_text(safe_get(item, [
        "clean_response",
        "model_response",
        "response",
        "generation",
        "solution",
        "corrected_reasoning",
        "reasoning",
        "target",
    ]))
    if response:
        if response.lower().startswith("reasoning:"):
            return response
        return build_target(response)

    answer = normalize_text(safe_get(item, ["ground_truth", "answer", "label"]))
    if answer:
        return f"Final answer: {answer}"
    return ""


class CleanPreserveDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length: int = 512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for i, item in enumerate(data_list):
            question = normalize_text(safe_get(item, ["question", "prompt"]))
            target = select_clean_target(item)
            if not question or not target:
                print(f"[CleanPreserveDataset] skip idx={i}: missing question or target")
                continue
            prompt = build_prompt(question, self.tokenizer)
            self.samples.append(self._encode(prompt, target))

    def _encode(self, prompt: str, target: str):
        full_text = prompt + target + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), input_ids.shape[0])
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SimpleCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }
        if "loss_weights" in features[0]:
            batch["loss_weights"] = torch.stack([f["loss_weights"] for f in features])
        return batch


def token_nll_loss(logits: torch.Tensor, labels: torch.Tensor, loss_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    if loss_weights is None:
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        return loss

    per_token = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    shift_weights = loss_weights[:, 1:].to(per_token.device, dtype=per_token.dtype)
    valid = shift_labels.ne(-100)
    shift_weights = shift_weights * valid
    return (per_token * shift_weights).sum() / shift_weights.sum().clamp_min(1.0)


def token_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:

    shift_student = student_logits[:, :-1, :].float().contiguous() / temperature
    shift_teacher = teacher_logits[:, :-1, :].float().contiguous() / temperature
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)

    student_log_probs = F.log_softmax(shift_student, dim=-1)
    teacher_probs = F.softmax(shift_teacher, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    kl = kl * valid
    denom = valid.sum().clamp_min(1)
    return kl.sum() / denom * (temperature ** 2)


class RESCUERepairTrainer(Trainer):
    def __init__(
        self,
        *args,
        sparsity_weight_mlp: float = 0.01,
        mlp_mask_params=None,
        best_mask_path: Optional[str] = None,
        tokenizer_for_generation=None,
        eval_generation_samples: Optional[List[Dict[str, Any]]] = None,
        clean_generation_samples: Optional[List[Dict[str, Any]]] = None,
        generation_output_dir: Optional[str] = None,
        generation_max_new_tokens: int = 256,
        generation_batch_size: int = 4,
        clean_dataset: Optional[Dataset] = None,
        clean_batch_size: int = 4,
        clean_kl_weight: float = 0.0,
        clean_nll_weight: float = 0.0,
        clean_kl_temperature: float = 1.0,
        use_sensitivity_regularizer: bool = True,
        sensitivity_probe_offset: float = 0.10,
        sensitivity_probe_every: int = 4,
        sensitivity_ema_beta: float = 0.95,
        repair_close_weight: float = 0.05,
        clean_open_weight: float = 0.20,
        closure_l2_weight: float = 0.02,
        max_soft_closure: float = 0.35,
        max_closure_weight: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sparsity_weight_mlp = sparsity_weight_mlp
        self.mlp_mask_params = mlp_mask_params or []
        self.best_mask_path = best_mask_path
        self.best_eval_loss = None
        self.best_generation_acc = -1.0
        self.tokenizer_for_generation = tokenizer_for_generation
        self.eval_generation_samples = eval_generation_samples or []
        self.clean_generation_samples = clean_generation_samples or []
        self.generation_output_dir = generation_output_dir
        self.generation_max_new_tokens = generation_max_new_tokens
        self.generation_batch_size = generation_batch_size
        self.clean_kl_weight = clean_kl_weight
        self.clean_nll_weight = clean_nll_weight
        self.clean_kl_temperature = clean_kl_temperature
        self.mask_params = self.mlp_mask_params
        self.use_sensitivity_regularizer = use_sensitivity_regularizer
        self.sensitivity_probe_offset = sensitivity_probe_offset
        self.sensitivity_probe_every = max(1, sensitivity_probe_every)
        self.sensitivity_ema_beta = sensitivity_ema_beta
        self.repair_close_weight = repair_close_weight
        self.clean_open_weight = clean_open_weight
        self.closure_l2_weight = closure_l2_weight
        self.max_soft_closure = max_soft_closure
        self.max_closure_weight = max_closure_weight
        self.repair_close_sens_ema = [torch.zeros_like(p, dtype=torch.float32) for p in self.mask_params]
        self.clean_open_sens_ema = [torch.zeros_like(p, dtype=torch.float32) for p in self.mask_params]
        self._sensitivity_step = 0
        self.clean_dataloader = None
        self.clean_iter = None
        if clean_dataset is not None and len(clean_dataset) > 0 and (clean_kl_weight > 0 or clean_nll_weight > 0):
            self.clean_dataloader = DataLoader(
                clean_dataset,
                batch_size=clean_batch_size,
                shuffle=True,
                collate_fn=SimpleCollator(),
                drop_last=False,
            )
        self._last_loss_stats = {}
        self.eval_history = []

    def _next_clean_batch(self):
        if self.clean_dataloader is None:
            return None
        if self.clean_iter is None:
            self.clean_iter = iter(self.clean_dataloader)
        try:
            return next(self.clean_iter)
        except StopIteration:
            self.clean_iter = iter(self.clean_dataloader)
            return next(self.clean_iter)

    def _update_sensitivity_ema(self, grads, ema_refs, direction: str):
        for i, (grad, param) in enumerate(zip(grads, self.mask_params)):
            if grad is None:
                signal = torch.zeros_like(param, dtype=torch.float32)
            elif direction == "close":
                signal = F.relu(grad.detach().float())
            elif direction == "open":
                signal = F.relu(-grad.detach().float())
            else:
                raise ValueError(f"Unsupported sensitivity direction: {direction}")
            ema_refs[i].mul_(self.sensitivity_ema_beta).add_(signal.to(ema_refs[i].device), alpha=1.0 - self.sensitivity_ema_beta)

    def _compute_probe_sensitivities(self, model, repair_inputs, clean_batch):
        if not torch.is_grad_enabled() or not self.use_sensitivity_regularizer or not self.mask_params:
            return
        self._sensitivity_step += 1
        if self._sensitivity_step % self.sensitivity_probe_every != 0:
            return

        set_mask_probe_offset(self.sensitivity_probe_offset)
        try:
            AblatedLinear.enabled = True
            repair_outputs = model(
                input_ids=repair_inputs["input_ids"],
                attention_mask=repair_inputs["attention_mask"],
            )
            repair_probe_loss = token_nll_loss(
                repair_outputs.logits,
                repair_inputs["labels"],
                repair_inputs.get("loss_weights"),
            )
            repair_grads = torch.autograd.grad(
                repair_probe_loss,
                self.mask_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            self._update_sensitivity_ema(repair_grads, self.repair_close_sens_ema, direction="close")

            if clean_batch is not None:
                AblatedLinear.enabled = True
                clean_outputs = model(
                    input_ids=clean_batch["input_ids"],
                    attention_mask=clean_batch["attention_mask"],
                )
                if self.clean_kl_weight > 0:
                    AblatedLinear.enabled = False
                    with torch.no_grad():
                        teacher_outputs = model(
                            input_ids=clean_batch["input_ids"],
                            attention_mask=clean_batch["attention_mask"],
                        )
                    AblatedLinear.enabled = True
                    clean_probe_loss = token_kl_loss(
                        student_logits=clean_outputs.logits,
                        teacher_logits=teacher_outputs.logits,
                        labels=clean_batch["labels"],
                        temperature=self.clean_kl_temperature,
                    )
                else:
                    clean_probe_loss = token_nll_loss(clean_outputs.logits, clean_batch["labels"])
                clean_grads = torch.autograd.grad(
                    clean_probe_loss,
                    self.mask_params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                self._update_sensitivity_ema(clean_grads, self.clean_open_sens_ema, direction="open")
        finally:
            set_mask_probe_offset(0.0)
            configure_ste(AblatedLinear.mask_temperature, AblatedLinear.mask_init_value)
            AblatedLinear.enabled = True

    def _compute_sensitivity_regularizer(self):
        if not self.use_sensitivity_regularizer or not self.mask_params:
            device = self.model.device
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=self.model.device)
        for param, repair_sens, clean_sens in zip(self.mask_params, self.repair_close_sens_ema, self.clean_open_sens_ema):
            closure = 1.0 - effective_mask_prob(param)
            repair_sens = repair_sens.to(param.device, dtype=closure.dtype)
            clean_sens = clean_sens.to(param.device, dtype=closure.dtype)
            score = self.clean_open_weight * clean_sens - self.repair_close_weight * repair_sens
            loss = loss + (closure * score.detach()).mean()
            loss = loss + self.closure_l2_weight * (closure ** 2).mean()
            loss = loss + self.max_closure_weight * F.relu(closure - self.max_soft_closure).pow(2).mean()
        return loss / max(len(self.mask_params), 1)

    def _compute_sparsity_loss(self):
        if not self.mlp_mask_params:
            return torch.tensor(0.0, device=self.model.device)
        return sum((1.0 - effective_mask_prob(p)).mean() for p in self.mlp_mask_params)

    def _compute_mask_stats(self):
        stats = {"mlp_total": 0, "mlp_keep": 0, "mlp_near_threshold": 0}
        for _, module in self.model.named_modules():
            if isinstance(module, AblatedLinear):
                prob = effective_mask_prob(module.logits.detach())
                hard = module.get_mask(hard=True).detach().float() > 0.5
                stats["mlp_total"] += hard.numel()
                stats["mlp_keep"] += hard.sum().item()
                stats["mlp_near_threshold"] += ((prob > 0.45) & (prob < 0.55)).sum().item()
        return stats

    def _print_soft_mask_summary(self):
        mlp_vals = []

        for _, module in self.model.named_modules():
            if isinstance(module, AblatedLinear):
                vals = effective_mask_prob(module.logits.detach()).float().cpu().flatten()
                mlp_vals.append(vals)

        def summarize(name, vals_list):
            if len(vals_list) == 0:
                print(f"  {name}: empty")
                return

            x = torch.cat(vals_list, dim=0)
            q = torch.quantile(
                x,
                torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
            )

            print(
                f"  {name}: "
                f"min={q[0].item():.4f}, "
                f"p25={q[1].item():.4f}, "
                f"p50={q[2].item():.4f}, "
                f"p75={q[3].item():.4f}, "
                f"max={q[4].item():.4f}, "
                f"mean={x.mean().item():.4f}"
            )

        print("[Soft Mask Summary]")
        summarize("MLP      ", mlp_vals)

    @torch.no_grad()
    def _run_generation_eval(self, metric_key_prefix: str = "eval", samples: Optional[List[Dict[str, Any]]] = None):
        samples = samples if samples is not None else self.eval_generation_samples
        if self.tokenizer_for_generation is None or not samples:
            return None

        tokenizer = self.tokenizer_for_generation
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        self.model.eval()
        AblatedLinear.enabled = True

        results = []
        total = 0
        correct = 0
        no_final = 0

        for i in range(0, len(samples), self.generation_batch_size):
            batch_items = samples[i:i + self.generation_batch_size]
            prompts = [build_prompt(normalize_text(safe_get(x, ["question", "prompt"])), tokenizer) for x in batch_items]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            ).to(self.model.device)

            out = self.model.generate(
                **enc,
                max_new_tokens=self.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=get_qwen_eos_ids(tokenizer),
            )

            prompt_len = enc["input_ids"].shape[1]
            for j, item in enumerate(batch_items):
                gen_ids = out[j][prompt_len:]
                raw_special = tokenizer.decode(gen_ids, skip_special_tokens=False)
                raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
                clean = clean_generation_for_scoring(raw)

                pred = parse_final_answer_strict(clean)
                gt = normalize_text(safe_get(item, ["ground_truth", "answer", "label"]))
                pred_norm = normalize_numeric_string(pred)
                gt_norm = normalize_numeric_string(gt)
                ok = bool(pred_norm and gt_norm and pred_norm == gt_norm)

                total += 1
                correct += int(ok)
                if not pred_norm:
                    no_final += 1

                results.append({
                    "id": safe_get(item, ["id"]),
                    "question": safe_get(item, ["question", "prompt"]),
                    "ground_truth": gt,
                    "ground_truth_norm": gt_norm,
                    "corrected_reasoning": safe_get(item, ["corrected_reasoning"]),
                    "raw_generation_with_special": raw_special,
                    "raw_generation": raw,
                    "generation": clean,
                    "pred_answer": pred if pred else None,
                    "pred_answer_norm": pred_norm,
                    "is_correct": ok,
                    "has_final_answer": bool(pred_norm),
                })

        acc = correct / max(total, 1)
        no_final_rate = no_final / max(total, 1)

        print("[Generation Eval]")
        print(f"  {metric_key_prefix}_generation_acc       : {acc:.6f} ({correct}/{total})")
        print(f"  {metric_key_prefix}_no_final_answer_rate : {no_final_rate:.6f} ({no_final}/{total})")

        if self.generation_output_dir is not None:
            ensure_dir(self.generation_output_dir)
            epoch = int(self.state.epoch) if self.state and self.state.epoch is not None else -1
            path = os.path.join(self.generation_output_dir, f"{metric_key_prefix}_generations_epoch_{epoch}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": {
                        "num_total": total,
                        "num_correct": correct,
                        "generation_acc": acc,
                        "num_no_final_answer": no_final,
                        "no_final_answer_rate": no_final_rate,
                    },
                    "results": results,
                }, f, ensure_ascii=False, indent=2)
            print(f"  saved generations: {path}")

        if metric_key_prefix == "repair" and self.best_mask_path is not None and acc > self.best_generation_acc:
            self.best_generation_acc = acc
            gen_best_path = self.best_mask_path.replace("best_rescue_masks.pt", "best_generation_acc_masks.pt")
            save_all_masks(self.model, gen_best_path)
            print(f"[Best-Gen] saved masks to {gen_best_path} (generation_acc={acc:.6f})")

        tokenizer.padding_side = old_padding_side
        return {
            f"{metric_key_prefix}_generation_acc": acc,
            f"{metric_key_prefix}_no_final_answer_rate": no_final_rate,
        }

    def _save_eval_history(self):
        if self.generation_output_dir is None or not self.eval_history:
            return
        ensure_dir(self.generation_output_dir)

        json_path = os.path.join(self.generation_output_dir, "generation_eval_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.eval_history, f, ensure_ascii=False, indent=2)

        keys = sorted({k for row in self.eval_history for k in row.keys()})
        csv_path = os.path.join(self.generation_output_dir, "generation_eval_history.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.eval_history)

        try:
            import matplotlib.pyplot as plt

            epochs = [x["epoch"] for x in self.eval_history]
            plt.figure(figsize=(7, 4.5))
            if "repair_generation_acc" in self.eval_history[-1]:
                plt.plot(epochs, [x.get("repair_generation_acc", 0.0) for x in self.eval_history], marker="o", label="repair acc")
            if "clean_generation_acc" in self.eval_history[-1]:
                plt.plot(epochs, [x.get("clean_generation_acc", 0.0) for x in self.eval_history], marker="o", label="clean acc")
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(self.generation_output_dir, "curve_generation_acc_repair_vs_clean.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()

            plt.figure(figsize=(7, 4.5))
            if "repair_no_final_answer_rate" in self.eval_history[-1]:
                plt.plot(epochs, [x.get("repair_no_final_answer_rate", 0.0) for x in self.eval_history], marker="o", label="repair no-final")
            if "clean_no_final_answer_rate" in self.eval_history[-1]:
                plt.plot(epochs, [x.get("clean_no_final_answer_rate", 0.0) for x in self.eval_history], marker="o", label="clean no-final")
            plt.xlabel("epoch")
            plt.ylabel("rate")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(self.generation_output_dir, "curve_no_final_rate_repair_vs_clean.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
        except Exception as e:
            print(f"  [Eval History] plot skipped: {e}")

        print(f"  saved eval history: {json_path}")
        print(f"  saved eval history csv: {csv_path}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        AblatedLinear.enabled = True

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits


        loss_faithfulness = token_nll_loss(logits, inputs["labels"], inputs.get("loss_weights"))


        loss_sparse_mlp = self._compute_sparsity_loss()
        loss_clean_kl = torch.tensor(0.0, device=device)
        loss_clean_nll = torch.tensor(0.0, device=device)

        clean_batch = self._next_clean_batch()
        if clean_batch is not None:
            clean_batch = {k: v.to(device) for k, v in clean_batch.items()}

            AblatedLinear.enabled = True
            clean_outputs = model(
                input_ids=clean_batch["input_ids"],
                attention_mask=clean_batch["attention_mask"],
            )
            if self.clean_kl_weight > 0:
                AblatedLinear.enabled = False
                with torch.no_grad():
                    teacher_outputs = model(
                        input_ids=clean_batch["input_ids"],
                        attention_mask=clean_batch["attention_mask"],
                    )
                    AblatedLinear.enabled = True
                loss_clean_kl = token_kl_loss(
                    student_logits=clean_outputs.logits,
                    teacher_logits=teacher_outputs.logits,
                    labels=clean_batch["labels"],
                    temperature=self.clean_kl_temperature,
                )
            if self.clean_nll_weight > 0:
                loss_clean_nll = token_nll_loss(clean_outputs.logits, clean_batch["labels"])

        self._compute_probe_sensitivities(model, inputs, clean_batch)
        loss_sensitivity = self._compute_sensitivity_regularizer()

        total_loss = (
            loss_faithfulness
            + self.sparsity_weight_mlp * loss_sparse_mlp
            + self.clean_kl_weight * loss_clean_kl
            + self.clean_nll_weight * loss_clean_nll
            + loss_sensitivity
        )

        self._last_loss_stats = {
            "total_loss": float(total_loss.detach().item()),
            "loss_faithfulness": float(loss_faithfulness.detach().item()),
            "loss_sparse_mlp": float(loss_sparse_mlp.detach().item()),
            "loss_clean_kl": float(loss_clean_kl.detach().item()),
            "loss_clean_nll": float(loss_clean_nll.detach().item()),
            "loss_sensitivity": float(loss_sensitivity.detach().item()),
        }

        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss.detach(), None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        print("\n[Eval Loss Breakdown]")
        for k, v in self._last_loss_stats.items():
            print(f"  {k:20s}: {v:.6f}")

        stats = self._compute_mask_stats()
        mlp_keep_ratio = stats["mlp_keep"] / max(stats["mlp_total"], 1)

        print("[Mask Stats]")
        print(f"  MLP keep      : {stats['mlp_keep']}/{stats['mlp_total']} ({mlp_keep_ratio*100:.2f}%)")
        print(f"  MLP near threshold      : {stats['mlp_near_threshold']}/{stats['mlp_total']}")

        self._print_soft_mask_summary()
        repair_gen_metrics = self._run_generation_eval(metric_key_prefix="repair", samples=self.eval_generation_samples)
        clean_gen_metrics = self._run_generation_eval(metric_key_prefix="clean", samples=self.clean_generation_samples)
        if repair_gen_metrics:
            metrics.update(repair_gen_metrics)
        if clean_gen_metrics:
            metrics.update(clean_gen_metrics)

        history_row = {"epoch": float(self.state.epoch) if self.state and self.state.epoch is not None else len(self.eval_history)}
        history_row.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        self.eval_history.append(history_row)
        self._save_eval_history()

        current = metrics.get(f"{metric_key_prefix}_loss", None)
        if self.best_mask_path is not None and current is not None:
            if self.best_eval_loss is None or current < self.best_eval_loss:
                self.best_eval_loss = current
                save_all_masks(self.model, self.best_mask_path)
                print(f"[Best] saved best masks to {self.best_mask_path} (eval_loss={current:.6f})")

        print("")
        return metrics


@torch.no_grad()
def generate_examples(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    output_file: str,
    max_new_tokens: int = 256,
    batch_size: int = 4,
):
    model.eval()
    tokenizer.padding_side = "left"
    results = []
    total = 0
    correct = 0
    no_final = 0

    for i in range(0, len(samples), batch_size):
        batch_items = samples[i:i+batch_size]
        prompts = [build_prompt(normalize_text(safe_get(x, ["question", "prompt"])), tokenizer) for x in batch_items]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        ).to(model.device)

        AblatedLinear.enabled = True
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=get_qwen_eos_ids(tokenizer),
        )

        prompt_len = enc["input_ids"].shape[1]
        for j, item in enumerate(batch_items):
            gen_ids = out[j][prompt_len:]
            raw_special = tokenizer.decode(gen_ids, skip_special_tokens=False)
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            clean = clean_generation_for_scoring(raw)

            pred = parse_final_answer_strict(clean)
            gt = normalize_text(safe_get(item, ["ground_truth", "answer", "label"]))
            pred_norm = normalize_numeric_string(pred)
            gt_norm = normalize_numeric_string(gt)
            ok = bool(pred_norm and gt_norm and pred_norm == gt_norm)

            total += 1
            correct += int(ok)
            no_final += int(not pred_norm)

            results.append({
                "id": safe_get(item, ["id"]),
                "question": safe_get(item, ["question", "prompt"]),
                "ground_truth": gt,
                "ground_truth_norm": gt_norm,
                "input_pred_answer": safe_get(item, ["pred_answer"]),
                "corrected_reasoning": safe_get(item, ["corrected_reasoning"]),
                "raw_generation_with_special": raw_special,
                "raw_generation": raw,
                "generation": clean,
                "pred_answer": pred if pred else None,
                "pred_answer_norm": pred_norm,
                "is_correct": ok,
                "has_final_answer": bool(pred_norm),
            })

    summary = {
        "num_total": total,
        "num_correct": correct,
        "generation_acc": correct / max(total, 1),
        "num_no_final_answer": no_final,
        "no_final_answer_rate": no_final / max(total, 1),
        "output_file": output_file,
    }

    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print("[Generation Summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Generation] saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Discover an MLP-neuron repair circuit with STE masks.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repair-data-path", required=True)
    parser.add_argument("--clean-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-mask-path")
    parser.add_argument("--train-repair-samples", type=int, default=80)
    parser.add_argument("--eval-repair-samples", type=int, default=20)
    parser.add_argument("--train-clean-samples", type=int, default=80)
    parser.add_argument("--eval-clean-samples", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clean-batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--clean-kl-weight", type=float, default=0.0)
    parser.add_argument("--clean-nll-weight", type=float, default=1.0)
    parser.add_argument("--clean-kl-temperature", type=float, default=1.0)
    parser.add_argument("--mask-init-value", type=float, default=0.2)
    parser.add_argument("--mask-temperature", type=float, default=1.0)
    parser.add_argument("--use-sensitivity-regularizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sensitivity-probe-offset", type=float, default=0.10)
    parser.add_argument("--sensitivity-probe-every", type=int, default=4)
    parser.add_argument("--repair-close-weight", type=float, default=0.05)
    parser.add_argument("--clean-open-weight", type=float, default=0.20)
    parser.add_argument("--closure-l2-weight", type=float, default=0.02)
    parser.add_argument("--max-closure", type=float, default=0.35)
    parser.add_argument("--max-closure-weight", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    ensure_dir(args.output_dir)
    configure_ste(args.mask_temperature, args.mask_init_value)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    disable_generation_max_length_warning(model)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    mlp_mask_params = patch_model(model)
    configure_ste(args.mask_temperature, args.mask_init_value)

    if args.resume_mask_path is not None and os.path.exists(args.resume_mask_path):
        state = torch.load(args.resume_mask_path, map_location="cpu")
        n2 = AblatedLinear.load_masks(model, state)
        print(f"[Resume] loaded {n2} neuron-mask modules from {args.resume_mask_path}")
    else:
        print("[Resume] no resume mask loaded")

    repair_data_all = load_json_auto(args.repair_data_path)
    clean_data_all = load_json_auto(args.clean_data_path) if args.clean_data_path else []

    train_repair = repair_data_all[:args.train_repair_samples]
    eval_repair = repair_data_all[args.train_repair_samples:args.train_repair_samples + args.eval_repair_samples]
    train_clean = clean_data_all[:args.train_clean_samples]
    eval_clean = clean_data_all[args.train_clean_samples:args.train_clean_samples + args.eval_clean_samples]

    print("\n[Data Split Check]")
    print(f"Total repair samples : {len(repair_data_all)}")
    print(f"Train repair samples : {len(train_repair)}")
    print(f"Eval repair samples  : {len(eval_repair)}")
    print(f"Total clean samples  : {len(clean_data_all)}")
    print(f"Train clean samples  : {len(train_clean)}")
    print(f"Eval clean samples   : {len(eval_clean)}")
    print("")

    if len(train_repair) == 0:
        raise ValueError("train_repair is empty.")
    if len(eval_repair) == 0:
        raise ValueError("eval_repair is empty. Reduce args.train_repair_samples or provide more data.")

    train_dataset = RESCUERepairDataset(
        data_list=train_repair,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    eval_dataset = RESCUERepairDataset(
        data_list=eval_repair,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    clean_dataset = CleanPreserveDataset(
        data_list=train_clean,
        tokenizer=tokenizer,
        max_length=args.max_length,
    ) if train_clean else None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_strategy="no",
        eval_strategy="epoch",
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        data_seed=42,
        bf16=True,
        max_grad_norm=1.0,
    )

    trainer = RESCUERepairTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SimpleCollator(),
        sparsity_weight_mlp=args.sparsity_weight,
        mlp_mask_params=mlp_mask_params,
        best_mask_path=os.path.join(args.output_dir, "best_rescue_masks.pt"),
        tokenizer_for_generation=tokenizer,
        eval_generation_samples=eval_repair,
        clean_generation_samples=eval_clean,
        generation_output_dir=args.output_dir,
        generation_max_new_tokens=256,
        generation_batch_size=16,
        clean_dataset=clean_dataset,
        clean_batch_size=args.clean_batch_size,
        clean_kl_weight=args.clean_kl_weight,
        clean_nll_weight=args.clean_nll_weight,
        clean_kl_temperature=args.clean_kl_temperature,
        use_sensitivity_regularizer=args.use_sensitivity_regularizer,
        sensitivity_probe_offset=args.sensitivity_probe_offset,
        sensitivity_probe_every=args.sensitivity_probe_every,
        repair_close_weight=args.repair_close_weight,
        clean_open_weight=args.clean_open_weight,
        closure_l2_weight=args.closure_l2_weight,
        max_soft_closure=args.max_closure,
        max_closure_weight=args.max_closure_weight,
    )

    trainer.train()

    final_mask_path = os.path.join(args.output_dir, "final_rescue_masks.pt")
    save_all_masks(model, final_mask_path)

    gen_path = os.path.join(args.output_dir, "eval_generations.json")
    generate_examples(model, tokenizer, eval_repair, gen_path)

    print("\n" + "=" * 90)
    print("Training finished.")
    print(f"Best masks : {os.path.join(args.output_dir, 'best_rescue_masks.pt')}")
    print(f"Final masks: {final_mask_path}")
    print(f"Generations: {gen_path}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
