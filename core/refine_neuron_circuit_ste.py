import argparse
import os
import re
import json
import math
import csv
import gc
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


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


def normalize_answer_str(x: str) -> str:

    x = normalize_text(x)
    if not x:
        return ""
    m = _NUMBER_RE.search(x)
    if not m:
        return x.lower().strip()
    val = m.group(0).replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        f = float(val)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return ("%.10f" % f).rstrip("0").rstrip(".")
    except Exception:
        return val


def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_generation_for_scoring(text: str) -> str:

    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    for token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        text = text.replace(token, "")
    cut_position = len(text)
    for marker in [
        "\nQuestion:", "\nHuman:", "\nComment:", "\nInstruction:",
        "\nUser:", "\nAssistant:", "\n[INST]",
    ]:
        position = text.find(marker)
        if position != -1:
            cut_position = min(cut_position, position)
    return text[:cut_position].strip()


def get_qwen_eos_ids(tokenizer):
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
    return eos_ids if eos_ids else None


def _last_number(text: str) -> str:
    nums = _NUMBER_RE.findall(text or "")
    return normalize_answer_str(nums[-1]) if nums else ""


def truncate_after_final_answer_for_display(text: str) -> str:

    text = strip_think_blocks(normalize_text(text))
    if not text:
        return ""


    text = re.sub(
        r"^\s*Final\s+answer\s*[:：]\s*[^\n]+\n+\s*(?:Step\s+by\s+step\.?)\s*",
        "Step by step.\n\n",
        text,
        flags=re.IGNORECASE,
    ).strip()

    m = re.search(r"Final\s+answer\s*[:：]\s*[^\n]+", text, flags=re.IGNORECASE)
    if m:
        return text[:m.end()].strip()
    return text.strip()


def extract_final_answer(text: str) -> str:

    text = clean_generation_for_scoring(text)
    if not text:
        return ""

    matches = list(re.finditer(r"Final\s+answer\s*[:：]\s*([^\n]+)", text, flags=re.IGNORECASE))
    if matches:
        ans_line = matches[-1].group(1).strip()
        if "=" in ans_line:
            rhs = ans_line.split("=")[-1]
            rhs_num = _last_number(rhs)
            if rhs_num:
                return rhs_num
        ans_num = _last_number(ans_line)
        return ans_num if ans_num else ans_line

    matches = list(re.finditer(r"####\s*([^\n\r]+)", text))
    if matches:
        ans_line = matches[-1].group(1).strip()
        ans_num = _last_number(ans_line)
        return ans_num if ans_num else ans_line

    return ""


def is_answer_correct(pred_text: str, ground_truth: str) -> float:
    pred = normalize_answer_str(extract_final_answer(pred_text))
    gt = normalize_answer_str(ground_truth)
    return 1.0 if pred != "" and pred == gt else 0.0


def bucketize_score(x: float, levels=(0.0, 0.25, 0.5, 0.75, 1.0)) -> float:
    x = float(max(0.0, min(1.0, x)))
    return min(levels, key=lambda y: abs(y - x))


def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_first_existing_path(paths: List[Optional[str]]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def disable_generation_max_length_warning(model):

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and hasattr(gen_cfg, "max_length"):
        gen_cfg.max_length = None
    return model


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
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.strip()},
        ]
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


class AblatedLinear(nn.Module):
    enabled = True
    mask_temperature = 1.0
    mask_init_value = 0.2
    hard_threshold = 0.5
    max_delete_ratio = 1.0
    sensitivity_probe_offset = 0.0

    def __init__(self, original_layer: nn.Linear, init_value: float = 0.2):
        super().__init__()
        self.original_layer = original_layer
        for p in self.original_layer.parameters():
            p.requires_grad = False
        out_features = original_layer.out_features
        device = original_layer.weight.device
        self.init_value = init_value
        self.logits = nn.Parameter(torch.ones(out_features, device=device, dtype=torch.float32) * init_value)

    def get_mask(self, hard: Optional[bool] = None):
        surrogate = effective_mask_prob(self.logits)
        mask_hard = hard_mask_from_soft(
            surrogate, self.hard_threshold, self.max_delete_ratio
        ).to(surrogate.dtype)
        mask = mask_hard - surrogate.detach() + surrogate
        return mask.to(self.original_layer.weight.dtype)

    def forward(self, x):
        out = self.original_layer(x)
        if not AblatedLinear.enabled:
            return out
        return out * self.get_mask()

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
                module.logits.data.copy_(state[name].to(module.logits.device, dtype=module.logits.dtype))
                count += 1
        return count


def save_all_masks(model, path: str):
    torch.save(AblatedLinear.save_masks(model), path)
    print(f"[Save] neuron-mask logits -> {path}")


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


def save_binary_masks(model, path: str):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, AblatedLinear):
            prob = effective_mask_prob(module.logits.detach())
            state[name] = hard_mask_from_soft(
                prob, AblatedLinear.hard_threshold, AblatedLinear.max_delete_ratio
            ).to(torch.uint8).cpu()
    torch.save(state, path)
    print(f"[Save] binary neuron masks -> {path}")


def configure_ste(temperature=1.0, init_value=0.2, hard_threshold=0.5, max_delete_ratio=1.0):

    if temperature <= 0:
        raise ValueError("mask temperature must be > 0")
    if not 0.0 < hard_threshold < 1.0:
        raise ValueError("hard threshold must be in (0, 1)")
    if not 0.0 <= max_delete_ratio <= 1.0:
        raise ValueError("max delete ratio must be in [0, 1]")
    AblatedLinear.mask_temperature = temperature
    AblatedLinear.mask_init_value = init_value
    AblatedLinear.hard_threshold = hard_threshold
    AblatedLinear.max_delete_ratio = max_delete_ratio


def set_mask_probe_offset(offset: float):
    AblatedLinear.sensitivity_probe_offset = offset


def get_delete_ratio_for_epoch(epoch: int, schedule: List[float], final_ratio: float) -> float:

    if epoch <= 0:
        return float(schedule[0] if schedule else final_ratio)
    if epoch <= len(schedule):
        return float(schedule[epoch - 1])
    return float(final_ratio)


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

    targets = ("gate_proj", "up_proj", "down_proj")
    params = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith(targets):
            parent_name, attr_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            wrapped = AblatedLinear(module, init_value=AblatedLinear.mask_init_value)
            setattr(parent, attr_name, wrapped)
            params.append(wrapped.logits)
    print(f"[Patch] patched {len(params)} neuron-mask projections")
    return params


def get_anchor_refs(mask_params, anchor_mode="init"):
    refs = []
    for p in mask_params:
        if anchor_mode == "init":
            refs.append(effective_mask_prob(p.detach()).clone())
        else:
            refs.append(p.detach().clone())
    return refs


def compute_anchor_loss(mask_params, refs, anchor_mode="init"):
    device = mask_params[0].device if mask_params else "cpu"
    loss = torch.tensor(0.0, device=device)
    for p, r in zip(mask_params, refs):
        if anchor_mode == "init":
            loss = loss + F.mse_loss(effective_mask_prob(p), r.to(p.device, dtype=torch.float32))
        else:
            loss = loss + F.mse_loss(p, r.to(p.device, dtype=torch.float32))
    return loss


@dataclass
class RLItem:
    qid: Any
    question: str
    ground_truth: str
    repair_target_text: str = ""


@dataclass
class CleanItem:
    qid: Any
    question: str
    target_text: str
    ground_truth: str


def build_rl_items(data_list: List[Dict[str, Any]]) -> List[RLItem]:
    items = []
    for i, item in enumerate(data_list):
        question = normalize_text(safe_get(item, ["question", "prompt"]))
        gt = normalize_text(safe_get(item, ["ground_truth", "answer"]))
        if question and gt:
            target = normalize_text(safe_get(item, [
                "corrected_reasoning",
                "corrected_response",
                "solution",
                "target",
            ]))
            if not target:
                target = f"Final Answer: {gt}"
            elif extract_final_answer(target) == "":
                target = target.rstrip() + f"\nFinal Answer: {gt}"
            items.append(RLItem(
                qid=safe_get(item, ["id"], i),
                question=question,
                ground_truth=gt,
                repair_target_text=target,
            ))
    return items


def compute_repair_replay_nll(model, tokenizer, items: List[RLItem],
                              max_total_len: int) -> torch.Tensor:
    replay_items = [
        CleanItem(
            qid=item.qid,
            question=item.question,
            target_text=item.repair_target_text,
            ground_truth=item.ground_truth,
        )
        for item in items
        if item.repair_target_text
    ]
    if not replay_items:
        return torch.tensor(0.0, device=model.device)
    _, replay_nll = compute_clean_preserve_loss(
        model=model,
        tokenizer=tokenizer,
        clean_items=replay_items,
        batch_size=len(replay_items),
        max_total_len=max_total_len,
        kl_temperature=1.0,
        use_kl=False,
    )
    return replay_nll


def compare_current_binary_mask(model, expected_state):

    mismatched, compared, modules = 0, 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, AblatedLinear) or name not in expected_state:
            continue
        soft_mask = effective_mask_prob(module.logits.detach())
        current = hard_mask_from_soft(
            soft_mask,
            AblatedLinear.hard_threshold,
            AblatedLinear.max_delete_ratio,
        ).to(torch.uint8).cpu()
        expected = expected_state[name].to(torch.uint8).cpu()
        if current.shape != expected.shape:
            raise ValueError(
                f"Binary mask shape mismatch for {name}: "
                f"{tuple(current.shape)} != {tuple(expected.shape)}"
            )
        mismatched += int((current != expected).sum().item())
        compared += current.numel()
        modules += 1
    return mismatched, compared, modules


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
    answer = normalize_text(safe_get(item, ["ground_truth", "answer", "label"]))
    if response:
        if extract_final_answer(response) == "" and answer:
            response = response.rstrip() + f"\nFinal answer: {answer}"
        return response
    if answer:
        return f"Final answer: {answer}"
    return ""


def build_clean_items(data_list: List[Dict[str, Any]]) -> List[CleanItem]:
    items = []
    for i, item in enumerate(data_list):
        question = normalize_text(safe_get(item, ["question", "prompt"]))
        target_text = select_clean_target(item)
        gt = normalize_text(safe_get(item, ["ground_truth", "answer", "label"]))
        if question and target_text:
            items.append(CleanItem(
                qid=safe_get(item, ["id"], i),
                question=question,
                target_text=target_text,
                ground_truth=gt,
            ))
    return items


    print("RL mask refinement v4: anchor + graded answer/reasoning reward + train/eval separated curves")


class ReasoningJudge:
    def __init__(self, model_path: str, torch_dtype=torch.bfloat16,
                 device_map="auto", use_graded_reward=True, batch_size: int = 1):
        self.model_path = model_path
        self.use_graded_reward = use_graded_reward
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype, device_map=device_map, trust_remote_code=True)
        disable_generation_max_length_warning(self.model)
        self.model.eval()

    def _build_prompt(self, question: str, ground_truth: str, response: str) -> str:
        if self.use_graded_reward:
            user_text = (
                "You are a strict math-solution judge.\n"
                "Evaluate the model response with one score chosen ONLY from {0, 0.25, 0.5, 0.75, 1.0}.\n"
                "Return only one JSON object in this exact schema:\n"
                '{"reasoning_score": 0.75}\n\n'
                "Scoring rules:\n"
                "- reasoning_score: 1.0 means fully correct reasoning and consistent final answer; 0 means wrong or nonsense; intermediate values indicate partial correctness.\n"
                f"Question:\n{question}\n\n"
                f"Ground truth answer:\n{ground_truth}\n\n"
                f"Model response:\n{response}\n"
            )
        else:
            user_text = (
                "You are a strict math-solution judge.\n"
                "Return only one JSON object in exactly one of the following forms:\n"
                '{"reasoning_correct": 1}\n'
                '{"reasoning_correct": 0}\n\n'
                f"Question:\n{question}\n\nGround truth answer:\n{ground_truth}\n\nModel response:\n{response}\n"
            )
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": user_text}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return user_text

    @staticmethod
    def _parse_graded_json(text: str) -> float:
        txt = normalize_text(text)
        ms = re.search(r'"reasoning_score"\s*:\s*([0-9.]+)', txt)
        if ms:
            return bucketize_score(float(ms.group(1)))
        nums = re.findall(r"(?:0(?:\.25|\.5|\.75)?|1(?:\.0)?)", txt)
        if len(nums) >= 1:
            return bucketize_score(float(nums[0]))
        return 0.0

    @staticmethod
    def _parse_binary_reasoning(text: str) -> float:
        txt = normalize_text(text)
        m = re.search(r'"reasoning_correct"\s*:\s*([01])', txt)
        if m:
            return float(int(m.group(1)))
        nums = re.findall(r"[01]", txt)
        return float(int(nums[-1])) if nums else 0.0

    @torch.no_grad()
    def score_batch(self, batch_questions: List[str], batch_ground_truths: List[str], batch_responses: List[str], max_new_tokens: int = 48):
        prompts = [self._build_prompt(q, gt, resp) for q, gt, resp in zip(batch_questions, batch_ground_truths, batch_responses)]
        results = []
        for start in range(0, len(prompts), self.batch_size):
            prompt_batch = prompts[start:start + self.batch_size]
            enc = self.tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                add_special_tokens=False,
            ).to(self.model.device)
            out = self.model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=get_qwen_eos_ids(self.tokenizer),
            )
            prompt_len = enc["input_ids"].shape[1]
            for i in range(len(prompt_batch)):
                gen_ids = out[i][prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                if self.use_graded_reward:
                    rs = self._parse_graded_json(text)
                else:
                    rs = self._parse_binary_reasoning(text)
                results.append({"reasoning_score": float(rs), "judge_raw": text})
            del enc, out
        return results


@torch.no_grad()
def sample_group_rollouts(model, tokenizer, items: List[RLItem], group_size: int,
                          max_prompt_len: int, max_new_tokens: int,
                          temperature: float, top_p: float) -> List[Dict[str, Any]]:
    tokenizer.padding_side = "left"
    model.eval()
    all_groups = []
    for item in items:
        prompt = build_chat_prompt(item.question, tokenizer)
        prompts = [prompt] * group_size
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_prompt_len, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                             max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=get_qwen_eos_ids(tokenizer))
        group = {"qid": item.qid, "question": item.question, "ground_truth": item.ground_truth, "rollouts": []}
        prompt_len = enc["input_ids"].shape[1]
        for j in range(group_size):
            real_prompt_len = int(enc["attention_mask"][j].sum().item())
            prompt_ids = enc["input_ids"][j][-real_prompt_len:].detach().cpu()
            completion_ids = out[j][prompt_len:].detach().cpu()
            text_raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
            text_for_reward = truncate_after_final_answer_for_display(text_raw)
            group["rollouts"].append({
                "prompt_text": prompt,
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "generation_text": text_raw,
                "generation_text_for_reward": text_for_reward,
                "pred_answer_for_reward": extract_final_answer(text_raw),
            })
        all_groups.append(group)
    return all_groups


def assign_rewards_to_groups(groups: List[Dict[str, Any]], judge: ReasoningJudge,
                             answer_reward_weight: float = 1.0,
                             reasoning_reward_weight: float = 1.0):
    batch_questions, batch_ground_truths, batch_responses = [], [], []
    for group in groups:
        for rollout in group["rollouts"]:
            batch_questions.append(group["question"])
            batch_ground_truths.append(group["ground_truth"])
            batch_responses.append(rollout.get("generation_text_for_reward", rollout["generation_text"]))
    judge_outputs = judge.score_batch(batch_questions, batch_ground_truths, batch_responses)

    idx = 0
    for group in groups:
        for rollout in group["rollouts"]:
            ans_reward = is_answer_correct(rollout["generation_text"], group["ground_truth"])
            reasoning_reward = float(judge_outputs[idx]["reasoning_score"])
            total_reward = (
                answer_reward_weight * ans_reward
                + reasoning_reward_weight * reasoning_reward
            )
            rollout["answer_reward"] = float(ans_reward)
            rollout["pred_answer_for_reward"] = extract_final_answer(rollout.get("generation_text", ""))
            rollout["generation_text_for_reward"] = rollout.get("generation_text_for_reward", truncate_after_final_answer_for_display(rollout.get("generation_text", "")))
            rollout["reasoning_reward"] = reasoning_reward
            rollout["reward"] = float(total_reward)
            rollout["judge_raw"] = judge_outputs[idx].get("judge_raw", "")
            idx += 1


def build_scoring_batch_from_rollouts(groups: List[Dict[str, Any]], pad_token_id: int, max_total_len: int = 768):
    flat_rollouts, advantages, meta = [], [], []
    for group in groups:
        rewards = [x["reward"] for x in group["rollouts"]]
        mean_r = float(np.mean(rewards))
        std_r = max(float(np.std(rewards)), 1e-6)
        local_adv = [(r - mean_r) / std_r for r in rewards]
        for rollout, adv in zip(group["rollouts"], local_adv):
            flat_rollouts.append(rollout)
            advantages.append(float(adv))
            meta.append({
                "qid": group["qid"],
                "question": group["question"],
                "ground_truth": group["ground_truth"],
                "generation_text": rollout["generation_text"],
                "generation_text_for_reward": rollout.get("generation_text_for_reward", ""),
                "pred_answer_for_reward": rollout.get("pred_answer_for_reward", ""),
                "reward": rollout["reward"],
                "answer_reward": rollout.get("answer_reward", 0.0),
                "reasoning_reward": rollout.get("reasoning_reward", 0.0),
            })
    full_ids_list, labels_list, attn_list = [], [], []
    kept_advantages, kept_meta = [], []
    max_len_in_batch = 0
    for rollout, adv, m in zip(flat_rollouts, advantages, meta):
        prompt_ids = rollout["prompt_ids"]
        completion_ids = rollout["completion_ids"]
        full_ids = torch.cat([prompt_ids, completion_ids], dim=0)
        if full_ids.numel() > max_total_len:
            full_ids = full_ids[:max_total_len]
        prompt_len = min(prompt_ids.numel(), full_ids.numel())
        if full_ids.numel() <= 1 or prompt_len >= full_ids.numel():
            continue
        labels = full_ids.clone(); labels[:prompt_len] = -100
        attn = torch.ones_like(full_ids)
        full_ids_list.append(full_ids); labels_list.append(labels); attn_list.append(attn)
        kept_advantages.append(adv); kept_meta.append(m)
        max_len_in_batch = max(max_len_in_batch, full_ids.numel())
    if not full_ids_list:
        raise ValueError("All rollouts were truncated into invalid scoring samples.")
    def pad_1d(x: torch.Tensor, pad_value: int, target_len: int):
        if x.numel() == target_len:
            return x
        pad = torch.full((target_len - x.numel(),), pad_value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)
    input_ids = torch.stack([pad_1d(x, pad_token_id, max_len_in_batch) for x in full_ids_list], dim=0)
    attention_mask = torch.stack([pad_1d(x, 0, max_len_in_batch) for x in attn_list], dim=0)
    labels = torch.stack([pad_1d(x, -100, max_len_in_batch) for x in labels_list], dim=0)
    return input_ids, attention_mask, labels, kept_advantages, kept_meta


def compute_sequence_mean_logprob(model, input_ids, attention_mask, labels) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    valid = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logprobs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * valid
    lengths = valid.sum(dim=-1).clamp_min(1)
    return token_logprobs.sum(dim=-1) / lengths


def token_nll_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def token_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    shift_student = student_logits[:, :-1, :].float().contiguous() / temperature
    shift_teacher = teacher_logits[:, :-1, :].float().contiguous() / temperature
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    student_log_probs = F.log_softmax(shift_student, dim=-1)
    teacher_probs = F.softmax(shift_teacher, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    kl = kl * valid
    return kl.sum() / valid.sum().clamp_min(1) * (temperature ** 2)


def build_clean_preserve_batch(items: List[CleanItem], tokenizer, batch_size: int, max_total_len: int):
    if not items or batch_size <= 0:
        return None
    batch = random.sample(items, k=min(batch_size, len(items)))
    full_ids_list, labels_list, attn_list = [], [], []
    max_len = 0
    for item in batch:
        prompt = build_chat_prompt(item.question, tokenizer)
        full_text = prompt + item.target_text + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_total_len]
        if len(full_ids) <= 1:
            continue
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), input_ids.numel())
        labels[:prompt_len] = -100
        if labels.ne(-100).sum().item() == 0:
            continue
        attn = torch.ones_like(input_ids)
        full_ids_list.append(input_ids)
        labels_list.append(labels)
        attn_list.append(attn)
        max_len = max(max_len, input_ids.numel())
    if not full_ids_list:
        return None

    def pad_1d(x: torch.Tensor, pad_value: int, target_len: int):
        if x.numel() == target_len:
            return x
        pad = torch.full((target_len - x.numel(),), pad_value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    input_ids = torch.stack([pad_1d(x, tokenizer.pad_token_id, max_len) for x in full_ids_list], dim=0)
    attention_mask = torch.stack([pad_1d(x, 0, max_len) for x in attn_list], dim=0)
    labels = torch.stack([pad_1d(x, -100, max_len) for x in labels_list], dim=0)
    return input_ids, attention_mask, labels


def compute_clean_preserve_loss(model, tokenizer, clean_items: List[CleanItem],
                                batch_size: int, max_total_len: int,
                                kl_temperature: float = 1.0,
                                use_kl: bool = True):
    batch = build_clean_preserve_batch(clean_items, tokenizer, batch_size, max_total_len)
    device = model.device
    if batch is None:
        zero = torch.tensor(0.0, device=device)
        return zero, zero
    input_ids, attention_mask, labels = [x.to(device) for x in batch]

    AblatedLinear.enabled = True
    AblatedLinear.enabled = True
    student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if use_kl:
        AblatedLinear.enabled = False
        AblatedLinear.enabled = False
        with torch.no_grad():
            teacher_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        AblatedLinear.enabled = True
        clean_kl = token_kl_loss(student_outputs.logits, teacher_outputs.logits, labels, temperature=kl_temperature)
    else:
        clean_kl = torch.tensor(0.0, device=device)
    clean_nll = token_nll_loss(student_outputs.logits, labels)
    return clean_kl, clean_nll


def clone_param_grads(params):
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def zero_param_grads(params):
    for p in params:
        p.grad = None


def project_grads_against_clean(
    repair_grads,
    clean_grads,
    strength: float = 1.0,
    eps: float = 1e-12,
):
    strength = float(max(0.0, min(1.0, strength)))
    dot = None
    norm = None
    for rg, cg in zip(repair_grads, clean_grads):
        if rg is None or cg is None:
            continue
        cur_dot = (rg.float() * cg.float()).sum()
        cur_norm = (cg.float() * cg.float()).sum()
        dot = cur_dot if dot is None else dot + cur_dot
        norm = cur_norm if norm is None else norm + cur_norm
    if dot is None or norm is None or norm.item() <= eps:
        return repair_grads, 0.0

    if dot.item() >= 0.0:
        return repair_grads, 0.0
    coef = dot / norm.clamp_min(eps)
    applied_coef = strength * coef
    projected = []
    for rg, cg in zip(repair_grads, clean_grads):
        if rg is None:
            projected.append(None)
        elif cg is None:
            projected.append(rg)
        else:
            projected.append(
                rg
                - applied_coef.to(rg.device, dtype=rg.dtype)
                * cg.to(rg.device, dtype=rg.dtype)
            )
    return projected, float(applied_coef.detach().item())


def set_param_grads(params, repair_grads, clean_grads=None, clean_weight: float = 0.0):
    for p, rg, cg in zip(params, repair_grads, clean_grads or [None] * len(params)):
        grad = None
        if rg is not None:
            grad = rg.clone()
        if cg is not None and clean_weight != 0.0:
            cg = cg.to(p.device, dtype=p.dtype)
            grad = clean_weight * cg if grad is None else grad + clean_weight * cg
        p.grad = grad


def update_sensitivity_ema(grads, ema_refs, params, beta: float, direction: str):
    for i, (grad, param) in enumerate(zip(grads, params)):
        if grad is None:
            signal = torch.zeros_like(param, dtype=torch.float32)
        elif direction == "close":
            signal = F.relu(grad.detach().float())
        elif direction == "open":
            signal = F.relu(-grad.detach().float())
        else:
            raise ValueError(f"Unsupported sensitivity direction: {direction}")
        ema_refs[i].mul_(beta).add_(signal.to(ema_refs[i].device), alpha=1.0 - beta)


def compute_probe_sensitivities_rl(model, tokenizer, mask_params, input_ids, attention_mask, labels, advantages,
                                   clean_items, clean_batch_size: int, max_total_len: int,
                                   kl_temperature: float, probe_offset: float,
                                   repair_ema, clean_ema, ema_beta: float,
                                   clean_use_kl: bool = True):
    if not torch.is_grad_enabled() or not mask_params:
        return

    set_mask_probe_offset(probe_offset)
    try:
        AblatedLinear.enabled = True
        AblatedLinear.enabled = True
        seq_mean_logprob = compute_sequence_mean_logprob(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        adv_tensor = torch.tensor(advantages, device=model.device, dtype=seq_mean_logprob.dtype)
        repair_probe_loss = -(adv_tensor.detach() * seq_mean_logprob).mean()
        repair_grads = torch.autograd.grad(
            repair_probe_loss,
            mask_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        update_sensitivity_ema(repair_grads, repair_ema, mask_params, ema_beta, direction="close")

        if clean_items:
            clean_probe_kl, clean_probe_nll = compute_clean_preserve_loss(
                model=model,
                tokenizer=tokenizer,
                clean_items=clean_items,
                batch_size=clean_batch_size,
                max_total_len=max_total_len,
                kl_temperature=kl_temperature,
                use_kl=clean_use_kl,
            )
            clean_probe_loss = clean_probe_kl if clean_use_kl else clean_probe_nll
            clean_grads = torch.autograd.grad(
                clean_probe_loss,
                mask_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            update_sensitivity_ema(clean_grads, clean_ema, mask_params, ema_beta, direction="open")
    finally:
        set_mask_probe_offset(0.0)
        configure_ste(
            AblatedLinear.mask_temperature,
            AblatedLinear.mask_init_value,
            AblatedLinear.hard_threshold,
            AblatedLinear.max_delete_ratio,
        )
        AblatedLinear.enabled = True
        AblatedLinear.enabled = True


def compute_sensitivity_regularizer(mask_params, repair_ema, clean_ema,
                                    repair_close_weight: float, clean_open_weight: float,
                                    closure_l2_weight: float,
                                    max_soft_closure: float, max_closure_weight: float,
                                    device):
    if not mask_params:
        return torch.tensor(0.0, device=device)

    loss = torch.tensor(0.0, device=device)
    for param, repair_sens, clean_sens in zip(mask_params, repair_ema, clean_ema):
        closure = 1.0 - effective_mask_prob(param)
        repair_sens = repair_sens.to(param.device, dtype=closure.dtype)
        clean_sens = clean_sens.to(param.device, dtype=closure.dtype)
        score = clean_open_weight * clean_sens - repair_close_weight * repair_sens
        loss = loss + (closure * score.detach()).mean()
        loss = loss + closure_l2_weight * (closure ** 2).mean()
        loss = loss + max_closure_weight * F.relu(closure - max_soft_closure).pow(2).mean()
    return loss / max(len(mask_params), 1)


def compute_sparsity_loss(neuron_mask_params, device):
    if not neuron_mask_params:
        return torch.tensor(0.0, device=device)
    return sum(effective_mask_prob(p).mean() for p in neuron_mask_params) / len(neuron_mask_params)


def compute_mask_stats(model):
    stats = {
        "mlp_total": 0, "mlp_keep": 0,
        "mlp_mask_min": 1.0, "mlp_mask_mean": 1.0,
        "mlp_closure_mean": 0.0, "mlp_changed_001": 0,
        "mlp_changed_005": 0, "mlp_changed_010": 0,
        "mlp_changed_020": 0, "mlp_deleted": 0,
        "mlp_delete_budget": 0, "mlp_delete_ratio": 0.0,
    }
    masks = []
    for _, module in model.named_modules():
        if isinstance(module, AblatedLinear):
            prob = effective_mask_prob(module.logits.detach()).float().cpu()
            hard = hard_mask_from_soft(prob, AblatedLinear.hard_threshold, AblatedLinear.max_delete_ratio).bool()
            stats["mlp_total"] += hard.numel()
            stats["mlp_keep"] += hard.sum().item()
            stats["mlp_deleted"] += int((~hard).sum().item())
            stats["mlp_delete_budget"] += int(hard.numel() * AblatedLinear.max_delete_ratio)
            masks.append(prob.flatten())
    if masks:
        values = torch.cat(masks)
        closure = 1.0 - values
        stats["mlp_mask_min"] = float(values.min().item())
        stats["mlp_mask_mean"] = float(values.mean().item())
        stats["mlp_closure_mean"] = float(closure.mean().item())
        for suffix, threshold in (("001", .01), ("005", .05), ("010", .10), ("020", .20)):
            stats[f"mlp_changed_{suffix}"] = int((closure > threshold).sum().item())
        stats["mlp_delete_ratio"] = stats["mlp_deleted"] / max(stats["mlp_total"], 1)
    return stats


def _parse_layer_and_proj(name: str):
    layer_match = re.search(r"layers\.(\d+)", name)
    layer = int(layer_match.group(1)) if layer_match else -1
    proj = name.split(".")[-1]
    return layer, proj


def save_mask_visualizations(model, output_dir: str, tag: str, top_k: int = 200):
    mask_dir = os.path.join(output_dir, "mask_visualizations")
    ensure_dir(mask_dir)

    module_rows = []
    top_entries = []
    all_masks = []
    heatmap = {}

    for name, module in model.named_modules():
        if not isinstance(module, AblatedLinear):
            continue
        mask = effective_mask_prob(module.logits.detach()).float().cpu().flatten()
        closure = 1.0 - mask
        layer, proj = _parse_layer_and_proj(name)
        q = torch.quantile(mask, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32))
        row = {
            "tag": tag,
            "module": name,
            "layer": layer,
            "proj": proj,
            "num_neurons": int(mask.numel()),
            "mask_min": float(q[0].item()),
            "mask_p25": float(q[1].item()),
            "mask_p50": float(q[2].item()),
            "mask_p75": float(q[3].item()),
            "mask_max": float(q[4].item()),
            "mask_mean": float(mask.mean().item()),
            "closure_mean": float(closure.mean().item()),
            "closure_max": float(closure.max().item()),
            "changed_gt_001": int((closure > 0.01).sum().item()),
            "changed_gt_005": int((closure > 0.05).sum().item()),
            "changed_gt_010": int((closure > 0.10).sum().item()),
            "changed_gt_020": int((closure > 0.20).sum().item()),
            "hard_closed": int((hard_mask_from_soft(mask, AblatedLinear.hard_threshold, AblatedLinear.max_delete_ratio) == 0).sum().item()),
        }
        module_rows.append(row)
        all_masks.append(mask)
        heatmap[(layer, proj)] = row["closure_mean"]

        k = min(top_k, closure.numel())
        vals, idxs = torch.topk(closure, k=k)
        for val, idx in zip(vals.tolist(), idxs.tolist()):
            top_entries.append({
                "tag": tag,
                "module": name,
                "layer": layer,
                "proj": proj,
                "neuron_index": int(idx),
                "mask": float(mask[idx].item()),
                "closure": float(val),
            })

    if not module_rows:
        return {}

    module_csv = os.path.join(mask_dir, f"{tag}_mask_module_summary.csv")
    with open(module_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(module_rows[0].keys()))
        writer.writeheader()
        writer.writerows(module_rows)

    top_entries = sorted(top_entries, key=lambda x: x["closure"], reverse=True)[:top_k]
    top_csv = os.path.join(mask_dir, f"{tag}_top_changed_neurons.csv")
    with open(top_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(top_entries[0].keys()))
        writer.writeheader()
        writer.writerows(top_entries)

    x = torch.cat(all_masks)
    closure = 1.0 - x
    summary = {
        "tag": tag,
        "num_mlp_neurons": int(x.numel()),
        "mask_min": float(x.min().item()),
        "mask_mean": float(x.mean().item()),
        "closure_mean": float(closure.mean().item()),
        "closure_max": float(closure.max().item()),
        "changed_gt_001": int((closure > 0.01).sum().item()),
        "changed_gt_005": int((closure > 0.05).sum().item()),
        "changed_gt_010": int((closure > 0.10).sum().item()),
        "changed_gt_020": int((closure > 0.20).sum().item()),
        "hard_threshold": float(AblatedLinear.hard_threshold),
        "hard_closed": int(sum(row["hard_closed"] for row in module_rows)),
        "module_summary_csv": module_csv,
        "top_changed_neurons_csv": top_csv,
    }
    write_json(os.path.join(mask_dir, f"{tag}_mask_summary.json"), summary)

    plt.figure(figsize=(7, 4.5))
    plt.hist(closure.numpy(), bins=80)
    plt.xlabel("closure = 1 - soft_mask")
    plt.ylabel("neuron count")
    plt.title(f"MLP soft closure distribution ({tag})")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(mask_dir, f"{tag}_closure_hist.png"), dpi=200)
    plt.close()

    layers = sorted({layer for layer, _ in heatmap.keys() if layer >= 0})
    projs = ["gate_proj", "up_proj", "down_proj"]
    if layers:
        mat = np.zeros((len(layers), len(projs)), dtype=np.float32)
        for i, layer in enumerate(layers):
            for j, proj in enumerate(projs):
                mat[i, j] = heatmap.get((layer, proj), 0.0)
        plt.figure(figsize=(6, max(4, len(layers) * 0.22)))
        plt.imshow(mat, aspect="auto", cmap="magma")
        plt.colorbar(label="mean closure")
        plt.xticks(range(len(projs)), projs, rotation=20)
        plt.yticks(range(len(layers)), layers)
        plt.xlabel("MLP projection")
        plt.ylabel("layer")
        plt.title(f"Mean closure by layer/proj ({tag})")
        plt.tight_layout()
        plt.savefig(os.path.join(mask_dir, f"{tag}_layer_proj_closure_heatmap.png"), dpi=200)
        plt.close()

    print("[Mask Visualization]")
    print(f"  tag={tag}")
    print(f"  changed closure>0.01: {summary['changed_gt_001']}/{summary['num_mlp_neurons']}")
    print(f"  changed closure>0.05: {summary['changed_gt_005']}/{summary['num_mlp_neurons']}")
    print(f"  changed closure>0.10: {summary['changed_gt_010']}/{summary['num_mlp_neurons']}")
    print(f"  max closure={summary['closure_max']:.6f}, mean closure={summary['closure_mean']:.6f}")
    print(f"  files under: {mask_dir}")
    return summary


def save_training_summary_csv(output_dir: str, step_logs: List[Dict[str, Any]], epoch_logs: List[Dict[str, Any]]):
    if step_logs:
        keys = list(step_logs[0].keys())
        with open(os.path.join(output_dir, "train_step_logs.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader(); writer.writerows(step_logs)
    if epoch_logs:
        keys = list(epoch_logs[0].keys())
        with open(os.path.join(output_dir, "train_epoch_logs.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader(); writer.writerows(epoch_logs)


def plot_single_curve(output_dir: str, epochs: List[int], values: List[float], title: str, ylabel: str, fname: str):
    if not epochs or not values:
        return
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, values, marker="o")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=200)
    plt.close()


def plot_curves(output_dir: str, epoch_logs: List[Dict[str, Any]]):
    if not epoch_logs:
        return
    epochs = [x["epoch"] for x in epoch_logs]
    plot_single_curve(output_dir, epochs, [x["train_total_loss"] for x in epoch_logs], "Train total loss", "loss", "curve_train_total_loss.png")
    plot_single_curve(output_dir, epochs, [x["train_rl_loss"] for x in epoch_logs], "Train RL loss", "loss", "curve_train_rl_loss.png")
    plot_single_curve(output_dir, epochs, [x["train_sparse_mlp"] for x in epoch_logs], "Train MLP sparsity loss", "loss", "curve_train_sparse_mlp_loss.png")
    plot_single_curve(output_dir, epochs, [x["train_anchor_loss"] for x in epoch_logs], "Train anchor loss", "loss", "curve_train_anchor_loss.png")
    plot_single_curve(output_dir, epochs, [x["train_clean_kl"] for x in epoch_logs], "Train clean KL", "loss", "curve_train_clean_kl.png")
    plot_single_curve(output_dir, epochs, [x["train_clean_nll"] for x in epoch_logs], "Train clean NLL", "loss", "curve_train_clean_nll.png")
    plot_single_curve(output_dir, epochs, [x["train_sensitivity_loss"] for x in epoch_logs], "Train sensitivity loss", "loss", "curve_train_sensitivity_loss.png")
    plot_single_curve(output_dir, epochs, [x["train_mean_reward"] for x in epoch_logs], "Train mean reward", "reward", "curve_train_mean_reward.png")
    plot_single_curve(output_dir, epochs, [x["train_mean_answer_reward"] for x in epoch_logs], "Train answer reward", "reward", "curve_train_answer_reward.png")
    plot_single_curve(output_dir, epochs, [x["train_mean_reasoning_reward"] for x in epoch_logs], "Train reasoning reward", "reward", "curve_train_reasoning_reward.png")
    plot_single_curve(output_dir, epochs, [x["eval_mean_reward"] for x in epoch_logs], "Eval mean reward", "reward", "curve_eval_mean_reward.png")
    plot_single_curve(output_dir, epochs, [x["eval_answer_reward_mean"] for x in epoch_logs], "Eval answer reward", "reward", "curve_eval_answer_reward.png")
    plot_single_curve(output_dir, epochs, [x["eval_reasoning_reward_mean"] for x in epoch_logs], "Eval reasoning reward", "reward", "curve_eval_reasoning_reward.png")
    plot_single_curve(output_dir, epochs, [x["repair_generation_acc"] for x in epoch_logs], "SFT-contract repair accuracy", "accuracy", "curve_repair_generation_acc.png")
    plot_single_curve(output_dir, epochs, [x["clean_eval_acc"] for x in epoch_logs], "Clean eval accuracy", "accuracy", "curve_clean_eval_acc.png")
    plot_single_curve(output_dir, epochs, [x["mlp_keep_ratio"] for x in epoch_logs], "MLP keep ratio", "ratio", "curve_mlp_keep_ratio.png")
    plot_single_curve(output_dir, epochs, [x["mlp_mask_mean"] for x in epoch_logs], "MLP mean soft mask", "mask", "curve_mlp_mask_mean.png")
    plot_single_curve(output_dir, epochs, [x["mlp_mask_min"] for x in epoch_logs], "MLP min soft mask", "mask", "curve_mlp_mask_min.png")
    plot_single_curve(output_dir, epochs, [x["mlp_closure_mean"] for x in epoch_logs], "MLP mean closure", "closure", "curve_mlp_closure_mean.png")
    plot_single_curve(output_dir, epochs, [x["mlp_changed_gt_001"] for x in epoch_logs], "MLP neurons closure > 0.01", "count", "curve_mlp_changed_gt_001.png")
    plot_single_curve(output_dir, epochs, [x["mlp_changed_gt_005"] for x in epoch_logs], "MLP neurons closure > 0.05", "count", "curve_mlp_changed_gt_005.png")
    plot_single_curve(output_dir, epochs, [x["mlp_changed_gt_010"] for x in epoch_logs], "MLP neurons closure > 0.10", "count", "curve_mlp_changed_gt_010.png")


@torch.no_grad()
def evaluate_groups(model, tokenizer, judge, eval_items: List[RLItem], group_size: int,
                    max_prompt_len: int, max_new_tokens: int, temperature: float, top_p: float,
                    answer_reward_weight: float = 1.0, reasoning_reward_weight: float = 1.0):
    AblatedLinear.enabled = True
    AblatedLinear.enabled = True
    groups = sample_group_rollouts(model=model, tokenizer=tokenizer, items=eval_items, group_size=group_size,
                                   max_prompt_len=max_prompt_len, max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_p=top_p)
    assign_rewards_to_groups(groups, judge, answer_reward_weight, reasoning_reward_weight)
    all_rewards, all_ans, all_reason = [], [], []
    for group in groups:
        for rollout in group["rollouts"]:
            all_rewards.append(rollout["reward"])
            all_ans.append(rollout["answer_reward"])
            all_reason.append(rollout["reasoning_reward"])
    metrics = {
        "eval_mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "eval_answer_reward_mean": float(np.mean(all_ans)) if all_ans else 0.0,
        "eval_reasoning_reward_mean": float(np.mean(all_reason)) if all_reason else 0.0,
    }
    return metrics, groups


@torch.no_grad()
def evaluate_repair_accuracy(model, tokenizer, repair_items: List[RLItem],
                             max_prompt_len: int, max_new_tokens: int,
                             batch_size: int = 4):

    if not repair_items:
        return {
            "repair_generation_acc": 0.0,
            "repair_no_final_answer_rate": 0.0,
            "repair_generation_total": 0,
        }, []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    total, correct, no_final = 0, 0, 0
    rows = []
    for start in range(0, len(repair_items), batch_size):
        batch = repair_items[start:start + batch_size]
        prompts = [build_chat_prompt(item.question, tokenizer) for item in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=False,
        ).to(model.device)
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=get_qwen_eos_ids(tokenizer),
        )
        prompt_len = enc["input_ids"].shape[1]
        for index, item in enumerate(batch):
            raw = tokenizer.decode(out[index][prompt_len:], skip_special_tokens=True)
            clean = clean_generation_for_scoring(raw)
            pred = normalize_answer_str(extract_final_answer(clean))
            target = normalize_answer_str(item.ground_truth)
            is_correct = int(pred != "" and pred == target)
            total += 1
            correct += is_correct
            no_final += int(pred == "")
            rows.append({
                "qid": item.qid,
                "question": item.question,
                "ground_truth": item.ground_truth,
                "prediction": pred,
                "is_correct": is_correct,
                "generation_text": clean,
            })
    tokenizer.padding_side = old_padding_side
    return {
        "repair_generation_acc": correct / max(total, 1),
        "repair_no_final_answer_rate": no_final / max(total, 1),
        "repair_generation_total": total,
    }, rows


@torch.no_grad()
def evaluate_clean_accuracy(model, tokenizer, clean_items: List[CleanItem],
                            max_prompt_len: int, max_new_tokens: int,
                            batch_size: int = 4):
    if not clean_items:
        return {"clean_eval_acc": 0.0, "clean_eval_total": 0}
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    total, correct = 0, 0
    for start in range(0, len(clean_items), batch_size):
        batch = clean_items[start:start + batch_size]
        prompts = [build_chat_prompt(x.question, tokenizer) for x in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_prompt_len, add_special_tokens=False).to(model.device)
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=get_qwen_eos_ids(tokenizer),
        )
        prompt_len = enc["input_ids"].shape[1]
        for i, item in enumerate(batch):
            gen_ids = out[i][prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = normalize_answer_str(extract_final_answer(text))
            gt = normalize_answer_str(item.ground_truth)
            if gt:
                total += 1
                correct += int(pred == gt)
    tokenizer.padding_side = old_padding_side
    return {
        "clean_eval_acc": correct / max(total, 1),
        "clean_eval_total": total,
    }


def make_json_serializable_groups(groups):
    cleaned = []
    for group in groups:
        new_group = {"qid": group["qid"], "question": group["question"], "ground_truth": group["ground_truth"], "rollouts": []}
        for rollout in group["rollouts"]:
            new_group["rollouts"].append({
                "prompt_text": rollout.get("prompt_text", ""),
                "generation_text": rollout.get("generation_text", ""),
                "generation_text_for_reward": rollout.get("generation_text_for_reward", ""),
                "pred_answer_for_reward": rollout.get("pred_answer_for_reward", ""),
                "answer_reward": float(rollout.get("answer_reward", 0.0)),
                "reasoning_reward": float(rollout.get("reasoning_reward", 0.0)),
                "reward": float(rollout.get("reward", 0.0)),
                "judge_raw": rollout.get("judge_raw", ""),
            })
        cleaned.append(new_group)
    return cleaned


def parse_args():
    parser = argparse.ArgumentParser(description="Refine an MLP-neuron circuit using STE masks.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--judge-model-path", required=True)
    parser.add_argument("--rl-data-path", required=True)
    parser.add_argument("--clean-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-mask-path")
    parser.add_argument("--init-binary-mask-path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    MODEL_PATH = args.model_path
    JUDGE_MODEL_PATH = args.judge_model_path
    RL_DATA_PATH = args.rl_data_path
    CLEAN_DATA_PATH = args.clean_data_path
    OUTPUT_DIR = args.output_dir
    INIT_MASK_PATH = args.init_mask_path
    INIT_BINARY_MASK_PATH = args.init_binary_mask_path
    set_random_seed(args.seed)
    ensure_dir(OUTPUT_DIR)

    MAX_PROMPT_LEN = 512
    MAX_TOTAL_LEN_FOR_SCORING = 768
    MAX_NEW_TOKENS = 256
    TRAIN_SAMPLES = 100
    EVAL_SAMPLES = 50
    TRAIN_CLEAN_SAMPLES = 100
    EVAL_CLEAN_SAMPLES = 50
    CLEAN_BATCH_SIZE = 1
    GROUP_SIZE = 4
    EVAL_GROUP_SIZE = 1
    JUDGE_BATCH_SIZE = 1
    REPAIR_GRAD_ACCUM_STEPS = 8
    NUM_EPOCHS = 20
    LR = 2e-4
    TEMPERATURE = 0.7
    TOP_P = 0.9

    SPARSITY_WEIGHT_MLP_WARMUP = 0.0
    SPARSITY_WEIGHT_MLP_COMPRESS = 0.01
    SPARSITY_WARMUP_STAGES = 2
    ANCHOR_WEIGHT = 0.03
    ANCHOR_MODE = "init"  
    CLEAN_KL_WEIGHT = 0.0
    CLEAN_NLL_WEIGHT = 1.0
    CLEAN_KL_TEMPERATURE = 1.0
    USE_CLEAN_GRAD_PROJECTION = True
    CLEAN_PROJECTION_STRENGTH_SEARCH = 0.20
    CLEAN_PROJECTION_STRENGTH_AFTER_TARGET = 1.0
    CLEAN_ACC_TOLERANCE = 0.0
    CLEAN_GRAD_WEIGHT_WHEN_OK = 0.05
    CLEAN_GRAD_WEIGHT_RECOVERY = 0.25
    CLEAN_EVAL_BATCH_SIZE = 1
    REPAIR_REPLAY_NLL_WEIGHT = 0.10
    UNIFORM_REWARD_REPLAY_NLL_WEIGHT = 0.50
    MASK_INIT_VALUE = 0.2
    MASK_TEMPERATURE = 1.0
    SFT_INIT_HARD_THRESHOLD = 0.95
    SFT_INIT_MAX_DELETE_RATIO = 0.03


    SEARCH_HARD_THRESHOLD = min(SFT_INIT_HARD_THRESHOLD + 0.01, 0.99)
    DELETE_RATIO_SCHEDULE = [0.030, 0.020, 0.015, 0.010, 0.008, 0.005]
    MAX_DELETE_RATIO = DELETE_RATIO_SCHEDULE[-1]


    MIN_INIT_REPAIR_ACC = 0.0
    REPAIR_TARGET_BEFORE_COMPRESSION = 0.90
    REPAIR_ACC_TOLERANCE_AFTER_TARGET = 0.05
    REPAIR_ACC_TOLERANCE_DURING_SEARCH = 0.10
    SEARCH_CLEAN_TEMP_TOLERANCE = 0.06
    SEARCH_MIN_REPAIR_IMPROVEMENT = 0.02
    USE_SENSITIVITY_REGULARIZER = False
    SENSITIVITY_PROBE_OFFSET = 0.10
    SENSITIVITY_EMA_BETA = 0.95
    REPAIR_CLOSE_WEIGHT = 0.03
    CLEAN_OPEN_WEIGHT = 0.35
    CLOSURE_L2_WEIGHT = 0.001
    MAX_SOFT_CLOSURE = 0.95
    MAX_CLOSURE_WEIGHT = 0.0

    ANSWER_REWARD_WEIGHT = 1.0
    REASONING_REWARD_WEIGHT = 1.0
    USE_GRADED_JUDGE = True

    print("=" * 100)
    print("RL mask refinement v6: hard binary MLP masks + clean preservation + sparse circuit")
    print("=" * 100)
    print(f"CLEAN_DATA_PATH              = {CLEAN_DATA_PATH}")
    print(f"INIT_MASK_PATH              = {INIT_MASK_PATH}")
    print(f"INIT_BINARY_MASK_PATH       = {INIT_BINARY_MASK_PATH}")
    print(f"CLEAN_KL_WEIGHT             = {CLEAN_KL_WEIGHT}")
    print(f"CLEAN_NLL_WEIGHT            = {CLEAN_NLL_WEIGHT}")
    print(f"USE_CLEAN_GRAD_PROJECTION   = {USE_CLEAN_GRAD_PROJECTION}")
    print(f"CLEAN_PROJECTION_SEARCH     = {CLEAN_PROJECTION_STRENGTH_SEARCH}")
    print(f"CLEAN_PROJECTION_AFTER      = {CLEAN_PROJECTION_STRENGTH_AFTER_TARGET}")
    print(f"CLEAN_ACC_TOLERANCE         = {CLEAN_ACC_TOLERANCE}")
    print(f"CLEAN_GRAD_WEIGHT_OK        = {CLEAN_GRAD_WEIGHT_WHEN_OK}")
    print(f"CLEAN_GRAD_WEIGHT_RECOVERY  = {CLEAN_GRAD_WEIGHT_RECOVERY}")
    print(f"REPAIR_REPLAY_WEIGHT        = {REPAIR_REPLAY_NLL_WEIGHT}")
    print(f"UNIFORM_REPLAY_WEIGHT       = {UNIFORM_REWARD_REPLAY_NLL_WEIGHT}")
    print(f"TRAIN_GROUP_SIZE            = {GROUP_SIZE}")
    print(f"EVAL_GROUP_SIZE             = {EVAL_GROUP_SIZE}")
    print(f"JUDGE_BATCH_SIZE            = {JUDGE_BATCH_SIZE}")
    print(f"REPAIR_GRAD_ACCUM_STEPS     = {REPAIR_GRAD_ACCUM_STEPS}")
    print(f"SPARSITY_WEIGHT_WARMUP      = {SPARSITY_WEIGHT_MLP_WARMUP}")
    print(f"SPARSITY_WEIGHT_COMPRESS    = {SPARSITY_WEIGHT_MLP_COMPRESS}")
    print("MASK_MODE                   = STE (fixed)")
    print(f"MASK_INIT_VALUE             = {MASK_INIT_VALUE}")
    print(f"MASK_TEMPERATURE            = {MASK_TEMPERATURE}")
    print(f"SFT_INIT_HARD_THRESHOLD     = {SFT_INIT_HARD_THRESHOLD}")
    print(f"SEARCH_HARD_THRESHOLD       = {SEARCH_HARD_THRESHOLD}")
    print(f"DELETE_RATIO_SCHEDULE       = {DELETE_RATIO_SCHEDULE}")
    print(f"FINAL_MAX_DELETE_RATIO      = {MAX_DELETE_RATIO}")
    print(f"REPAIR_TARGET_TO_COMPRESS   = {REPAIR_TARGET_BEFORE_COMPRESSION}")
    print(f"REPAIR_TOLERANCE_AFTER      = {REPAIR_ACC_TOLERANCE_AFTER_TARGET}")
    print(f"REPAIR_TOLERANCE_SEARCH     = {REPAIR_ACC_TOLERANCE_DURING_SEARCH}")
    print(f"SEARCH_CLEAN_TEMP_TOLERANCE = {SEARCH_CLEAN_TEMP_TOLERANCE}")
    print(f"SEARCH_MIN_REPAIR_GAIN      = {SEARCH_MIN_REPAIR_IMPROVEMENT}")
    print(f"USE_SENSITIVITY_REG         = {USE_SENSITIVITY_REGULARIZER}")
    print(f"SENSITIVITY_PROBE           = {SENSITIVITY_PROBE_OFFSET}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    disable_generation_max_length_warning(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    mlp_mask_params = patch_model(model)
    all_mask_params = mlp_mask_params
    current_delete_ratio = SFT_INIT_MAX_DELETE_RATIO
    configure_ste(
        MASK_TEMPERATURE, MASK_INIT_VALUE,
        SFT_INIT_HARD_THRESHOLD, current_delete_ratio,
    )

    if INIT_MASK_PATH and os.path.exists(INIT_MASK_PATH):
        state = torch.load(INIT_MASK_PATH, map_location="cpu")
        n2 = AblatedLinear.load_masks(model, state)
        print(f"[Init] loaded {n2} neuron-mask modules from {INIT_MASK_PATH}")
        if n2 == 0:
            raise ValueError(
                "No MLP logits were loaded. Use a logits mask, not a binary mask."
            )
        if INIT_BINARY_MASK_PATH and os.path.exists(INIT_BINARY_MASK_PATH):
            expected_binary_state = torch.load(
                INIT_BINARY_MASK_PATH, map_location="cpu"
            )
            mismatch_count, compared_count, compared_modules = (
                compare_current_binary_mask(model, expected_binary_state)
            )
            print(
                "[Init mask parity] "
                f"mismatched={mismatch_count}/{compared_count}, "
                f"mlp_modules={compared_modules}"
            )
            if compared_modules == 0:
                raise ValueError(
                    "INIT_BINARY_MASK_PATH contains no matching MLP tensors."
                )
            if mismatch_count != 0:
                raise RuntimeError(
                    "Loaded logits do not reconstruct the companion SFT binary "
                    "mask. Check the mask pair, threshold, and delete ratio."
                )
        else:
            print("[Init mask parity] companion binary mask not found; skipped.")
    else:
        print("[Init] no previous mask loaded, training from default all-keep init.")

    anchor_refs = get_anchor_refs(all_mask_params, anchor_mode=ANCHOR_MODE)
    optimizer = torch.optim.AdamW(all_mask_params, lr=LR)
    repair_close_sens_ema = [torch.zeros_like(p, dtype=torch.float32) for p in all_mask_params]
    clean_open_sens_ema = [torch.zeros_like(p, dtype=torch.float32) for p in all_mask_params]

    judge = ReasoningJudge(
        model_path=JUDGE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_graded_reward=USE_GRADED_JUDGE,
        batch_size=JUDGE_BATCH_SIZE,
    )

    raw_data = load_json_auto(RL_DATA_PATH)
    rl_items = build_rl_items(raw_data)
    train_items = rl_items[:TRAIN_SAMPLES]
    eval_items = rl_items[TRAIN_SAMPLES:TRAIN_SAMPLES + EVAL_SAMPLES]
    if not train_items or not eval_items:
        raise ValueError("Train or eval split is empty.")

    clean_raw_data = load_json_auto(CLEAN_DATA_PATH) if CLEAN_DATA_PATH else []
    clean_items_all = build_clean_items(clean_raw_data)
    train_clean_items = clean_items_all[:TRAIN_CLEAN_SAMPLES]
    eval_clean_items = clean_items_all[TRAIN_CLEAN_SAMPLES:TRAIN_CLEAN_SAMPLES + EVAL_CLEAN_SAMPLES]
    print("\n[Clean Data]")
    print(f"  total clean items = {len(clean_items_all)}")
    print(f"  train clean items = {len(train_clean_items)}")
    print(f"  eval clean items  = {len(eval_clean_items)}")

    baseline_clean_acc = 0.0
    if eval_clean_items:
        AblatedLinear.enabled = False
        AblatedLinear.enabled = False
        baseline_clean_metrics = evaluate_clean_accuracy(
            model=model, tokenizer=tokenizer, clean_items=eval_clean_items,
            max_prompt_len=MAX_PROMPT_LEN, max_new_tokens=MAX_NEW_TOKENS,
            batch_size=CLEAN_EVAL_BATCH_SIZE,
        )
        baseline_clean_acc = baseline_clean_metrics["clean_eval_acc"]
        AblatedLinear.enabled = True
        AblatedLinear.enabled = True
        print(f"  clean baseline acc without masks = {baseline_clean_acc:.6f}")

    step_logs, epoch_logs = [], []
    step_log_path = os.path.join(OUTPUT_DIR, "train_step_logs.jsonl")
    best_repair_generation_acc = None
    best_epoch = None
    best_max_delete_ratio = None
    best_clean_acc = None
    best_delete_ratio = None
    best_mask_path = os.path.join(OUTPUT_DIR, "best_hard_logits_masks.pt")
    best_binary_mask_path = os.path.join(OUTPUT_DIR, "best_hard_binary_masks.pt")
    accepted_mask_path = os.path.join(
        OUTPUT_DIR, "last_repair_safe_hard_logits_masks.pt"
    )
    accepted_binary_mask_path = os.path.join(
        OUTPUT_DIR, "last_repair_safe_hard_binary_masks.pt"
    )

    init_repair_metrics, init_repair_rows = evaluate_repair_accuracy(
        model=model,
        tokenizer=tokenizer,
        repair_items=eval_items,
        max_prompt_len=MAX_PROMPT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=CLEAN_EVAL_BATCH_SIZE,
    )
    init_clean_metrics = evaluate_clean_accuracy(
        model=model,
        tokenizer=tokenizer,
        clean_items=eval_clean_items,
        max_prompt_len=MAX_PROMPT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=CLEAN_EVAL_BATCH_SIZE,
    )
    init_stats = compute_mask_stats(model)
    write_json(os.path.join(OUTPUT_DIR, "init_repair_generations.json"), init_repair_rows)
    write_json(os.path.join(OUTPUT_DIR, "init_eval_metrics.json"), {
        **init_repair_metrics,
        **init_clean_metrics,
        "clean_baseline_acc_without_masks": baseline_clean_acc,
        "mlp_delete_ratio": init_stats["mlp_delete_ratio"],
        "sft_init_hard_threshold": SFT_INIT_HARD_THRESHOLD,
        "sft_init_max_delete_ratio": SFT_INIT_MAX_DELETE_RATIO,
    })
    print("\n[Init Evaluation - SFT contract]")
    print(f"  repair_generation_acc = {init_repair_metrics['repair_generation_acc']:.6f}")
    print(f"  clean_eval_acc              = {init_clean_metrics['clean_eval_acc']:.6f}")
    print(
        "  SFT-mask deleted neurons   = "
        f"{init_stats['mlp_deleted']}/{init_stats['mlp_total']} "
        f"({init_stats['mlp_delete_ratio']:.4%})"
    )
    if init_repair_metrics["repair_generation_acc"] < MIN_INIT_REPAIR_ACC:
        raise RuntimeError(
            "Loaded SFT mask does not reproduce the expected repair result: "
            f"{init_repair_metrics['repair_generation_acc']:.6f} "
            f"< {MIN_INIT_REPAIR_ACC:.6f}. Do not start RL."
        )

    init_clean_ok = (
        not eval_clean_items
        or init_clean_metrics["clean_eval_acc"] >= baseline_clean_acc - CLEAN_ACC_TOLERANCE
    )
    best_repair_generation_acc = init_repair_metrics["repair_generation_acc"]
    best_epoch = 0
    best_max_delete_ratio = current_delete_ratio
    best_hard_threshold = SFT_INIT_HARD_THRESHOLD
    best_clean_acc = init_clean_metrics["clean_eval_acc"]
    best_delete_ratio = init_stats["mlp_delete_ratio"]
    accepted_budget = SFT_INIT_MAX_DELETE_RATIO
    accepted_hard_threshold = SFT_INIT_HARD_THRESHOLD
    accepted_repair_acc = init_repair_metrics["repair_generation_acc"]
    accepted_clean_acc = init_clean_metrics["clean_eval_acc"]
    peak_repair_acc = init_repair_metrics["repair_generation_acc"]
    repair_target_reached = (
        peak_repair_acc >= REPAIR_TARGET_BEFORE_COMPRESSION
    )
    best_strict_available = init_clean_ok
    budget_stage = 0
    last_clean_acc = init_clean_metrics["clean_eval_acc"]
    save_all_masks(model, best_mask_path)
    save_binary_masks(model, best_binary_mask_path)
    save_all_masks(model, accepted_mask_path)
    save_binary_masks(model, accepted_binary_mask_path)
    if init_clean_ok:
        print("  [Best] epoch 0 satisfies repair and clean gates")
    else:
        print(
            "  [Recovery] epoch 0 repair is safe, but clean is below its floor; "
            "stay at the SFT budget until clean recovers."
        )

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        current_delete_ratio = DELETE_RATIO_SCHEDULE[
            min(budget_stage, len(DELETE_RATIO_SCHEDULE) - 1)
        ]
        current_sparsity_weight_mlp = (
            SPARSITY_WEIGHT_MLP_WARMUP
            if budget_stage < SPARSITY_WARMUP_STAGES
            else SPARSITY_WEIGHT_MLP_COMPRESS
        )
        clean_floor = baseline_clean_acc - CLEAN_ACC_TOLERANCE
        search_clean_floor = max(
            0.0, baseline_clean_acc - SEARCH_CLEAN_TEMP_TOLERANCE
        )
        clean_grad_weight = (
            CLEAN_GRAD_WEIGHT_WHEN_OK
            if last_clean_acc >= clean_floor
            else CLEAN_GRAD_WEIGHT_RECOVERY
        )
        current_hard_threshold = SEARCH_HARD_THRESHOLD
        clean_projection_strength = (
            CLEAN_PROJECTION_STRENGTH_AFTER_TARGET
            if repair_target_reached
            else CLEAN_PROJECTION_STRENGTH_SEARCH
        )
        configure_ste(
            MASK_TEMPERATURE,
            MASK_INIT_VALUE,
            current_hard_threshold,
            current_delete_ratio,
        )
        print(
            f"\n[Delete Budget] epoch {epoch}: "
            f"max_delete_ratio={current_delete_ratio:.6f}, "
            f"hard_threshold={current_hard_threshold:.4f}, "
            f"clean_grad_weight={clean_grad_weight:.3f}, "
            f"clean_projection_strength={clean_projection_strength:.2f}, "
            f"sparsity_weight_mlp={current_sparsity_weight_mlp:.4f}"
        )
        random.shuffle(train_items)
        epoch_total_loss = []
        epoch_rl_loss = []
        epoch_sparse_mlp = []
        epoch_anchor_loss = []
        epoch_clean_kl = []
        epoch_clean_nll = []
        epoch_clean_grad_coef = []
        epoch_sensitivity_loss = []
        epoch_mean_reward = []
        epoch_mean_answer = []
        epoch_mean_reason = []
        epoch_repair_replay_nll = []
        epoch_repair_replay_weight = []

        for start in range(0, len(train_items), REPAIR_GRAD_ACCUM_STEPS):
            accumulation_items = train_items[
                start:start + REPAIR_GRAD_ACCUM_STEPS
            ]
            if not accumulation_items:
                continue
            global_step += 1
            accumulation_count = len(accumulation_items)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            micro_values = {
                "repair": [], "rl": [], "sparse_mlp": [],
                "anchor": [], "sensitivity": [], "replay_nll": [],
                "replay_weight": [], "reward_std": [], "reward": [],
                "answer": [], "reasoning": [],
            }

            for item in accumulation_items:
                AblatedLinear.enabled = True
                AblatedLinear.enabled = True
                groups = sample_group_rollouts(
                    model=model,
                    tokenizer=tokenizer,
                    items=[item],
                    group_size=GROUP_SIZE,
                    max_prompt_len=MAX_PROMPT_LEN,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
                assign_rewards_to_groups(
                    groups=groups,
                    judge=judge,
                    answer_reward_weight=ANSWER_REWARD_WEIGHT,
                    reasoning_reward_weight=REASONING_REWARD_WEIGHT,
                )
                flat_rollouts = groups[0]["rollouts"]
                item_rewards = [row["reward"] for row in flat_rollouts]
                item_answers = [row["answer_reward"] for row in flat_rollouts]
                item_reasoning = [row["reasoning_reward"] for row in flat_rollouts]

                input_ids, attention_mask, labels, advantages, _ = (
                    build_scoring_batch_from_rollouts(
                        groups=groups,
                        pad_token_id=tokenizer.pad_token_id,
                        max_total_len=MAX_TOTAL_LEN_FOR_SCORING,
                    )
                )
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
                labels = labels.to(model.device)
                seq_mean_logprob = compute_sequence_mean_logprob(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                adv_tensor = torch.tensor(
                    advantages,
                    device=model.device,
                    dtype=seq_mean_logprob.dtype,
                )
                rl_loss = -(adv_tensor.detach() * seq_mean_logprob).mean()
                sparse_mlp = compute_sparsity_loss(
                    neuron_mask_params=mlp_mask_params,
                    device=model.device,
                )
                anchor_loss = compute_anchor_loss(
                    all_mask_params, anchor_refs, anchor_mode=ANCHOR_MODE
                )
                if USE_SENSITIVITY_REGULARIZER:
                    compute_probe_sensitivities_rl(
                        model=model,
                        tokenizer=tokenizer,
                        mask_params=all_mask_params,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        advantages=advantages,
                        clean_items=train_clean_items,
                        clean_batch_size=CLEAN_BATCH_SIZE,
                        max_total_len=MAX_TOTAL_LEN_FOR_SCORING,
                        kl_temperature=CLEAN_KL_TEMPERATURE,
                        probe_offset=SENSITIVITY_PROBE_OFFSET,
                        repair_ema=repair_close_sens_ema,
                        clean_ema=clean_open_sens_ema,
                        ema_beta=SENSITIVITY_EMA_BETA,
                        clean_use_kl=CLEAN_KL_WEIGHT > 0,
                    )
                sensitivity_loss = compute_sensitivity_regularizer(
                    mask_params=all_mask_params,
                    repair_ema=repair_close_sens_ema,
                    clean_ema=clean_open_sens_ema,
                    repair_close_weight=REPAIR_CLOSE_WEIGHT,
                    clean_open_weight=CLEAN_OPEN_WEIGHT,
                    closure_l2_weight=CLOSURE_L2_WEIGHT,
                    max_soft_closure=MAX_SOFT_CLOSURE,
                    max_closure_weight=MAX_CLOSURE_WEIGHT,
                    device=model.device,
                ) if USE_SENSITIVITY_REGULARIZER else torch.tensor(
                    0.0, device=model.device
                )
                repair_core_loss = (
                    rl_loss
                    + current_sparsity_weight_mlp * sparse_mlp
                    + ANCHOR_WEIGHT * anchor_loss
                    + sensitivity_loss
                )
                reward_std = float(np.std(item_rewards)) if item_rewards else 0.0
                replay_weight = (
                    UNIFORM_REWARD_REPLAY_NLL_WEIGHT
                    if reward_std <= 1e-6
                    else REPAIR_REPLAY_NLL_WEIGHT
                )

                (repair_core_loss / accumulation_count).backward()
                repair_core_value = float(repair_core_loss.detach().item())
                del input_ids, attention_mask, labels, seq_mean_logprob
                del repair_core_loss

                repair_replay_nll = compute_repair_replay_nll(
                    model=model,
                    tokenizer=tokenizer,
                    items=[item],
                    max_total_len=MAX_TOTAL_LEN_FOR_SCORING,
                )
                if repair_replay_nll.requires_grad and replay_weight != 0.0:
                    (
                        replay_weight
                        * repair_replay_nll
                        / accumulation_count
                    ).backward()
                replay_nll_value = float(repair_replay_nll.detach().item())
                micro_values["repair"].append(
                    repair_core_value + replay_weight * replay_nll_value
                )
                micro_values["rl"].append(float(rl_loss.detach().item()))
                micro_values["sparse_mlp"].append(
                    float(sparse_mlp.detach().item())
                )
                micro_values["anchor"].append(
                    float(anchor_loss.detach().item())
                )
                micro_values["sensitivity"].append(
                    float(sensitivity_loss.detach().item())
                )
                micro_values["replay_nll"].append(replay_nll_value)
                micro_values["replay_weight"].append(replay_weight)
                micro_values["reward_std"].append(reward_std)
                micro_values["reward"].append(
                    float(np.mean(item_rewards)) if item_rewards else 0.0
                )
                micro_values["answer"].append(
                    float(np.mean(item_answers)) if item_answers else 0.0
                )
                micro_values["reasoning"].append(
                    float(np.mean(item_reasoning)) if item_reasoning else 0.0
                )
                del repair_replay_nll, groups, flat_rollouts

            repair_grads = clone_param_grads(all_mask_params)
            zero_param_grads(all_mask_params)
            clean_kl, clean_nll = compute_clean_preserve_loss(
                model=model,
                tokenizer=tokenizer,
                clean_items=train_clean_items,
                batch_size=CLEAN_BATCH_SIZE,
                max_total_len=MAX_TOTAL_LEN_FOR_SCORING,
                kl_temperature=CLEAN_KL_TEMPERATURE,
                use_kl=CLEAN_KL_WEIGHT > 0,
            )
            clean_loss = CLEAN_KL_WEIGHT * clean_kl + CLEAN_NLL_WEIGHT * clean_nll
            grad_projection_coef = 0.0
            if train_clean_items and (CLEAN_KL_WEIGHT > 0 or CLEAN_NLL_WEIGHT > 0):
                clean_loss.backward()
                clean_grads = clone_param_grads(all_mask_params)
                if USE_CLEAN_GRAD_PROJECTION:
                    repair_grads, grad_projection_coef = (
                        project_grads_against_clean(
                            repair_grads,
                            clean_grads,
                            strength=clean_projection_strength,
                        )
                    )
                set_param_grads(
                    all_mask_params,
                    repair_grads,
                    clean_grads,
                    clean_weight=clean_grad_weight,
                )
            else:
                set_param_grads(all_mask_params, repair_grads)

            total_loss_value = float(
                np.mean(micro_values["repair"])
                + clean_grad_weight * clean_loss.detach().item()
            )
            torch.nn.utils.clip_grad_norm_(all_mask_params, 1.0)
            optimizer.step()

            epoch_total_loss.append(total_loss_value)
            epoch_rl_loss.extend(micro_values["rl"])
            epoch_sparse_mlp.extend(micro_values["sparse_mlp"])
            epoch_anchor_loss.extend(micro_values["anchor"])
            epoch_clean_kl.append(float(clean_kl.detach().item()))
            epoch_clean_nll.append(float(clean_nll.detach().item()))
            epoch_clean_grad_coef.append(float(grad_projection_coef))
            epoch_sensitivity_loss.extend(micro_values["sensitivity"])
            epoch_mean_reward.extend(micro_values["reward"])
            epoch_mean_answer.extend(micro_values["answer"])
            epoch_mean_reason.extend(micro_values["reasoning"])
            epoch_repair_replay_nll.extend(micro_values["replay_nll"])
            epoch_repair_replay_weight.extend(micro_values["replay_weight"])

            step_row = {
                "global_step": global_step,
                "epoch": epoch,
                "optimizer_step_index": start // REPAIR_GRAD_ACCUM_STEPS,
                "repair_questions_accumulated": accumulation_count,
                "total_loss": total_loss_value,
                "rl_loss": float(np.mean(micro_values["rl"])),
                "sparse_mlp": float(np.mean(micro_values["sparse_mlp"])),
                "anchor_loss": float(np.mean(micro_values["anchor"])),
                "clean_kl": float(clean_kl.detach().item()),
                "clean_nll": float(clean_nll.detach().item()),
                "clean_grad_weight": clean_grad_weight,
                "clean_grad_projection_coef": float(grad_projection_coef),
                "clean_projection_strength": float(clean_projection_strength),
                "hard_threshold": float(current_hard_threshold),
                "sensitivity_loss": float(np.mean(micro_values["sensitivity"])),
                "repair_replay_nll": float(np.mean(micro_values["replay_nll"])),
                "repair_replay_weight": float(
                    np.mean(micro_values["replay_weight"])
                ),
                "reward_std": float(np.mean(micro_values["reward_std"])),
                "mean_reward": float(np.mean(micro_values["reward"])),
                "mean_answer_reward": float(np.mean(micro_values["answer"])),
                "mean_reasoning_reward": float(
                    np.mean(micro_values["reasoning"])
                ),
            }
            step_logs.append(step_row)
            append_jsonl(step_log_path, step_row)


        del clean_loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        metrics, eval_groups = evaluate_groups(
            model=model, tokenizer=tokenizer, judge=judge, eval_items=eval_items,
            group_size=EVAL_GROUP_SIZE, max_prompt_len=MAX_PROMPT_LEN,
            max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P,
            answer_reward_weight=ANSWER_REWARD_WEIGHT,
            reasoning_reward_weight=REASONING_REWARD_WEIGHT,
        )
        repair_generation_metrics, repair_generation_rows = evaluate_repair_accuracy(
            model=model,
            tokenizer=tokenizer,
            repair_items=eval_items,
            max_prompt_len=MAX_PROMPT_LEN,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=CLEAN_EVAL_BATCH_SIZE,
        )
        clean_eval_metrics = evaluate_clean_accuracy(
            model=model,
            tokenizer=tokenizer,
            clean_items=eval_clean_items,
            max_prompt_len=MAX_PROMPT_LEN,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=CLEAN_EVAL_BATCH_SIZE,
        )
        stats = compute_mask_stats(model)
        mask_viz_summary = save_mask_visualizations(model, OUTPUT_DIR, tag=f"epoch_{epoch}", top_k=200)
        epoch_mask_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch:02d}_hard_logits_masks.pt")
        epoch_binary_mask_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch:02d}_hard_binary_masks.pt")
        save_all_masks(model, epoch_mask_path)
        save_binary_masks(model, epoch_binary_mask_path)
        mlp_keep_ratio = stats["mlp_keep"] / max(stats["mlp_total"], 1)
        epoch_row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(epoch_total_loss)) if epoch_total_loss else 0.0,
            "train_rl_loss": float(np.mean(epoch_rl_loss)) if epoch_rl_loss else 0.0,
            "train_sparse_mlp": float(np.mean(epoch_sparse_mlp)) if epoch_sparse_mlp else 0.0,
            "train_anchor_loss": float(np.mean(epoch_anchor_loss)) if epoch_anchor_loss else 0.0,
            "train_clean_kl": float(np.mean(epoch_clean_kl)) if epoch_clean_kl else 0.0,
            "train_clean_nll": float(np.mean(epoch_clean_nll)) if epoch_clean_nll else 0.0,
            "train_clean_grad_projection_coef": float(np.mean(epoch_clean_grad_coef)) if epoch_clean_grad_coef else 0.0,
            "train_sensitivity_loss": float(np.mean(epoch_sensitivity_loss)) if epoch_sensitivity_loss else 0.0,
            "train_repair_replay_nll": float(np.mean(epoch_repair_replay_nll)) if epoch_repair_replay_nll else 0.0,
            "train_repair_replay_weight": float(np.mean(epoch_repair_replay_weight)) if epoch_repair_replay_weight else 0.0,
            "clean_grad_weight": float(clean_grad_weight),
            "sparsity_weight_mlp": float(current_sparsity_weight_mlp),
            "train_mean_reward": float(np.mean(epoch_mean_reward)) if epoch_mean_reward else 0.0,
            "train_mean_answer_reward": float(np.mean(epoch_mean_answer)) if epoch_mean_answer else 0.0,
            "train_mean_reasoning_reward": float(np.mean(epoch_mean_reason)) if epoch_mean_reason else 0.0,
            **metrics,
            **repair_generation_metrics,
            **clean_eval_metrics,
            "mlp_keep": int(stats["mlp_keep"]),
            "mlp_total": int(stats["mlp_total"]),
            "mlp_keep_ratio": float(mlp_keep_ratio),
            "mlp_mask_min": float(stats["mlp_mask_min"]),
            "mlp_mask_mean": float(stats["mlp_mask_mean"]),
            "mlp_closure_mean": float(stats["mlp_closure_mean"]),
            "mlp_changed_gt_001": int(stats["mlp_changed_001"]),
            "mlp_changed_gt_005": int(stats["mlp_changed_005"]),
            "mlp_changed_gt_010": int(stats["mlp_changed_010"]),
            "mlp_changed_gt_020": int(stats["mlp_changed_020"]),
            "mlp_closure_max": float(mask_viz_summary.get("closure_max", 0.0)),
            "mlp_deleted": int(stats["mlp_deleted"]),
            "mlp_delete_budget": int(stats["mlp_delete_budget"]),
            "mlp_delete_ratio": float(stats["mlp_delete_ratio"]),
            "max_delete_ratio": float(current_delete_ratio),
            "hard_threshold": float(current_hard_threshold),
            "clean_projection_strength": float(clean_projection_strength),
            "final_max_delete_ratio": float(MAX_DELETE_RATIO),
            "epoch_mask_path": epoch_mask_path,
            "epoch_binary_mask_path": epoch_binary_mask_path,
            "best_saved": False,
            "best_epoch": int(best_epoch or 0),
            "best_repair_generation_acc_so_far": float(best_repair_generation_acc) if best_repair_generation_acc is not None else 0.0,
            "best_max_delete_ratio": float(best_max_delete_ratio) if best_max_delete_ratio is not None else 0.0,
            "best_hard_threshold": float(best_hard_threshold),
            "best_clean_acc": float(best_clean_acc) if best_clean_acc is not None else 0.0,
            "best_delete_ratio": float(best_delete_ratio) if best_delete_ratio is not None else 0.0,
            "accepted_as_repair_safe": False,
            "budget_advanced": False,
            "rolled_back": False,
            "budget_stage": int(budget_stage),
            "repair_target_reached_before_epoch": bool(repair_target_reached),
            "peak_repair_acc_before_epoch": float(peak_repair_acc),
            "strict_clean_floor": float(clean_floor),
            "search_clean_floor": float(search_clean_floor),
        }
        write_json(
            os.path.join(OUTPUT_DIR, f"repair_generations_epoch_{epoch:02d}.json"),
            repair_generation_rows,
        )
        epoch_logs.append(epoch_row)
        print("\n" + "=" * 100)
        print(f"[Epoch {epoch}/{NUM_EPOCHS}]")
        for k in [
            "train_total_loss", "train_rl_loss", "train_sparse_mlp", "train_anchor_loss",
            "train_clean_kl", "train_clean_nll", "train_clean_grad_projection_coef", "train_sensitivity_loss",
            "train_repair_replay_nll", "train_repair_replay_weight",
            "train_mean_reward", "train_mean_answer_reward", "train_mean_reasoning_reward",
            "eval_mean_reward", "eval_answer_reward_mean", "eval_reasoning_reward_mean",
            "repair_generation_acc", "repair_no_final_answer_rate", "clean_eval_acc",
        ]:
            print(f"  {k:24s} = {epoch_row[k]:.6f}")
        print(f"  mlp_keep                 = {stats['mlp_keep']}/{stats['mlp_total']} ({mlp_keep_ratio*100:.2f}%)")
        print(f"  mlp_deleted              = {stats['mlp_deleted']}/{stats['mlp_total']} ({stats['mlp_delete_ratio']*100:.4f}%, budget={stats['mlp_delete_budget']})")
        print(f"  mlp_mask_mean            = {stats['mlp_mask_mean']:.6f}")
        print(f"  mlp_mask_min             = {stats['mlp_mask_min']:.6f}")
        print(f"  mlp_changed_gt_001       = {stats['mlp_changed_001']}/{stats['mlp_total']}")
        print(f"  mlp_changed_gt_005       = {stats['mlp_changed_005']}/{stats['mlp_total']}")
        print(f"  mlp_changed_gt_010       = {stats['mlp_changed_010']}/{stats['mlp_total']}")
        print(f"  epoch logits masks       = {epoch_mask_path}")
        print(f"  epoch binary masks       = {epoch_binary_mask_path}")
        current_repair_acc = repair_generation_metrics["repair_generation_acc"]
        clean_ok = (not eval_clean_items) or (epoch_row["clean_eval_acc"] >= clean_floor)
        delete_ok = epoch_row["mlp_delete_ratio"] <= current_delete_ratio + 1e-8
        repair_better = (
            not best_strict_available
            or best_repair_generation_acc is None
            or current_repair_acc > best_repair_generation_acc
            or (
                abs(current_repair_acc - best_repair_generation_acc) <= 1e-12
                and (
                    best_clean_acc is None
                    or best_clean_acc < clean_floor <= epoch_row["clean_eval_acc"]
                    or best_delete_ratio is None
                    or epoch_row["mlp_delete_ratio"] < best_delete_ratio
                )
            )
        )
        if clean_ok and delete_ok and repair_better:
            best_strict_available = True
            best_repair_generation_acc = current_repair_acc
            best_epoch = epoch
            best_max_delete_ratio = current_delete_ratio
            best_hard_threshold = current_hard_threshold
            best_clean_acc = epoch_row["clean_eval_acc"]
            best_delete_ratio = epoch_row["mlp_delete_ratio"]
            epoch_row.update({
                "best_saved": True,
                "best_epoch": int(best_epoch),
                "best_repair_generation_acc_so_far": float(best_repair_generation_acc),
                "best_max_delete_ratio": float(best_max_delete_ratio),
                "best_hard_threshold": float(best_hard_threshold),
                "best_clean_acc": float(best_clean_acc),
                "best_delete_ratio": float(best_delete_ratio),
            })
            save_all_masks(model, best_mask_path)
            save_binary_masks(model, best_binary_mask_path)
            print(
                "  [Best] saved: "
                f"epoch={epoch}, repair_generation_acc={current_repair_acc:.6f}, "
                f"clean_eval_acc={epoch_row['clean_eval_acc']:.6f}, "
                f"mlp_delete_ratio={epoch_row['mlp_delete_ratio']:.6f}, "
                f"max_delete_ratio={current_delete_ratio:.6f}, path={best_mask_path}"
            )
        elif not clean_ok:
            print(
                "  [Best] skipped: clean_eval_acc "
                f"{epoch_row['clean_eval_acc']:.6f} < floor {(baseline_clean_acc - CLEAN_ACC_TOLERANCE):.6f}"
            )
        elif not delete_ok:
            print(
                "  [Best] skipped: mlp_delete_ratio "
                f"{epoch_row['mlp_delete_ratio']:.6f} > max {current_delete_ratio:.6f}"
            )
        else:
            print(
                "  [Best] skipped: repair_generation_acc "
                f"{current_repair_acc:.6f} <= best {best_repair_generation_acc:.6f} "
                f"(best_epoch={best_epoch}, best_max_delete_ratio={best_max_delete_ratio:.6f})"
            )

        repair_floor = max(
            0.0,
            peak_repair_acc - (
                REPAIR_ACC_TOLERANCE_AFTER_TARGET
                if repair_target_reached
                else REPAIR_ACC_TOLERANCE_DURING_SEARCH
            ),
        )
        repair_safe = current_repair_acc + 1e-12 >= repair_floor
        epoch_row["repair_acceptance_floor"] = float(repair_floor)
        search_clean_ok = (
            not eval_clean_items
            or epoch_row["clean_eval_acc"] >= search_clean_floor
        )
        repair_improved = (
            current_repair_acc
            >= accepted_repair_acc + SEARCH_MIN_REPAIR_IMPROVEMENT - 1e-12
        )
        clean_improved = (
            epoch_row["clean_eval_acc"] > accepted_clean_acc + 1e-12
        )
        maintain_target_while_recovering = (
            repair_target_reached
            and current_repair_acc + 1e-12 >= repair_floor
            and epoch_row["clean_eval_acc"] >= accepted_clean_acc - 1e-12
        )
        search_progress = (
            repair_improved
            or clean_improved
            or maintain_target_while_recovering
        )
        if repair_safe and search_clean_ok and delete_ok and search_progress:
            accepted_budget = current_delete_ratio
            accepted_hard_threshold = current_hard_threshold
            accepted_repair_acc = current_repair_acc
            accepted_clean_acc = epoch_row["clean_eval_acc"]
            peak_repair_acc = max(peak_repair_acc, current_repair_acc)
            if peak_repair_acc >= REPAIR_TARGET_BEFORE_COMPRESSION:
                repair_target_reached = True
            epoch_row["accepted_as_repair_safe"] = True
            save_all_masks(model, accepted_mask_path)
            save_binary_masks(model, accepted_binary_mask_path)
            if (
                repair_target_reached
                and clean_ok
                and budget_stage < len(DELETE_RATIO_SCHEDULE) - 1
            ):
                budget_stage += 1
                epoch_row["budget_advanced"] = True
                print(
                    "  [Budget] repair and clean gates passed; next epoch uses "
                    f"{DELETE_RATIO_SCHEDULE[budget_stage]:.4%}."
                )
            elif not repair_target_reached:
                print(
                    "  [Search] accepted cumulative repair progress under the "
                    f"temporary clean floor {search_clean_floor:.4f}; retain "
                    f"the {current_delete_ratio:.4%} budget."
                )
            elif not clean_ok:
                print(
                    "  [Recovery] repair-safe update retained, but clean has "
                    "not reached its floor; budget remains unchanged."
                )
            else:
                print("  [Budget] final compression stage remains feasible.")
            last_clean_acc = accepted_clean_acc
        else:
            accepted_state = torch.load(accepted_mask_path, map_location="cpu")
            AblatedLinear.load_masks(model, accepted_state)
            configure_ste(
                MASK_TEMPERATURE, MASK_INIT_VALUE,
                accepted_hard_threshold, accepted_budget,
            )
            optimizer = torch.optim.AdamW(all_mask_params, lr=LR)
            repair_close_sens_ema = [
                torch.zeros_like(p, dtype=torch.float32)
                for p in all_mask_params
            ]
            clean_open_sens_ema = [
                torch.zeros_like(p, dtype=torch.float32)
                for p in all_mask_params
            ]
            epoch_row["rolled_back"] = True
            last_clean_acc = accepted_clean_acc
            print(
                "  [Rollback] restored last repair-safe mask: "
                f"repair={accepted_repair_acc:.4f}, "
                f"clean={accepted_clean_acc:.4f}, "
                f"budget={accepted_budget:.4%}, "
                f"required_repair_floor={repair_floor:.4f}, "
                f"search_clean_floor={search_clean_floor:.4f}, "
                f"progress={search_progress}."
            )
        write_json(os.path.join(OUTPUT_DIR, "train_epoch_logs.json"), epoch_logs)
        print("=" * 100 + "\n")

    accepted_state = torch.load(accepted_mask_path, map_location="cpu")
    AblatedLinear.load_masks(model, accepted_state)
    configure_ste(
        MASK_TEMPERATURE, MASK_INIT_VALUE,
        accepted_hard_threshold, accepted_budget,
    )
    search_mask_path = os.path.join(
        OUTPUT_DIR, "final_search_hard_logits_masks.pt"
    )
    search_binary_mask_path = os.path.join(
        OUTPUT_DIR, "final_search_hard_binary_masks.pt"
    )
    save_all_masks(model, search_mask_path)
    save_binary_masks(model, search_binary_mask_path)

    final_source = "last_repair_safe_search"
    final_budget = accepted_budget
    if best_strict_available:
        strict_state = torch.load(best_mask_path, map_location="cpu")
        AblatedLinear.load_masks(model, strict_state)
        final_budget = best_max_delete_ratio
        configure_ste(
            MASK_TEMPERATURE, MASK_INIT_VALUE,
            best_hard_threshold, final_budget,
        )
        final_source = "best_strict_repair_clean"

    final_mask_path = os.path.join(OUTPUT_DIR, "final_hard_logits_masks.pt")
    final_binary_mask_path = os.path.join(OUTPUT_DIR, "final_hard_binary_masks.pt")
    save_all_masks(model, final_mask_path)
    save_binary_masks(model, final_binary_mask_path)
    save_mask_visualizations(model, OUTPUT_DIR, tag="final", top_k=500)
    metrics, final_eval_groups = evaluate_groups(
        model=model, tokenizer=tokenizer, judge=judge, eval_items=eval_items,
        group_size=EVAL_GROUP_SIZE, max_prompt_len=MAX_PROMPT_LEN,
        max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P,
        answer_reward_weight=ANSWER_REWARD_WEIGHT,
        reasoning_reward_weight=REASONING_REWARD_WEIGHT,
    )
    final_repair_metrics, final_repair_rows = evaluate_repair_accuracy(
        model=model,
        tokenizer=tokenizer,
        repair_items=eval_items,
        max_prompt_len=MAX_PROMPT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=CLEAN_EVAL_BATCH_SIZE,
    )
    final_clean_metrics = evaluate_clean_accuracy(
        model=model,
        tokenizer=tokenizer,
        clean_items=eval_clean_items,
        max_prompt_len=MAX_PROMPT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=CLEAN_EVAL_BATCH_SIZE,
    )
    final_stats = compute_mask_stats(model)
    write_json(os.path.join(OUTPUT_DIR, "final_eval_rollouts.json"), make_json_serializable_groups(final_eval_groups))
    write_json(os.path.join(OUTPUT_DIR, "final_repair_generations.json"), final_repair_rows)
    write_json(os.path.join(OUTPUT_DIR, "final_repair_metrics.json"), {
        **final_repair_metrics,
        **final_clean_metrics,
        "mlp_deleted": final_stats["mlp_deleted"],
        "mlp_total": final_stats["mlp_total"],
        "mlp_delete_ratio": final_stats["mlp_delete_ratio"],
        "accepted_budget": accepted_budget,
        "accepted_hard_threshold": accepted_hard_threshold,
        "accepted_repair_acc": accepted_repair_acc,
        "accepted_clean_acc": accepted_clean_acc,
        "best_strict_available": best_strict_available,
        "final_source": final_source,
        "final_budget": final_budget,
        "final_hard_threshold": (
            best_hard_threshold
            if final_source == "best_strict_repair_clean"
            else accepted_hard_threshold
        ),
        "best_epoch": best_epoch,
    })
    write_json(os.path.join(OUTPUT_DIR, "final_clean_metrics.json"), final_clean_metrics)
    write_json(os.path.join(OUTPUT_DIR, "train_epoch_logs.json"), epoch_logs)
    save_training_summary_csv(OUTPUT_DIR, step_logs, epoch_logs)
    plot_curves(OUTPUT_DIR, epoch_logs)

    print("\n" + "=" * 100)
    print("RL training finished.")
    if best_epoch is not None:
        print(
            "Best epoch       : "
            f"{best_epoch} (repair_generation_acc={best_repair_generation_acc:.6f}, "
            f"clean_eval_acc={best_clean_acc:.6f}, "
            f"mlp_delete_ratio={best_delete_ratio:.6f}, "
            f"max_delete_ratio={best_max_delete_ratio:.6f})"
        )
    else:
        print("Best epoch       : none saved")
    print(f"Best logits masks : {best_mask_path}")
    print(f"Best binary masks : {best_binary_mask_path}")
    print(f"Accepted budget   : {accepted_budget:.6%}")
    print(f"Accepted threshold: {accepted_hard_threshold:.6f}")
    print(f"Accepted repair   : {accepted_repair_acc:.6f}")
    print(f"Accepted clean    : {accepted_clean_acc:.6f}")
    print(f"Final source      : {final_source}")
    print(
        "Final threshold   : "
        f"{best_hard_threshold if final_source == 'best_strict_repair_clean' else accepted_hard_threshold:.6f}"
    )
    print(f"Search logits     : {search_mask_path}")
    print(f"Search binary     : {search_binary_mask_path}")
    print(
        "Final deleted     : "
        f"{final_stats['mlp_deleted']}/{final_stats['mlp_total']} "
        f"({final_stats['mlp_delete_ratio']:.6%})"
    )
    print(f"Final logits masks: {final_mask_path}")
    print(f"Final binary masks: {final_binary_mask_path}")
    print(f"Curves dir : {OUTPUT_DIR}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
