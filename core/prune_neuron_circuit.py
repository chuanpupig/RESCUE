import argparse
import os
import json
import random
import sys
import importlib.util
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = None
REPAIR_DATA_PATH = None
CLEAN_DATA_PATH = None
RL_OUTPUT_DIR = None
OUTPUT_DIR = None

BASE_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "refine_neuron_circuit_ste.py")
INIT_LOGITS_MASK_PATH = None
INIT_BINARY_MASK_PATH = None
MASK_INIT_VALUE = 0.2
MASK_TEMPERATURE = 1.0
HARD_THRESHOLD = 0.95
MAX_DELETE_RATIO = 0.005

MAX_PROMPT_LEN = 512
MAX_NEW_TOKENS = 256
MAX_TOTAL_LEN_FOR_SCORING = 768
TRAIN_SAMPLES = 80
EVAL_SAMPLES = 20
TRAIN_CLEAN_SAMPLES = 80
EVAL_CLEAN_SAMPLES = 20
PRUNE_REPAIR_SAMPLES = 100
PRUNE_CLEAN_SAMPLES = 100
REPAIR_EVAL_BATCH_SIZE = 32
CLEAN_EVAL_BATCH_SIZE = 32


REPAIR_ACC_TOLERANCE = 0.0
CLEAN_ACC_TOLERANCE = 0.0


PRUNE_CHUNK_SIZES = [256, 128, 64, 32]
PRUNE_MAX_TRIALS_PER_PASS = 100000
PRUNE_SAVE_EVERY_ACCEPT = True


KEEP_LOGIT = MASK_INIT_VALUE + 20.0
CLOSE_LOGIT = MASK_INIT_VALUE - 20.0


def load_base_module(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Base RL script not found: {path}\n"
            "Keep refine_neuron_circuit_ste.py next to this script or pass --base-script-path."
        )
    spec = importlib.util.spec_from_file_location("rl_mask_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module(BASE_SCRIPT_PATH)
_ORIGINAL_ABLATED_GET_MASK = base.AblatedLinear.get_mask


def _exact_binary_aware_get_mask(self, hard=None):
    exact_mask = getattr(self, "_post_exact_binary_mask", None)
    if getattr(base.AblatedLinear, "use_post_exact_binary_forward", False) and exact_mask is not None:
        return exact_mask.to(self.original_layer.weight.dtype)
    return _ORIGINAL_ABLATED_GET_MASK(self, hard=hard)


base.AblatedLinear.get_mask = _exact_binary_aware_get_mask
base.AblatedLinear.use_post_exact_binary_forward = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_logits_masks_if_available(model, path: Optional[str]) -> int:
    if not path or not os.path.exists(path):
        return 0
    state = torch.load(path, map_location="cpu", weights_only=True)
    n_mlp = base.AblatedLinear.load_masks(model, state)
    print(f"[Load] neuron-mask logits: modules={n_mlp}, path={path}")
    return n_mlp


def extract_binary_state_from_model(model) -> Dict[str, torch.Tensor]:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, base.AblatedLinear):
            mask_soft = base.effective_mask_prob(module.logits.detach())
            hard = base.hard_mask_from_soft(
                mask_soft,
                base.AblatedLinear.hard_threshold,
                base.AblatedLinear.max_delete_ratio,
            )
            state[name] = hard.to(torch.uint8).cpu()
    return state


def load_binary_state(path: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
    if not path or not os.path.exists(path):
        return None
    raw = torch.load(path, map_location="cpu", weights_only=True)
    state = {}
    for name, mask in raw.items():
        if torch.is_tensor(mask):
            state[name] = (mask.detach().cpu() > 0).to(torch.uint8)
    print(f"[Load] binary masks: {len(state)} tensors, path={path}")
    return state


def apply_binary_state_to_model(model, state: Dict[str, torch.Tensor]):
    for name, module in model.named_modules():
        if isinstance(module, base.AblatedLinear) and name in state:
            mask = state[name].to(module.logits.device).view_as(module.logits).bool()
            keep = torch.full_like(module.logits.data, KEEP_LOGIT)
            close = torch.full_like(module.logits.data, CLOSE_LOGIT)
            module.logits.data.copy_(torch.where(mask, keep, close))


def install_exact_binary_masks(model, state: Dict[str, torch.Tensor]):
    for name, module in model.named_modules():
        if isinstance(module, base.AblatedLinear) and name in state:
            module._post_exact_binary_mask = state[name].to(module.logits.device).view_as(module.logits).to(torch.uint8)


def set_exact_binary_forward(enabled: bool):
    base.AblatedLinear.use_post_exact_binary_forward = bool(enabled)


def save_binary_state(path: str, state: Dict[str, torch.Tensor]):
    torch.save({k: v.to(torch.uint8).cpu() for k, v in state.items()}, path)
    print(f"[Save] binary state -> {path}")


def copy_binary_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in state.items()}


def mask_deleted_count(state: Dict[str, torch.Tensor]) -> Tuple[int, int, float]:
    total = sum(int(v.numel()) for v in state.values())
    deleted = sum(int((v == 0).sum().item()) for v in state.values())
    return deleted, total, deleted / max(total, 1)


def binary_state_from_paths_or_logits(model, logits_path: Optional[str], binary_path: Optional[str]) -> Dict[str, torch.Tensor]:
    binary_state = load_binary_state(binary_path)
    if binary_state is not None:
        return binary_state
    loaded = load_logits_masks_if_available(model, logits_path)
    if loaded <= 0:
        raise FileNotFoundError("No initial binary/logits mask was found.")
    return extract_binary_state_from_model(model)


def collect_deleted_candidates(model, binary_state: Dict[str, torch.Tensor]) -> List[Tuple[str, int, float]]:

    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, base.AblatedLinear) or name not in binary_state:
            continue
        binary = binary_state[name].view(-1)
        mask_soft = base.effective_mask_prob(module.logits.detach()).float().cpu().view(-1)
        closure = 1.0 - mask_soft
        deleted = torch.nonzero(binary == 0, as_tuple=False).view(-1).tolist()
        for idx in deleted:
            candidates.append((name, int(idx), float(closure[idx].item())))
    candidates.sort(key=lambda x: x[2])
    return candidates


def set_positions_to_keep(state: Dict[str, torch.Tensor], positions: List[Tuple[str, int, float]]):
    for name, idx, _ in positions:
        if name in state:
            state[name].view(-1)[idx] = 1


def make_candidate_train_mask(binary_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: (mask == 0).float() for name, mask in binary_state.items()}


def register_candidate_grad_hooks(model, candidate_train_mask: Dict[str, torch.Tensor]):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, base.AblatedLinear) and name in candidate_train_mask:
            m = candidate_train_mask[name].to(module.logits.device).view_as(module.logits)
            hooks.append(module.logits.register_hook(lambda grad, m=m: grad * m))
    return hooks


@torch.no_grad()
def force_non_candidates_keep(model, candidate_train_mask: Dict[str, torch.Tensor]):
    for name, module in model.named_modules():
        if isinstance(module, base.AblatedLinear) and name in candidate_train_mask:
            cand = candidate_train_mask[name].to(module.logits.device).view_as(module.logits).bool()
            module.logits.data.copy_(torch.where(
                cand,
                module.logits.data,
                torch.full_like(module.logits.data, KEEP_LOGIT),
            ))


def candidate_closure_stats(model, candidate_train_mask: Dict[str, torch.Tensor]) -> Dict[str, float]:
    vals = []
    for name, module in model.named_modules():
        if isinstance(module, base.AblatedLinear) and name in candidate_train_mask:
            cand = candidate_train_mask[name].to(module.logits.device).view_as(module.logits).bool()
            if cand.any():
                mask = base.effective_mask_prob(module.logits.detach()).float()
                vals.append((1.0 - mask[cand]).detach().cpu())
    if not vals:
        return {"candidate_count": 0, "closure_mean": 0.0, "closure_min": 0.0, "closure_max": 0.0}
    x = torch.cat([v.flatten() for v in vals])
    return {
        "candidate_count": int(x.numel()),
        "closure_mean": float(x.mean().item()),
        "closure_min": float(x.min().item()),
        "closure_max": float(x.max().item()),
    }


@torch.no_grad()
def evaluate_repair_answer_acc(model, tokenizer, items, max_prompt_len: int, max_new_tokens: int,
                               batch_size: int = 1, save_path: Optional[str] = None):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    base.AblatedLinear.enabled = True

    rows = []
    total = 0
    correct = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [base.build_chat_prompt(item.question, tokenizer) for item in batch]
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
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for i, item in enumerate(batch):
            gen_ids = out[i][prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = base.normalize_answer_str(base.extract_final_answer(text))
            gt = base.normalize_answer_str(item.ground_truth)
            ok = pred != "" and pred == gt
            total += 1
            correct += int(ok)
            rows.append({
                "qid": item.qid,
                "question": item.question,
                "ground_truth": item.ground_truth,
                "pred": pred,
                "correct": bool(ok),
                "generation": text,
            })
    tokenizer.padding_side = old_padding_side
    if save_path:
        write_json(save_path, rows)
    return {"repair_acc": correct / max(total, 1), "repair_total": total, "repair_correct": correct}


def build_repair_sft_items(raw_data: List[Dict[str, Any]]):
    items = []
    for i, item in enumerate(raw_data):
        question = base.normalize_text(base.safe_get(item, ["question", "prompt"]))
        gt = base.normalize_text(base.safe_get(item, ["ground_truth", "answer"]))
        target = base.normalize_text(base.safe_get(item, [
            "corrected_reasoning",
            "repair_response",
            "target",
            "solution",
            "reasoning",
        ]))
        if target and gt and base.extract_final_answer(target) == "":
            target = target.rstrip() + f"\nFinal answer: {gt}"
        if question and target:
            items.append(base.CleanItem(qid=base.safe_get(item, ["id"], i), question=question, target_text=target, ground_truth=gt))
    return items


def build_supervised_batch(items, tokenizer, batch_size: int, max_total_len: int):
    if not items:
        return None
    batch = random.sample(items, k=min(batch_size, len(items)))
    full_ids_list, labels_list, attn_list = [], [], []
    max_len = 0
    for item in batch:
        prompt = base.build_chat_prompt(item.question, tokenizer)
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


def compute_repair_supervised_loss(model, tokenizer, repair_sft_items, batch_size: int, max_total_len: int):
    batch = build_supervised_batch(repair_sft_items, tokenizer, batch_size, max_total_len)
    if batch is None:
        return torch.tensor(0.0, device=model.device)
    input_ids, attention_mask, labels = [x.to(model.device) for x in batch]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return base.token_nll_loss(outputs.logits, labels)


def compute_candidate_open_loss(model, candidate_train_mask: Dict[str, torch.Tensor]):
    losses = []
    for name, module in model.named_modules():
        if isinstance(module, base.AblatedLinear) and name in candidate_train_mask:
            cand = candidate_train_mask[name].to(module.logits.device).view_as(module.logits).bool()
            if cand.any():
                mask = base.effective_mask_prob(module.logits)
                losses.append((1.0 - mask[cand]).mean())
    if not losses:
        return torch.tensor(0.0, device=model.device)
    return torch.stack(losses).mean()


def evaluate_current(model, tokenizer, repair_eval_items, clean_eval_items, output_dir: str,
                     tag: str, save_generations: bool = True):
    repair_metrics = evaluate_repair_answer_acc(
        model=model,
        tokenizer=tokenizer,
        items=repair_eval_items,
        max_prompt_len=MAX_PROMPT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=REPAIR_EVAL_BATCH_SIZE,
        save_path=os.path.join(output_dir, f"{tag}_repair_generations.json") if save_generations else None,
    )
    clean_metrics = base.evaluate_clean_accuracy(
        model=model,
        tokenizer=tokenizer,
        clean_items=clean_eval_items,
        max_prompt_len=MAX_PROMPT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=CLEAN_EVAL_BATCH_SIZE,
    )
    return {**repair_metrics, **clean_metrics}


def run_prune(model, tokenizer, init_state, candidates, repair_eval_items, clean_eval_items,
              repair_floor: float, clean_floor: float):
    print("\n" + "=" * 100)
    print("[Post Prune] reopening redundant deleted neurons")
    print("=" * 100)
    state = copy_binary_state(init_state)
    init_deleted, init_total, init_ratio = mask_deleted_count(state)
    apply_binary_state_to_model(model, state)
    install_exact_binary_masks(model, state)
    set_exact_binary_forward(True)

    log_path = os.path.join(OUTPUT_DIR, "post_prune_logs.jsonl")
    accepted_total = 0
    trial_total = 0

    for chunk_size in PRUNE_CHUNK_SIZES:
        print(f"\n[Prune Pass] chunk_size={chunk_size}")
        i = 0
        trials_this_pass = 0
        while i < len(candidates) and trials_this_pass < PRUNE_MAX_TRIALS_PER_PASS:
            chunk = []
            while i < len(candidates) and len(chunk) < chunk_size:
                name, idx, score = candidates[i]
                if state.get(name) is not None and int(state[name].view(-1)[idx].item()) == 0:
                    chunk.append((name, idx, score))
                i += 1
            if not chunk:
                continue

            trial_total += 1
            trials_this_pass += 1
            proposal = copy_binary_state(state)
            set_positions_to_keep(proposal, chunk)
            apply_binary_state_to_model(model, proposal)
            install_exact_binary_masks(model, proposal)

            metrics = evaluate_current(
                model, tokenizer, repair_eval_items, clean_eval_items, OUTPUT_DIR,
                tag=f"prune_trial_{trial_total}", save_generations=False,
            )
            deleted, total, ratio = mask_deleted_count(proposal)
            ok = metrics["repair_acc"] >= repair_floor and metrics["clean_eval_acc"] >= clean_floor
            row = {
                "trial": trial_total,
                "chunk_size": chunk_size,
                "tried_reopen": len(chunk),
                "accepted": bool(ok),
                "deleted": deleted,
                "total": total,
                "delete_ratio": ratio,
                **metrics,
            }
            append_jsonl(log_path, row)
            if ok:
                state = proposal
                install_exact_binary_masks(model, state)
                accepted_total += len(chunk)
                print(
                    f"  accept reopen={len(chunk)}, deleted={deleted}/{total} "
                    f"({ratio*100:.4f}%), repair={metrics['repair_acc']:.4f}, clean={metrics['clean_eval_acc']:.4f}"
                )
                if PRUNE_SAVE_EVERY_ACCEPT:
                    save_binary_state(os.path.join(OUTPUT_DIR, "pruned_binary_masks_latest.pt"), state)
                    base.save_all_masks(model, os.path.join(OUTPUT_DIR, "pruned_logits_masks_latest.pt"))
            else:
                apply_binary_state_to_model(model, state)
                install_exact_binary_masks(model, state)
                print(
                    f"  reject reopen={len(chunk)}, repair={metrics['repair_acc']:.4f}, "
                    f"clean={metrics['clean_eval_acc']:.4f}"
                )

    apply_binary_state_to_model(model, state)
    install_exact_binary_masks(model, state)
    final_metrics = evaluate_current(model, tokenizer, repair_eval_items, clean_eval_items, OUTPUT_DIR, tag="pruned_final")
    deleted, total, ratio = mask_deleted_count(state)
    summary = {
        "accepted_reopened": accepted_total,
        "initial_deleted": init_deleted,
        "initial_total": init_total,
        "initial_delete_ratio": init_ratio,
        "final_deleted": deleted,
        "total": total,
        "final_delete_ratio": ratio,
        "repair_floor": repair_floor,
        "clean_floor": clean_floor,
        "exact_binary_forward": True,
        **final_metrics,
    }
    write_json(os.path.join(OUTPUT_DIR, "post_prune_summary.json"), summary)
    write_json(os.path.join(OUTPUT_DIR, "final_deleted_neuron_report.json"), {
        "initial_deleted_neurons": init_deleted,
        "final_deleted_neurons": deleted,
        "reopened_redundant_neurons": init_deleted - deleted,
        "total_neurons": total,
        "initial_delete_ratio": init_ratio,
        "final_delete_ratio": ratio,
        "initial_delete_percent": init_ratio * 100.0,
        "final_delete_percent": ratio * 100.0,
        "repair_acc": final_metrics.get("repair_acc", 0.0),
        "clean_eval_acc": final_metrics.get("clean_eval_acc", 0.0),
        "repair_floor": repair_floor,
        "clean_floor": clean_floor,
        "exact_binary_forward": True,
        "source_summary": "post_prune_summary.json",
        "final_binary_masks": "pruned_binary_masks.pt",
        "final_logits_masks": "pruned_logits_masks.pt",
    })
    save_binary_state(os.path.join(OUTPUT_DIR, "pruned_binary_masks.pt"), state)
    base.save_all_masks(model, os.path.join(OUTPUT_DIR, "pruned_logits_masks.pt"))
    print(f"[Post Prune] final: {summary}")
    return state, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Greedily reopen redundant neurons from an STE circuit.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repair-data-path", required=True)
    parser.add_argument("--clean-data-path", required=True)
    parser.add_argument("--rl-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-logits-mask-path")
    parser.add_argument("--init-binary-mask-path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    global MODEL_PATH, REPAIR_DATA_PATH, CLEAN_DATA_PATH, RL_OUTPUT_DIR, OUTPUT_DIR
    global INIT_LOGITS_MASK_PATH, INIT_BINARY_MASK_PATH
    args = parse_args()
    MODEL_PATH = args.model_path
    REPAIR_DATA_PATH = args.repair_data_path
    CLEAN_DATA_PATH = args.clean_data_path
    RL_OUTPUT_DIR = args.rl_output_dir
    OUTPUT_DIR = args.output_dir
    INIT_LOGITS_MASK_PATH = args.init_logits_mask_path
    INIT_BINARY_MASK_PATH = args.init_binary_mask_path
    set_seed(args.seed)
    ensure_dir(OUTPUT_DIR)

    logits_path = INIT_LOGITS_MASK_PATH or resolve_first_existing([
        os.path.join(RL_OUTPUT_DIR, "best_hard_logits_masks.pt"),
        os.path.join(RL_OUTPUT_DIR, "final_hard_logits_masks.pt"),
        os.path.join(RL_OUTPUT_DIR, "pruned_logits_masks.pt"),
    ])
    binary_path = INIT_BINARY_MASK_PATH or resolve_first_existing([
        os.path.join(RL_OUTPUT_DIR, "best_hard_binary_masks.pt"),
        os.path.join(RL_OUTPUT_DIR, "final_hard_binary_masks.pt"),
        os.path.join(RL_OUTPUT_DIR, "pruned_binary_masks.pt"),
    ])

    print("=" * 100)
    print("Post-RL hard neuron-mask pruning")
    print("=" * 100)
    print(f"BASE_SCRIPT_PATH     = {BASE_SCRIPT_PATH}")
    print(f"INIT_LOGITS_MASK_PATH= {logits_path}")
    print(f"INIT_BINARY_MASK_PATH= {binary_path}")
    print(f"OUTPUT_DIR           = {OUTPUT_DIR}")
    print(f"PRUNE_REPAIR_SAMPLES = {PRUNE_REPAIR_SAMPLES}")
    print(f"PRUNE_CLEAN_SAMPLES  = {PRUNE_CLEAN_SAMPLES}")
    print(f"REPAIR_EVAL_BATCH    = {REPAIR_EVAL_BATCH_SIZE}")
    print(f"CLEAN_EVAL_BATCH     = {CLEAN_EVAL_BATCH_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base.disable_generation_max_length_warning(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    base.patch_model(model)
    base.configure_ste(MASK_TEMPERATURE, MASK_INIT_VALUE, HARD_THRESHOLD, MAX_DELETE_RATIO)
    load_logits_masks_if_available(model, logits_path)
    init_state = binary_state_from_paths_or_logits(model, logits_path, binary_path)
    candidates = collect_deleted_candidates(model, init_state)
    apply_binary_state_to_model(model, init_state)
    install_exact_binary_masks(model, init_state)
    set_exact_binary_forward(True)

    raw_repair = base.load_json_auto(REPAIR_DATA_PATH)
    repair_items = base.build_rl_items(raw_repair)
    repair_eval_items = repair_items[:PRUNE_REPAIR_SAMPLES]

    raw_clean = base.load_json_auto(CLEAN_DATA_PATH) if CLEAN_DATA_PATH else []
    clean_items = base.build_clean_items(raw_clean)
    clean_eval_items = clean_items[:PRUNE_CLEAN_SAMPLES]

    if not repair_eval_items:
        raise ValueError("repair_eval_items is empty.")

    init_deleted, init_total, init_ratio = mask_deleted_count(init_state)
    init_metrics = evaluate_current(model, tokenizer, repair_eval_items, clean_eval_items, OUTPUT_DIR, tag="init")
    repair_floor = max(0.0, init_metrics["repair_acc"] - REPAIR_ACC_TOLERANCE)
    clean_floor = max(0.0, init_metrics["clean_eval_acc"] - CLEAN_ACC_TOLERANCE)
    write_json(os.path.join(OUTPUT_DIR, "init_metrics.json"), {
        "init_deleted": init_deleted,
        "init_total": init_total,
        "init_delete_ratio": init_ratio,
        "repair_floor": repair_floor,
        "clean_floor": clean_floor,
        "repair_eval_items": len(repair_eval_items),
        "clean_eval_items": len(clean_eval_items),
        "repair_eval_batch_size": REPAIR_EVAL_BATCH_SIZE,
        "clean_eval_batch_size": CLEAN_EVAL_BATCH_SIZE,
        "exact_binary_forward": True,
        **init_metrics,
    })
    print(
        f"[Init] deleted={init_deleted}/{init_total} ({init_ratio*100:.4f}%), "
        f"repair={init_metrics['repair_acc']:.4f}, clean={init_metrics['clean_eval_acc']:.4f}"
    )


    write_json(os.path.join(OUTPUT_DIR, "deleted_candidates_ranked.json"), [
        {"name": n, "index": i, "closure": s} for n, i, s in candidates
    ])
    print(f"[Candidates] deleted neurons available for pruning = {len(candidates)}")

    run_prune(
        model=model,
        tokenizer=tokenizer,
        init_state=init_state,
        candidates=candidates,
        repair_eval_items=repair_eval_items,
        clean_eval_items=clean_eval_items,
        repair_floor=repair_floor,
        clean_floor=clean_floor,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
