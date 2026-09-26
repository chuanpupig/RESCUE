#!/usr/bin/env python3


import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch


MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count closed MLP mask units and generate random binary masks with "
            "the same number of closed units."
        )
    )
    parser.add_argument("mask_path", help="Source .pt mask file.")
    parser.add_argument(
        "--input-kind",
        choices=("auto", "binary", "logits", "probability"),
        default="auto",
        help=(
            "How to interpret source tensors. Auto detects exact binary values "
            "and otherwise uses filename hints."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("global", "per_tensor"),
        default="per_tensor",
        help=(
            "global samples uniformly from every MLP mask position; per_tensor "
            "preserves the source closed count in every layer/projection tensor."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <mask directory>/random_mask_baselines.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=None,
        help="Override the number of globally closed MLP units.",
    )
    parser.add_argument(
        "--random-ratio",
        type=float,
        default=None,
        help="Override the globally closed MLP ratio, for example 0.003.",
    )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=0.95,
        help="Threshold used when converting logits/probabilities to binary.",
    )
    parser.add_argument(
        "--max-delete-ratio",
        type=float,
        default=1.0,
        help=(
            "Per-tensor deletion cap used by the existing hard-mask conversion. "
            "Use the value from the run that produced a logits mask."
        ),
    )
    parser.add_argument(
        "--mask-init-value",
        type=float,
        default=0.2,
        help="MASK_INIT_VALUE used by the run that produced a logits mask.",
    )
    parser.add_argument(
        "--mask-temperature",
        type=float,
        default=1.0,
        help="MASK_TEMPERATURE used by the run that produced a logits mask.",
    )
    parser.add_argument(
        "--include-non-mlp",
        action="store_true",
        help="Also write non-MLP tensor masks after binary conversion.",
    )
    args = parser.parse_args()
    if args.random_count is not None and args.random_ratio is not None:
        parser.error("--random-count and --random-ratio are mutually exclusive.")
    if args.random_count is not None and args.random_count < 0:
        parser.error("--random-count must be non-negative.")
    if args.random_ratio is not None and not 0.0 <= args.random_ratio <= 1.0:
        parser.error("--random-ratio must be in [0, 1].")
    if not 0.0 < args.hard_threshold < 1.0:
        parser.error("--hard-threshold must be in (0, 1).")
    if not 0.0 <= args.max_delete_ratio <= 1.0:
        parser.error("--max-delete-ratio must be in [0, 1].")
    if args.mask_temperature <= 0.0:
        parser.error("--mask-temperature must be positive.")
    return args


def load_torch_mapping(path: str) -> Dict[str, torch.Tensor]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location="cpu")

    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected a mapping in {path}, got {type(raw).__name__}.")

    for wrapper_key in ("state_dict", "masks", "mask_state"):
        wrapped = raw.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            raw = wrapped
            break

    state = {
        str(name): value.detach().cpu()
        for name, value in raw.items()
        if torch.is_tensor(value)
    }
    if not state:
        raise ValueError(f"No tensor entries found in {path}.")
    return state


def normalized_mask_name(name: str) -> str:
    result = name
    for suffix in (".logits", ".mask", ".weight"):
        if result.endswith(suffix):
            result = result[: -len(suffix)]
    return result


def is_mlp_mask_name(name: str) -> bool:
    normalized = normalized_mask_name(name)
    return any(normalized.endswith(projection) for projection in MLP_PROJECTIONS)


def is_exact_binary(tensor: torch.Tensor) -> bool:
    if tensor.dtype == torch.bool:
        return True
    values = tensor.detach()
    if values.numel() == 0 or not torch.isfinite(values.float()).all():
        return False
    return bool(torch.logical_or(values == 0, values == 1).all().item())


def infer_input_kind(
    requested_kind: str,
    path: str,
    tensors: Iterable[torch.Tensor],
) -> str:
    if requested_kind != "auto":
        return requested_kind
    tensors = list(tensors)
    if tensors and all(is_exact_binary(tensor) for tensor in tensors):
        return "binary"
    filename = Path(path).name.lower()
    if "logit" in filename:
        return "logits"
    if "prob" in filename or "soft" in filename:
        return "probability"
    raise ValueError(
        "Could not safely infer whether the non-binary mask contains logits or "
        "probabilities. Pass --input-kind logits or --input-kind probability."
    )


def effective_mask_probability(
    logits: torch.Tensor,
    init_value: float,
    temperature: float,
) -> torch.Tensor:
    delta = (init_value - logits.float()) / temperature
    close = (2.0 * (torch.sigmoid(delta) - 0.5)).clamp(0.0, 1.0)
    return 1.0 - close


def hard_mask_from_probability(
    probability: torch.Tensor,
    hard_threshold: float,
    max_delete_ratio: float,
) -> torch.Tensor:
    probability = probability.float()
    candidate_close = probability <= hard_threshold
    max_delete = int(probability.numel() * max_delete_ratio)
    if max_delete <= 0:
        return torch.ones_like(probability, dtype=torch.uint8)
    candidate_count = int(candidate_close.sum().item())
    if candidate_count <= max_delete:
        return (~candidate_close).to(torch.uint8)

    flat = probability.flatten()
    _, indices = torch.topk(-flat, k=max_delete)
    close_flat = torch.zeros_like(flat, dtype=torch.bool)
    close_flat[indices] = True
    close_flat &= candidate_close.flatten()
    return (~close_flat).view_as(probability).to(torch.uint8)


def convert_to_binary(
    tensor: torch.Tensor,
    input_kind: str,
    hard_threshold: float,
    max_delete_ratio: float,
    mask_init_value: float,
    mask_temperature: float,
) -> torch.Tensor:
    if input_kind == "binary":
        if not is_exact_binary(tensor):
            raise ValueError("A tensor contains non-binary values under --input-kind binary.")
        return (tensor > 0).to(torch.uint8)
    if input_kind == "logits":
        probability = effective_mask_probability(
            tensor,
            init_value=mask_init_value,
            temperature=mask_temperature,
        )
    elif input_kind == "probability":
        probability = tensor.float()
        if (
            not torch.isfinite(probability).all()
            or float(probability.min().item()) < 0.0
            or float(probability.max().item()) > 1.0
        ):
            raise ValueError("Probability masks must contain finite values in [0, 1].")
    else:
        raise ValueError(f"Unsupported input kind: {input_kind}")
    return hard_mask_from_probability(
        probability,
        hard_threshold=hard_threshold,
        max_delete_ratio=max_delete_ratio,
    )


def mask_stats(state: Mapping[str, torch.Tensor]) -> Tuple[int, int, List[Dict[str, Any]]]:
    total = 0
    closed = 0
    modules: List[Dict[str, Any]] = []
    for name in sorted(state):
        mask = state[name]
        module_total = int(mask.numel())
        module_closed = int((mask == 0).sum().item())
        total += module_total
        closed += module_closed
        modules.append(
            {
                "name": name,
                "shape": list(mask.shape),
                "total": module_total,
                "closed": module_closed,
                "closed_ratio": module_closed / max(module_total, 1),
            }
        )
    return total, closed, modules


def empty_keep_state(source: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        name: torch.ones_like(mask, dtype=torch.uint8)
        for name, mask in source.items()
    }


def make_per_tensor_random_state(
    source: Mapping[str, torch.Tensor],
    seed: int,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result = empty_keep_state(source)
    for name in sorted(source):
        source_mask = source[name]
        count = int((source_mask == 0).sum().item())
        if count == 0:
            continue
        flat = result[name].view(-1)
        indices = torch.randperm(flat.numel(), generator=generator)[:count]
        flat[indices] = 0
    return result


def make_global_random_state(
    source: Mapping[str, torch.Tensor],
    closed_count: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    total = sum(int(mask.numel()) for mask in source.values())
    if not 0 <= closed_count <= total:
        raise ValueError(f"Requested closed count {closed_count} is outside [0, {total}].")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = torch.randperm(total, generator=generator)[:closed_count]
    selected, _ = torch.sort(selected)
    result = empty_keep_state(source)

    names = sorted(source)
    offset = 0
    selected_start = 0
    for name in names:
        flat = result[name].view(-1)
        end = offset + flat.numel()
        selected_end = int(
            torch.searchsorted(selected, torch.tensor(end), right=False).item()
        )
        local = selected[selected_start:selected_end] - offset
        if local.numel():
            flat[local] = 0
        selected_start = selected_end
        offset = end
    return result


def closed_positions(state: Mapping[str, torch.Tensor]) -> set:
    positions = set()
    for name in sorted(state):
        indices = torch.nonzero(state[name].view(-1) == 0, as_tuple=False).view(-1)
        positions.update((name, int(index)) for index in indices.tolist())
    return positions


def overlap_stats(
    source: Mapping[str, torch.Tensor],
    random_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    source_positions = closed_positions(source)
    random_positions = closed_positions(random_state)
    intersection = len(source_positions & random_positions)
    union = len(source_positions | random_positions)
    return {
        "source_closed": len(source_positions),
        "random_closed": len(random_positions),
        "intersection": intersection,
        "source_recall": intersection / max(len(source_positions), 1),
        "jaccard": intersection / max(union, 1),
    }


def parse_seeds(value: str) -> List[int]:
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    return seeds


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    source_path = Path(args.mask_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    raw_state = load_torch_mapping(str(source_path))
    mlp_raw = {name: tensor for name, tensor in raw_state.items() if is_mlp_mask_name(name)}
    if not mlp_raw:
        preview = ", ".join(sorted(raw_state)[:5])
        raise ValueError(
            "No MLP masks ending in gate_proj/up_proj/down_proj were found. "
            f"First keys: {preview}"
        )

    input_kind = infer_input_kind(args.input_kind, str(source_path), mlp_raw.values())
    converted_all = {
        name: convert_to_binary(
            tensor=tensor,
            input_kind=input_kind,
            hard_threshold=args.hard_threshold,
            max_delete_ratio=args.max_delete_ratio,
            mask_init_value=args.mask_init_value,
            mask_temperature=args.mask_temperature,
        )
        for name, tensor in raw_state.items()
    }
    source_mlp = {
        name: tensor
        for name, tensor in converted_all.items()
        if is_mlp_mask_name(name)
    }

    total, source_closed, source_modules = mask_stats(source_mlp)
    requested_closed = source_closed
    if args.random_count is not None:
        requested_closed = args.random_count
    elif args.random_ratio is not None:
        requested_closed = int(round(total * args.random_ratio))

    if args.mode == "per_tensor" and requested_closed != source_closed:
        raise ValueError(
            "--random-count/--random-ratio cannot be combined with --mode per_tensor, "
            "because per_tensor preserves each source tensor's exact closed count."
        )
    if requested_closed > total:
        raise ValueError(f"Requested {requested_closed} closed units, but only {total} exist.")

    seeds = parse_seeds(args.seeds)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else source_path.parent / "random_mask_baselines"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "source_mask": str(source_path),
        "input_kind": input_kind,
        "mask_semantics": {"keep": 1, "closed": 0},
        "mlp_projection_names": list(MLP_PROJECTIONS),
        "conversion": {
            "hard_threshold": args.hard_threshold,
            "max_delete_ratio": args.max_delete_ratio,
            "mask_init_value": args.mask_init_value,
            "mask_temperature": args.mask_temperature,
        },
        "sampling_mode": args.mode,
        "source_total_mlp_mask_units": total,
        "source_closed_mlp_mask_units": source_closed,
        "source_closed_ratio": source_closed / max(total, 1),
        "requested_random_closed_units": requested_closed,
        "source_modules": source_modules,
        "generated": [],
    }

    for seed in seeds:
        if args.mode == "per_tensor":
            random_mlp = make_per_tensor_random_state(source_mlp, seed)
        else:
            random_mlp = make_global_random_state(source_mlp, requested_closed, seed)

        output_state = dict(random_mlp)
        if args.include_non_mlp:
            output_state = {
                name: (
                    random_mlp[name]
                    if name in random_mlp
                    else converted_all[name].to(torch.uint8)
                )
                for name in converted_all
            }

        stem = source_path.stem
        output_path = output_dir / f"{stem}.random_{args.mode}.seed_{seed}.binary_masks.pt"
        torch.save(output_state, output_path)

        random_total, random_closed, random_modules = mask_stats(random_mlp)
        generated_report = {
            "seed": seed,
            "path": str(output_path),
            "total": random_total,
            "closed": random_closed,
            "closed_ratio": random_closed / max(random_total, 1),
            "overlap_with_source": overlap_stats(source_mlp, random_mlp),
            "modules": random_modules,
        }
        report["generated"].append(generated_report)
        print(
            f"[seed={seed}] closed={random_closed}/{random_total} "
            f"({100.0 * random_closed / max(random_total, 1):.6f}%) -> {output_path}"
        )

    report_path = output_dir / f"{source_path.stem}.random_{args.mode}.report.json"
    write_json(report_path, report)

    print("")
    print(f"Input kind: {input_kind}")
    print(
        f"Source MLP closed units: {source_closed}/{total} "
        f"({100.0 * source_closed / max(total, 1):.6f}%)"
    )
    print(f"Sampling mode: {args.mode}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
