# RESCUE

This repository contains the neuron-only reproduction code and released data for RESCUE. The implementation masks neurons in the MLP projections (`gate_proj`, `up_proj`, and `down_proj`) and uses binary masks in the forward pass with straight-through-estimator (STE) gradients. Attention-head masking and alternative soft-refinement modes are not included.

## Repository layout

```text
RESCUE-release/
├── core/          Circuit discovery, STE refinement, pruning, and model baking
├── baselines/     LoRA, circuit-only fine-tuning, mask randomization, and LoRA merging
├── configs/       Notes for recording experiment-specific commands
├── data/          Released repair and test JSON files
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment

Python 3.10 or newer is recommended. Install the dependencies in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The model checkpoints are not distributed in this repository. Supply local paths or Hugging Face model identifiers through the command-line arguments.

## Data

The repository includes two clean-preservation datasets, four repair datasets, and five numbered test groups for both Qwen and Llama, with corresponding rephrased variants. See [`data/README.md`](data/README.md) for file counts, schemas, and important usage notes.

The optimization scripts require a clean-preservation dataset. Use `data/gsm8k_qwen_clean.json` for the Qwen GSM8K experiments and `data/gsm8k_llama_clean.json` for the Llama GSM8K experiments. Clean rows contain:

- a question in `question` or `prompt`;
- a reference answer in `ground_truth`, `answer`, or `label`;
- preferably a full clean response in `generation`, `response`, `solution`, or `corrected_reasoning`.

Do not substitute a repair set for the clean set unless that is part of the intended experimental protocol.

## Main pipeline

All paths are configurable; the scripts do not contain machine-specific model or data paths.

### 1. Discover a neuron circuit

```bash
python core/discover_neuron_circuit.py \
  --model-path /path/to/Qwen3-8B \
  --repair-data-path data/gsm8k_qwen_repair.json \
  --clean-data-path data/gsm8k_qwen_clean.json \
  --output-dir outputs/discovery
```

This stage optimizes only MLP-neuron mask logits. Its forward pass is binary and its backward pass uses STE. By default, the first 80 repair rows are used for training and the next 20 for evaluation; the clean split follows the same 80/20 convention. The split sizes can be changed with the corresponding command-line arguments.

### 2. Refine the circuit with STE

```bash
python core/refine_neuron_circuit_ste.py \
  --model-path /path/to/Qwen3-8B \
  --judge-model-path /path/to/judge-model \
  --rl-data-path data/gsm8k_qwen_repair.json \
  --clean-data-path data/gsm8k_qwen_clean.json \
  --init-mask-path outputs/discovery/best_rescue_masks.pt \
  --output-dir outputs/refinement
```

If an exact companion binary mask is available, also pass `--init-binary-mask-path PATH`. The principal outputs include `best_hard_logits_masks.pt`, `best_hard_binary_masks.pt`, `final_hard_logits_masks.pt`, and `final_hard_binary_masks.pt`.

### 3. Prune redundant closed neurons

```bash
python core/prune_neuron_circuit.py \
  --model-path /path/to/Qwen3-8B \
  --repair-data-path data/gsm8k_qwen_repair.json \
  --clean-data-path data/gsm8k_qwen_clean.json \
  --rl-output-dir outputs/refinement \
  --init-logits-mask-path outputs/refinement/best_hard_logits_masks.pt \
  --init-binary-mask-path outputs/refinement/best_hard_binary_masks.pt \
  --output-dir outputs/pruned
```

The final pruning outputs are `pruned_logits_masks.pt` and `pruned_binary_masks.pt`.

### 4. Bake a binary mask into a model

```bash
python core/bake_neuron_circuit.py \
  --base-model /path/to/Qwen3-8B \
  --binary-mask-path outputs/pruned/pruned_binary_masks.pt \
  --output-dir outputs/baked-model
```

The baking script accepts exact 0/1 masks only and rejects probability or logit masks.

## Baselines and utilities

- `baselines/full_lora.py`: LoRA fine-tuning baseline. Add `--mlp-only` to target only `gate_proj`, `up_proj`, and `down_proj`.
- `baselines/circuit_only_finetune.py`: fine-tunes weights restricted to a supplied neuron circuit.
- `baselines/random_neuron_mask.py`: generates seeded random neuron masks matched to a source mask.
- `baselines/merge_lora.py`: merges a trained LoRA adapter into its base model.

Each script exposes its complete interface through `python SCRIPT --help` after the dependencies have been installed.

## Data and evaluation notes

The included GSM8K files use numerical answers and match the numerical answer normalization used by the current training-time evaluators.

The included medical files are multiple-choice data. Their answer fields differ between the Qwen and Llama exports, and the present numerical evaluator is not a standalone medical multiple-choice evaluator. To reproduce medical-task metrics, normalize option indices and labels consistently (for example, `0/1/2/3` to `A/B/C/D`) and use the protocol reported for that experiment.

This release does not include a separate general-purpose evaluation script. Discovery and refinement write their own intermediate generations and metrics to the selected output directory. The `test_*.json` files are released for final evaluation under the experiment's evaluation protocol.

## Validation

The source files can be checked without loading a model:

```bash
python -m compileall -q core baselines
```

A complete numerical run additionally requires compatible model checkpoints, sufficient GPU memory, and the exact experiment settings.

## Release hygiene

Do not commit model checkpoints, output directories, API keys, personal paths, or restricted data. The provided `.gitignore` excludes common generated files and model-weight formats. Before redistributing a derived dataset or model output, verify the applicable upstream license and terms.
