# Released data

All files in this directory are UTF-8 JSON files whose top-level value is an array. Row order is significant because the training scripts create deterministic train/evaluation slices from the beginning of each array.

## Repair datasets

| File | Rows | Model family | Task |
| --- | ---: | --- | --- |
| `gsm8k_qwen_repair.json` | 223 | Qwen | GSM8K numerical reasoning |
| `gsm8k_llama_repair.json` | 205 | Llama | GSM8K numerical reasoning |
| `med_qwen_repair.json` | 222 | Qwen | Medical multiple choice |
| `med_llama_repair.json` | 260 | Llama | Medical multiple choice |

GSM8K repair rows contain `id`, `question`, `ground_truth`, the original `generation`, and `corrected_reasoning`, together with fields recording the original masked-model prediction and correctness.

The medical exports retain additional source and option metadata. Their answer representation is not identical:

- `med_qwen_repair.json` stores the correct option index in `ground_truth` (`0` through `3`) and the option text in `ground_truth_text`.
- `med_llama_repair.json` stores the option label in `ground_truth_norm`/`correct_answer`, the option text in `ground_truth_text`, and the formatted answer in `target_final_answer`.

The current numerical answer evaluator is suitable for the GSM8K files. Medical evaluation requires task-specific normalization of option indices and labels, such as `0/1/2/3` to `A/B/C/D`.

`gsm8k_llama_repair.json` contains two entries with ID `1390` and the same question but different corrected reasoning traces. They are retained as separate repair traces in this release.

## Clean-preservation datasets

| File | Rows | Model family | Task |
| --- | ---: | --- | --- |
| `gsm8k_qwen_clean.json` | 957 | Qwen | GSM8K clean preservation |
| `gsm8k_llama_clean.json` | 6,458 | Llama | GSM8K clean preservation |

Both files contain questions that the corresponding model answered correctly, along with `ground_truth`, the model `generation`, normalized answer fields, the source prompt, and the original GSM8K answer. They can be passed directly through `--clean-data-path` in the matching Qwen or Llama experiment.

## Test data

There are five numbered test groups for each model family. Every file contains 100 rows:

```text
test_1qwen.json                 test_1qwen_rephrase.json
test_2qwen.json                 test_2qwen_rephrase.json
test_3qwen.json                 test_3qwen_rephrase.json
test_4qwen.json                 test_4qwen_rephrase.json
test_5qwen.json                 test_5qwen_rephrase.json

test_1llama.json                test_1llama_rephrase.json
test_2llama.json                test_2llama_rephrase.json
test_3llama.json                test_3llama_rephrase.json
test_4llama.json                test_4llama_rephrase.json
test_5llama.json                test_5llama_rephrase.json
```

Most test files include the question, reference answer, generation, corrected reasoning, predicted answer, and correctness metadata. The four `test_4*_rephrase.json` and `test_5*_rephrase.json` files contain only `id`, `question`, and `ground_truth`; they are evaluation inputs rather than supervised repair targets.

No exact question overlap was found between the four repair datasets and their same-model `test_*.json` files during the release audit.

## Provenance and licensing

These files contain derived task examples and model generations. Before redistribution, document the exact source dataset version, split construction, generation model, processing procedure, and applicable upstream license in the accompanying paper or release record. Do not add private model paths, credentials, or author-identifying metadata to these JSON files.
