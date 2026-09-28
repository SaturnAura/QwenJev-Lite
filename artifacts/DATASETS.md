# Dataset results

- generated: 2026-09-28T09:55:55+00:00
- items per task: 100
- variants: `laya`, `qwen_zeroshot`, `qwen_rlcd`

## Accuracy

| task | laya | qwen_zeroshot | qwen_rlcd | majority | chance |
|---|---|---|---|---|---|
| mnli | 0.549 | 0.764 | 0.754 | 0.399 | 0.333 |
| mnli_ood | 0.542 | 0.828 | 0.815 | 0.394 | 0.333 |
| snli | 0.639 | 0.889 | 0.919 | 0.385 | 0.333 |
| fever | 0.280 | 0.330 | 0.410 | 0.360 | 0.333 |
| jigsaw | 0.907 | 0.863 | 0.970 | 0.962 | 0.583 |
| goemotions | 0.896 | 0.718 | 0.956 | 0.957 | 0.571 |
| truthfulqa | 0.210 | 0.570 | 0.620 | 0.310 | 0.200 |
| mmlu_pro | 0.030 | 0.240 | 0.270 | 0.140 | 0.100 |
| intentgrasp | 0.280 | 0.370 | 0.350 | 0.190 | 0.100 |
| clinc150 | 0.020 | - | 0.020 | 0.010 | 0.010 |
| hwu64 | 0.030 | - | 0.050 | 0.030 | 0.021 |

## Balanced accuracy (mean per-class recall; use this when one class dominates)

| task | laya | qwen_zeroshot | qwen_rlcd | majority |
|---|---|---|---|---|
| mnli | 0.548 | 0.770 | 0.763 | 0.333 |
| mnli_ood | 0.540 | 0.828 | 0.817 | 0.333 |
| snli | 0.637 | 0.889 | 0.919 | 0.333 |
| fever | 0.333 | 0.381 | 0.373 | 0.333 |
| jigsaw | 0.783 | 0.929 | 0.550 | 0.583 |
| goemotions | 0.691 | 0.669 | 0.500 | 0.571 |
| truthfulqa | 0.208 | 0.559 | 0.632 | 0.200 |
| mmlu_pro | 0.032 | 0.249 | 0.231 | 0.100 |
| intentgrasp | 0.225 | 0.358 | 0.414 | 0.100 |
| clinc150 | 0.020 | - | 0.020 | 0.010 |
| hwu64 | 0.021 | - | 0.056 | 0.021 |

## Calibration (ECE over the answer the model chose)

| task | laya | qwen_zeroshot | qwen_rlcd |
|---|---|---|---|
| mnli | 0.232 | 0.093 | 0.136 |
| mnli_ood | 0.225 | 0.110 | 0.077 |
| snli | 0.148 | 0.098 | 0.049 |
| fever | 0.709 | 0.449 | 0.391 |
| jigsaw | 0.026 | 0.142 | 0.023 |
| goemotions | 0.023 | 0.093 | 0.042 |
| truthfulqa | 0.323 | 0.138 | 0.236 |
| mmlu_pro | 0.313 | 0.224 | 0.363 |
| intentgrasp | 0.262 | 0.241 | 0.519 |
| clinc150 | 0.003 | - | 0.687 |
| hwu64 | 0.037 | - | 0.572 |

## Speed

| task | variant | requests | ms/request | tok/req | n |
|---|---|---|---|---|---|
| mnli | laya | 297 | 24.7 | 115.5 | 297 |
| mnli | qwen_zeroshot | 100 | 518.3 | 286.6 | 297 |
| mnli | qwen_rlcd | 100 | 518.6 | 286.6 | 297 |
| mnli_ood | laya | 297 | 23.9 | 116.9 | 297 |
| mnli_ood | qwen_zeroshot | 100 | 517.7 | 291.1 | 297 |
| mnli_ood | qwen_rlcd | 100 | 511.4 | 291.1 | 297 |
| snli | laya | 296 | 24.2 | 101.7 | 296 |
| snli | qwen_zeroshot | 100 | 512.9 | 266.0 | 296 |
| snli | qwen_rlcd | 100 | 513.6 | 266.0 | 296 |
| fever | laya | 100 | 23.7 | 94.9 | 100 |
| fever | qwen_zeroshot | 100 | 509.1 | 96.9 | 100 |
| fever | qwen_rlcd | 100 | 507.2 | 96.9 | 100 |
| jigsaw | laya | 600 | 25.2 | 127.6 | 600 |
| jigsaw | qwen_zeroshot | 100 | 636.5 | 301.4 | 600 |
| jigsaw | qwen_rlcd | 100 | 635.6 | 301.4 | 600 |
| goemotions | laya | 2800 | 27.4 | 62.1 | 2800 |
| goemotions | qwen_zeroshot | 100 | 694.1 | 979.9 | 2800 |
| goemotions | qwen_rlcd | 100 | 672.5 | 979.9 | 2800 |
| truthfulqa | laya | 100 | 29.5 | 105.3 | 100 |
| truthfulqa | qwen_zeroshot | 100 | 652.0 | 110.6 | 100 |
| truthfulqa | qwen_rlcd | 100 | 641.2 | 110.6 | 100 |
| mmlu_pro | laya | 100 | 29.0 | 200.3 | 100 |
| mmlu_pro | qwen_zeroshot | 100 | 640.5 | 211.4 | 100 |
| mmlu_pro | qwen_rlcd | 100 | 517.7 | 211.4 | 100 |
| intentgrasp | laya | 100 | 25.3 | 237.9 | 100 |
| intentgrasp | qwen_zeroshot | 100 | 566.8 | 246.4 | 100 |
| intentgrasp | qwen_rlcd | 100 | 568.0 | 246.4 | 100 |
| clinc150 | laya | 100 | 33.6 | 626.2 | 100 |
| clinc150 | qwen_rlcd | 100 | 753.1 | 1254.2 | 100 |
| hwu64 | laya | 100 | 27.0 | 279.7 | 100 |
| hwu64 | qwen_rlcd | 100 | 622.5 | 555.8 | 100 |

## Task notes

- **clinc150**: a many-label closed-set task; the readout needs the slot head label ids only: drop the intent names into CLINC150/intents.txt (one per line, in label-id order) to make the options readable
- **fever**: the provided files carry no wiki sentences, so this is claim-only (closed-book) verification no evidence text: state = claim
- **hwu64**: a many-label closed-set task; the readout needs the slot head label ids only: drop the intent names into HWU64/intents.txt (one per line, in label-id order) to make the options readable
- **intentgrasp**: options ship with the data, so no label-name lookup is needed
- **jigsaw**: test split is a deterministic 20% hold-out of train.csv
- **mnli**: test = dev_matched, test_ood = dev_mismatched (out of domain)
- **snli**: 5 hypotheses share one premise, so this is the shared-state showcase
- **truthfulqa**: MC reading of the set, not the official MC1/MC2 metrics

## Failures

- `clinc150/qwen_zeroshot`: ValueError: question 'intent' has 150 options but the 'reserved_label' readout supports at most 26
- `hwu64/qwen_zeroshot`: ValueError: question 'intent' has 64 options but the 'reserved_label' readout supports at most 26
