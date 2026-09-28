# Dataset results

- items per task: all
- variants: `qwen_trained`

## Accuracy and calibration

| task | type | n | majority | qwen_trained acc | qwen_trained ECE |
|---|---|---|---|---|---|
| arguana_rel_bool | bool | 300 | 0.800 | 0.783 | 0.096 |
| arguana_rel_choice | choice | 60 | 0.367 | 0.917 | 0.065 |
| arguana_rel_score | score | 300 | 0.800 | 0.817 | 0.133 |
| banking77 | choice | 400 | 0.025 | 0.182 | 0.390 |
| banking77_top15 | choice | 150 | 0.067 | 0.627 | 0.140 |
| clinc150 | choice | 400 | 0.013 | 0.085 | 0.525 |
| clinc150_top15 | choice | 75 | 0.067 | 0.907 | 0.163 |
| fever | choice | 400 | 0.352 | 0.415 | 0.448 |
| fever_support | score | 400 | 0.352 | 0.537 | 0.197 |
| goemotions | bool | 11200 | 0.958 | 0.862 | 0.031 |
| goemotions_sentiment | score | 400 | 0.400 | 0.573 | 0.174 |
| hwu64 | choice | 400 | 0.025 | 0.278 | 0.416 |
| hwu64_top15 | choice | 150 | 0.067 | 0.673 | 0.128 |
| intentgrasp | choice | 400 | 0.212 | 0.273 | 0.471 |
| jigsaw | bool | 2400 | 0.962 | 0.924 | 0.034 |
| jigsaw_severity | score | 400 | 0.905 | 0.930 | 0.020 |
| mmlu_pro | choice | 400 | 0.142 | 0.405 | 0.270 |
| mnli | choice | 1178 | 0.366 | 0.796 | 0.019 |
| nfcorpus_rel_bool | bool | 355 | 0.676 | 0.730 | 0.077 |
| nfcorpus_rel_choice | choice | 60 | 0.283 | 0.583 | 0.200 |
| nfcorpus_rel_score | score | 355 | 0.676 | 0.628 | 0.122 |
| scidocs_rel_bool | bool | 360 | 0.667 | 0.778 | 0.098 |
| scidocs_rel_choice | choice | 60 | 0.367 | 0.883 | 0.098 |
| scidocs_rel_score | score | 360 | 0.667 | 0.781 | 0.058 |
| scifact_rel_bool | bool | 306 | 0.784 | 0.915 | 0.213 |
| scifact_rel_choice | choice | 60 | 0.383 | 0.917 | 0.071 |
| scifact_rel_score | score | 306 | 0.784 | 0.944 | 0.089 |
| snli | choice | 1184 | 0.382 | 0.859 | 0.016 |
| trec-covid_rel_bool | bool | 300 | 0.667 | 0.753 | 0.116 |
| trec-covid_rel_choice | choice | 50 | 0.420 | 0.780 | 0.079 |
| trec-covid_rel_score | score | 300 | 0.667 | 0.687 | 0.181 |
| truthfulqa | choice | 240 | 0.237 | 0.608 | 0.137 |
| vihealthqa_rel_bool | bool | 359 | 0.669 | 0.682 | 0.040 |
| vihealthqa_rel_choice | choice | 60 | 0.350 | 0.250 | 0.500 |
| vihealthqa_rel_score | score | 359 | 0.669 | 0.708 | 0.194 |
| mnli_ood | choice | 1186 | 0.378 | 0.807 | 0.015 |

## Speed

| task | variant | requests | ms/request | decisions/request |
|---|---|---|---|---|
| arguana_rel_bool | qwen_trained | 60 | 560.0 | 5.00 |
| arguana_rel_choice | qwen_trained | 60 | 142.3 | 1.00 |
| arguana_rel_score | qwen_trained | 60 | 564.4 | 5.00 |
| banking77 | qwen_trained | 400 | 141.0 | 1.00 |
| banking77_top15 | qwen_trained | 150 | 42.0 | 1.00 |
| clinc150 | qwen_trained | 400 | 197.5 | 1.00 |
| clinc150_top15 | qwen_trained | 75 | 37.6 | 1.00 |
| fever | qwen_trained | 400 | 31.5 | 1.00 |
| fever_support | qwen_trained | 400 | 25.5 | 1.00 |
| goemotions | qwen_trained | 400 | 460.0 | 28.00 |
| goemotions_sentiment | qwen_trained | 400 | 28.0 | 1.00 |
| hwu64 | qwen_trained | 400 | 99.9 | 1.00 |
| hwu64_top15 | qwen_trained | 150 | 38.9 | 1.00 |
| intentgrasp | qwen_trained | 400 | 96.5 | 1.00 |
| jigsaw | qwen_trained | 400 | 248.3 | 6.00 |
| jigsaw_severity | qwen_trained | 400 | 82.9 | 1.00 |
| mmlu_pro | qwen_trained | 400 | 98.4 | 1.00 |
| mnli | qwen_trained | 400 | 113.9 | 2.94 |
| nfcorpus_rel_bool | qwen_trained | 60 | 249.9 | 5.92 |
| nfcorpus_rel_choice | qwen_trained | 60 | 54.8 | 1.00 |
| nfcorpus_rel_score | qwen_trained | 60 | 259.7 | 5.92 |
| scidocs_rel_bool | qwen_trained | 60 | 256.8 | 6.00 |
| scidocs_rel_choice | qwen_trained | 60 | 54.6 | 1.00 |
| scidocs_rel_score | qwen_trained | 60 | 268.7 | 6.00 |
| scifact_rel_bool | qwen_trained | 60 | 236.7 | 5.10 |
| scifact_rel_choice | qwen_trained | 60 | 58.6 | 1.00 |
| scifact_rel_score | qwen_trained | 60 | 245.2 | 5.10 |
| snli | qwen_trained | 400 | 90.7 | 2.96 |
| trec-covid_rel_bool | qwen_trained | 50 | 261.5 | 6.00 |
| trec-covid_rel_choice | qwen_trained | 50 | 58.9 | 1.00 |
| trec-covid_rel_score | qwen_trained | 50 | 271.5 | 6.00 |
| truthfulqa | qwen_trained | 240 | 40.7 | 1.00 |
| vihealthqa_rel_bool | qwen_trained | 60 | 307.3 | 5.98 |
| vihealthqa_rel_choice | qwen_trained | 60 | 78.1 | 1.00 |
| vihealthqa_rel_score | qwen_trained | 60 | 314.4 | 5.98 |
| mnli_ood | qwen_trained | 400 | 116.7 | 2.96 |
