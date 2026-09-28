# Dataset results

- items per task: all
- variants: `laya`, `qwen_zeroshot`, `qwen_trained`

## Accuracy and calibration

| task | type | n | majority | laya acc | qwen_zeroshot acc | qwen_trained acc | laya ECE | qwen_zeroshot ECE | qwen_trained ECE |
|---|---|---|---|---|---|---|---|---|---|
| arguana_rel_bool | bool | 300 | 0.800 | 0.450 | 0.330 | 0.263 | 0.345 | 0.250 | 0.571 |
| arguana_rel_choice | choice | 60 | 0.367 | 0.650 | 0.900 | 0.950 | 0.115 | 0.189 | 0.038 |
| arguana_rel_score | score | 300 | 0.800 | 0.190 | 0.647 | 0.810 | 0.558 | 0.153 | 0.138 |
| banking77 | choice | 400 | 0.025 | 0.247 | err | 0.060 | 0.589 | err | 0.785 |
| banking77_top15 | choice | 150 | 0.067 | 0.507 | 0.587 | 0.447 | 0.409 | 0.097 | 0.275 |
| clinc150 | choice | 400 | 0.013 | 0.512 | err | 0.045 | 0.343 | err | 0.789 |
| clinc150_top15 | choice | 75 | 0.067 | 0.893 | 0.893 | 0.533 | 0.088 | 0.193 | 0.260 |
| fever | choice | 400 | 0.352 | 0.307 | 0.415 | 0.445 | 0.674 | 0.329 | 0.426 |
| fever_support | score | 400 | 0.352 | 0.355 | 0.515 | 0.458 | 0.224 | 0.164 | 0.427 |
| goemotions | bool | 11200 | 0.958 | 0.907 | 0.722 | 0.877 | 0.013 | 0.101 | 0.056 |
| goemotions_sentiment | score | 400 | 0.400 | 0.425 | 0.605 | 0.605 | 0.385 | 0.103 | 0.241 |
| hwu64 | choice | 400 | 0.025 | 0.270 | err | 0.092 | 0.389 | err | 0.812 |
| hwu64_top15 | choice | 150 | 0.067 | 0.600 | 0.613 | 0.333 | 0.267 | 0.141 | 0.486 |
| intentgrasp | choice | 400 | 0.212 | 0.315 | 0.323 | 0.307 | 0.214 | 0.266 | 0.547 |
| jigsaw | bool | 2400 | 0.962 | 0.897 | 0.869 | 0.907 | 0.039 | 0.148 | 0.028 |
| jigsaw_severity | score | 400 | 0.905 | 0.385 | 0.860 | 0.935 | 0.153 | 0.067 | 0.036 |
| mmlu_pro | choice | 400 | 0.142 | 0.133 | 0.393 | 0.318 | 0.209 | 0.145 | 0.492 |
| mnli | choice | 1178 | 0.366 | 0.582 | 0.792 | 0.806 | 0.198 | 0.073 | 0.091 |
| nfcorpus_rel_bool | bool | 355 | 0.676 | 0.349 | 0.701 | 0.454 | 0.509 | 0.130 | 0.300 |
| nfcorpus_rel_choice | choice | 60 | 0.283 | 0.333 | 0.617 | 0.683 | 0.215 | 0.132 | 0.211 |
| nfcorpus_rel_score | score | 355 | 0.676 | 0.262 | 0.555 | 0.659 | 0.412 | 0.113 | 0.190 |
| scidocs_rel_bool | bool | 360 | 0.667 | 0.361 | 0.800 | 0.586 | 0.461 | 0.224 | 0.161 |
| scidocs_rel_choice | choice | 60 | 0.367 | 0.467 | 0.950 | 0.917 | 0.191 | 0.278 | 0.033 |
| scidocs_rel_score | score | 360 | 0.667 | 0.217 | 0.622 | 0.708 | 0.448 | 0.220 | 0.208 |
| scifact_rel_bool | bool | 306 | 0.784 | 0.258 | 0.889 | 0.471 | 0.598 | 0.298 | 0.259 |
| scifact_rel_choice | choice | 60 | 0.383 | 0.417 | 0.967 | 0.983 | 0.174 | 0.199 | 0.021 |
| scifact_rel_score | score | 306 | 0.784 | 0.144 | 0.781 | 0.856 | 0.513 | 0.199 | 0.085 |
| snli | choice | 1184 | 0.382 | 0.617 | 0.861 | 0.888 | 0.131 | 0.070 | 0.040 |
| trec-covid_rel_bool | bool | 300 | 0.667 | 0.420 | 0.707 | 0.350 | 0.412 | 0.136 | 0.514 |
| trec-covid_rel_choice | choice | 50 | 0.420 | 0.420 | 0.640 | 0.840 | 0.155 | 0.145 | 0.145 |
| trec-covid_rel_score | score | 300 | 0.667 | 0.187 | 0.697 | 0.670 | 0.459 | 0.109 | 0.304 |
| truthfulqa | choice | 240 | 0.237 | 0.242 | 0.604 | 0.575 | 0.276 | 0.070 | 0.276 |
| vihealthqa_rel_bool | bool | 359 | 0.669 | 0.379 | 0.599 | 0.396 | 0.425 | 0.017 | 0.342 |
| vihealthqa_rel_choice | choice | 60 | 0.350 | 0.233 | 0.233 | 0.267 | 0.292 | 0.429 | 0.553 |
| vihealthqa_rel_score | score | 359 | 0.669 | 0.270 | 0.666 | 0.705 | 0.355 | 0.196 | 0.267 |
| mnli_ood | choice | 1186 | 0.378 | 0.596 | 0.824 | 0.797 | 0.183 | 0.093 | 0.085 |

## Speed

| task | variant | requests | ms/request | decisions/request |
|---|---|---|---|---|
| arguana_rel_bool | laya | 300 | 26.0 | 5.00 |
| arguana_rel_bool | qwen_zeroshot | 60 | 574.3 | 5.00 |
| arguana_rel_bool | qwen_trained | 60 | 567.9 | 5.00 |
| arguana_rel_choice | laya | 60 | 25.5 | 1.00 |
| arguana_rel_choice | qwen_zeroshot | 60 | 146.9 | 1.00 |
| arguana_rel_choice | qwen_trained | 60 | 143.9 | 1.00 |
| arguana_rel_score | laya | 300 | 24.4 | 5.00 |
| arguana_rel_score | qwen_zeroshot | 60 | 584.1 | 5.00 |
| arguana_rel_score | qwen_trained | 60 | 573.2 | 5.00 |
| banking77 | laya | 400 | 28.7 | 1.00 |
| banking77 | qwen_trained | 400 | 143.2 | 1.00 |
| banking77_top15 | laya | 150 | 24.5 | 1.00 |
| banking77_top15 | qwen_zeroshot | 150 | 46.1 | 1.00 |
| banking77_top15 | qwen_trained | 150 | 42.9 | 1.00 |
| clinc150 | laya | 400 | 29.6 | 1.00 |
| clinc150 | qwen_trained | 400 | 224.7 | 1.00 |
| clinc150_top15 | laya | 75 | 24.3 | 1.00 |
| clinc150_top15 | qwen_zeroshot | 75 | 41.0 | 1.00 |
| clinc150_top15 | qwen_trained | 75 | 38.2 | 1.00 |
| fever | laya | 400 | 23.5 | 1.00 |
| fever | qwen_zeroshot | 400 | 33.3 | 1.00 |
| fever | qwen_trained | 400 | 34.0 | 1.00 |
| fever_support | laya | 400 | 23.0 | 1.00 |
| fever_support | qwen_zeroshot | 400 | 27.2 | 1.00 |
| fever_support | qwen_trained | 400 | 30.9 | 1.00 |
| goemotions | laya | 11200 | 22.3 | 28.00 |
| goemotions | qwen_zeroshot | 400 | 491.7 | 28.00 |
| goemotions | qwen_trained | 400 | 478.9 | 28.00 |
| goemotions_sentiment | laya | 400 | 21.4 | 1.00 |
| goemotions_sentiment | qwen_zeroshot | 400 | 28.9 | 1.00 |
| goemotions_sentiment | qwen_trained | 400 | 28.1 | 1.00 |
| hwu64 | laya | 400 | 24.8 | 1.00 |
| hwu64 | qwen_trained | 400 | 102.4 | 1.00 |
| hwu64_top15 | laya | 150 | 23.5 | 1.00 |
| hwu64_top15 | qwen_zeroshot | 150 | 41.6 | 1.00 |
| hwu64_top15 | qwen_trained | 150 | 39.9 | 1.00 |
| intentgrasp | laya | 400 | 23.6 | 1.00 |
| intentgrasp | qwen_zeroshot | 400 | 98.8 | 1.00 |
| intentgrasp | qwen_trained | 400 | 97.4 | 1.00 |
| jigsaw | laya | 2400 | 22.9 | 6.00 |
| jigsaw | qwen_zeroshot | 400 | 251.2 | 6.00 |
| jigsaw | qwen_trained | 400 | 267.2 | 6.00 |
| jigsaw_severity | laya | 400 | 22.4 | 1.00 |
| jigsaw_severity | qwen_zeroshot | 400 | 83.8 | 1.00 |
| jigsaw_severity | qwen_trained | 400 | 87.2 | 1.00 |
| mmlu_pro | laya | 400 | 25.3 | 1.00 |
| mmlu_pro | qwen_zeroshot | 400 | 100.9 | 1.00 |
| mmlu_pro | qwen_trained | 400 | 99.1 | 1.00 |
| mnli | laya | 1178 | 24.5 | 2.94 |
| mnli | qwen_zeroshot | 400 | 115.0 | 2.94 |
| mnli | qwen_trained | 400 | 116.0 | 2.94 |
| nfcorpus_rel_bool | laya | 355 | 25.5 | 5.92 |
| nfcorpus_rel_bool | qwen_zeroshot | 60 | 251.2 | 5.92 |
| nfcorpus_rel_bool | qwen_trained | 60 | 270.7 | 5.92 |
| nfcorpus_rel_choice | laya | 60 | 25.1 | 1.00 |
| nfcorpus_rel_choice | qwen_zeroshot | 60 | 55.2 | 1.00 |
| nfcorpus_rel_choice | qwen_trained | 60 | 64.7 | 1.00 |
| nfcorpus_rel_score | laya | 355 | 24.6 | 5.92 |
| nfcorpus_rel_score | qwen_zeroshot | 60 | 264.4 | 5.92 |
| nfcorpus_rel_score | qwen_trained | 60 | 306.8 | 5.92 |
| scidocs_rel_bool | laya | 360 | 26.2 | 6.00 |
| scidocs_rel_bool | qwen_zeroshot | 60 | 259.2 | 6.00 |
| scidocs_rel_bool | qwen_trained | 60 | 302.8 | 6.00 |
| scidocs_rel_choice | laya | 60 | 25.9 | 1.00 |
| scidocs_rel_choice | qwen_zeroshot | 60 | 55.1 | 1.00 |
| scidocs_rel_choice | qwen_trained | 60 | 67.9 | 1.00 |
| scidocs_rel_score | laya | 360 | 25.8 | 6.00 |
| scidocs_rel_score | qwen_zeroshot | 60 | 270.7 | 6.00 |
| scidocs_rel_score | qwen_trained | 60 | 316.6 | 6.00 |
| scifact_rel_bool | laya | 306 | 24.8 | 5.10 |
| scifact_rel_bool | qwen_zeroshot | 60 | 238.0 | 5.10 |
| scifact_rel_bool | qwen_trained | 60 | 278.3 | 5.10 |
| scifact_rel_choice | laya | 60 | 29.0 | 1.00 |
| scifact_rel_choice | qwen_zeroshot | 60 | 59.0 | 1.00 |
| scifact_rel_choice | qwen_trained | 60 | 69.3 | 1.00 |
| scifact_rel_score | laya | 306 | 24.8 | 5.10 |
| scifact_rel_score | qwen_zeroshot | 60 | 247.3 | 5.10 |
| scifact_rel_score | qwen_trained | 60 | 288.6 | 5.10 |
| snli | laya | 1184 | 24.6 | 2.96 |
| snli | qwen_zeroshot | 400 | 91.4 | 2.96 |
| snli | qwen_trained | 400 | 97.6 | 2.96 |
| trec-covid_rel_bool | laya | 300 | 25.2 | 6.00 |
| trec-covid_rel_bool | qwen_zeroshot | 50 | 264.3 | 6.00 |
| trec-covid_rel_bool | qwen_trained | 50 | 257.3 | 6.00 |
| trec-covid_rel_choice | laya | 50 | 28.4 | 1.00 |
| trec-covid_rel_choice | qwen_zeroshot | 50 | 59.4 | 1.00 |
| trec-covid_rel_choice | qwen_trained | 50 | 58.6 | 1.00 |
| trec-covid_rel_score | laya | 300 | 24.6 | 6.00 |
| trec-covid_rel_score | qwen_zeroshot | 50 | 272.6 | 6.00 |
| trec-covid_rel_score | qwen_trained | 50 | 267.1 | 6.00 |
| truthfulqa | laya | 240 | 24.5 | 1.00 |
| truthfulqa | qwen_zeroshot | 240 | 41.2 | 1.00 |
| truthfulqa | qwen_trained | 240 | 39.8 | 1.00 |
| vihealthqa_rel_bool | laya | 359 | 24.6 | 5.98 |
| vihealthqa_rel_bool | qwen_zeroshot | 60 | 316.6 | 5.98 |
| vihealthqa_rel_bool | qwen_trained | 60 | 320.5 | 5.98 |
| vihealthqa_rel_choice | laya | 60 | 24.3 | 1.00 |
| vihealthqa_rel_choice | qwen_zeroshot | 60 | 78.2 | 1.00 |
| vihealthqa_rel_choice | qwen_trained | 60 | 83.2 | 1.00 |
| vihealthqa_rel_score | laya | 359 | 24.4 | 5.98 |
| vihealthqa_rel_score | qwen_zeroshot | 60 | 315.7 | 5.98 |
| vihealthqa_rel_score | qwen_trained | 60 | 335.3 | 5.98 |
| mnli_ood | laya | 1186 | 24.4 | 2.96 |
| mnli_ood | qwen_zeroshot | 400 | 117.7 | 2.96 |
| mnli_ood | qwen_trained | 400 | 147.3 | 2.96 |
