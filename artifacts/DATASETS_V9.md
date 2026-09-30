# Dataset results

- items per task: all
- variants: `laya`, `qwen_zeroshot`, `qwen_trained`

## Accuracy and calibration

| task | type | n | majority | laya acc | qwen_zeroshot acc | qwen_trained acc | laya ECE | qwen_zeroshot ECE | qwen_trained ECE |
|---|---|---|---|---|---|---|---|---|---|
| arguana_rel_bool | bool | 300 | 0.800 | 0.450 | 0.330 | 0.803 | 0.345 | 0.250 | 0.025 |
| arguana_rel_choice | choice | 60 | 0.367 | 0.650 | 0.900 | 0.983 | 0.115 | 0.189 | 0.055 |
| arguana_rel_score | score | 300 | 0.800 | 0.190 | 0.647 | 0.840 | 0.558 | 0.153 | 0.125 |
| clinc150 | choice | 400 | 0.007 | 0.545 | err | 0.527 | 0.317 | err | 0.304 |
| clinc150_top15 | choice | 75 | 0.067 | 0.893 | 0.893 | 0.733 | 0.088 | 0.193 | 0.072 |
| goemotions | bool | 11200 | 0.958 | 0.907 | 0.722 | 0.958 | 0.013 | 0.101 | 0.023 |
| goemotions_sentiment | score | 400 | 0.400 | 0.425 | 0.605 | 0.578 | 0.385 | 0.103 | 0.225 |
| hwu64 | choice | 400 | 0.018 | 0.345 | err | 0.540 | 0.329 | err | 0.321 |
| hwu64_top15 | choice | 150 | 0.067 | 0.600 | 0.613 | 0.467 | 0.267 | 0.141 | 0.192 |
| intentgrasp | choice | 400 | 0.212 | 0.315 | 0.323 | 0.302 | 0.214 | 0.266 | 0.448 |
| jigsaw | bool | 2400 | 0.962 | 0.897 | 0.869 | 0.965 | 0.039 | 0.148 | 0.007 |
| jigsaw_severity | score | 400 | 0.905 | 0.385 | 0.860 | 0.915 | 0.153 | 0.067 | 0.031 |
| mnli | choice | 1178 | 0.366 | 0.582 | 0.792 | 0.782 | 0.198 | 0.073 | 0.091 |
| nfcorpus_rel_bool | bool | 355 | 0.676 | 0.349 | 0.701 | 0.685 | 0.509 | 0.130 | 0.174 |
| nfcorpus_rel_choice | choice | 60 | 0.283 | 0.333 | 0.617 | 0.733 | 0.215 | 0.132 | 0.159 |
| nfcorpus_rel_score | score | 355 | 0.676 | 0.262 | 0.555 | 0.583 | 0.412 | 0.113 | 0.195 |
| scidocs_rel_bool | bool | 360 | 0.667 | 0.361 | 0.800 | 0.669 | 0.461 | 0.224 | 0.200 |
| scidocs_rel_choice | choice | 60 | 0.367 | 0.467 | 0.950 | 0.950 | 0.191 | 0.278 | 0.076 |
| scidocs_rel_score | score | 360 | 0.667 | 0.217 | 0.622 | 0.831 | 0.448 | 0.220 | 0.069 |
| scifact_rel_bool | bool | 306 | 0.784 | 0.258 | 0.889 | 0.804 | 0.598 | 0.298 | 0.112 |
| scifact_rel_choice | choice | 60 | 0.383 | 0.417 | 0.967 | 1.000 | 0.174 | 0.199 | 0.043 |
| scifact_rel_score | score | 306 | 0.784 | 0.144 | 0.781 | 0.961 | 0.513 | 0.199 | 0.085 |
| snli | choice | 1184 | 0.382 | 0.617 | 0.861 | 0.868 | 0.131 | 0.070 | 0.057 |
| trec-covid_rel_bool | bool | 300 | 0.667 | 0.420 | 0.707 | 0.670 | 0.412 | 0.136 | 0.225 |
| trec-covid_rel_choice | choice | 50 | 0.420 | 0.420 | 0.640 | 0.900 | 0.155 | 0.145 | 0.097 |
| trec-covid_rel_score | score | 300 | 0.667 | 0.187 | 0.697 | 0.690 | 0.459 | 0.109 | 0.139 |
| truthfulqa | choice | 240 | 0.237 | 0.242 | 0.604 | 0.592 | 0.276 | 0.070 | 0.250 |
| mnli_ood | choice | 1186 | 0.378 | 0.596 | 0.824 | 0.808 | 0.183 | 0.093 | 0.073 |

## Speed

| task | variant | requests | ms/request | decisions/request |
|---|---|---|---|---|
| arguana_rel_bool | laya | 300 | 25.0 | 5.00 |
| arguana_rel_bool | qwen_zeroshot | 60 | 561.9 | 5.00 |
| arguana_rel_bool | qwen_trained | 60 | 563.1 | 5.00 |
| arguana_rel_choice | laya | 60 | 24.1 | 1.00 |
| arguana_rel_choice | qwen_zeroshot | 60 | 144.5 | 1.00 |
| arguana_rel_choice | qwen_trained | 60 | 144.0 | 1.00 |
| arguana_rel_score | laya | 300 | 25.1 | 5.00 |
| arguana_rel_score | qwen_zeroshot | 60 | 570.7 | 5.00 |
| arguana_rel_score | qwen_trained | 60 | 571.2 | 5.00 |
| clinc150 | laya | 400 | 27.7 | 1.00 |
| clinc150 | qwen_trained | 400 | 199.4 | 1.00 |
| clinc150_top15 | laya | 75 | 24.2 | 1.00 |
| clinc150_top15 | qwen_zeroshot | 75 | 39.8 | 1.00 |
| clinc150_top15 | qwen_trained | 75 | 36.8 | 1.00 |
| goemotions | laya | 11200 | 22.1 | 28.00 |
| goemotions | qwen_zeroshot | 400 | 474.3 | 28.00 |
| goemotions | qwen_trained | 400 | 465.9 | 28.00 |
| goemotions_sentiment | laya | 400 | 22.2 | 1.00 |
| goemotions_sentiment | qwen_zeroshot | 400 | 28.9 | 1.00 |
| goemotions_sentiment | qwen_trained | 400 | 27.8 | 1.00 |
| hwu64 | laya | 400 | 25.9 | 1.00 |
| hwu64 | qwen_trained | 400 | 100.3 | 1.00 |
| hwu64_top15 | laya | 150 | 24.0 | 1.00 |
| hwu64_top15 | qwen_zeroshot | 150 | 41.5 | 1.00 |
| hwu64_top15 | qwen_trained | 150 | 38.9 | 1.00 |
| intentgrasp | laya | 400 | 24.5 | 1.00 |
| intentgrasp | qwen_zeroshot | 400 | 97.8 | 1.00 |
| intentgrasp | qwen_trained | 400 | 97.2 | 1.00 |
| jigsaw | laya | 2400 | 23.6 | 6.00 |
| jigsaw | qwen_zeroshot | 400 | 250.8 | 6.00 |
| jigsaw | qwen_trained | 400 | 255.3 | 6.00 |
| jigsaw_severity | laya | 400 | 24.2 | 1.00 |
| jigsaw_severity | qwen_zeroshot | 400 | 84.3 | 1.00 |
| jigsaw_severity | qwen_trained | 400 | 83.6 | 1.00 |
| mnli | laya | 1178 | 23.7 | 2.94 |
| mnli | qwen_zeroshot | 400 | 115.3 | 2.94 |
| mnli | qwen_trained | 400 | 114.3 | 2.94 |
| nfcorpus_rel_bool | laya | 355 | 23.8 | 5.92 |
| nfcorpus_rel_bool | qwen_zeroshot | 60 | 252.5 | 5.92 |
| nfcorpus_rel_bool | qwen_trained | 60 | 249.4 | 5.92 |
| nfcorpus_rel_choice | laya | 60 | 23.5 | 1.00 |
| nfcorpus_rel_choice | qwen_zeroshot | 60 | 55.3 | 1.00 |
| nfcorpus_rel_choice | qwen_trained | 60 | 54.8 | 1.00 |
| nfcorpus_rel_score | laya | 355 | 23.8 | 5.92 |
| nfcorpus_rel_score | qwen_zeroshot | 60 | 263.0 | 5.92 |
| nfcorpus_rel_score | qwen_trained | 60 | 260.0 | 5.92 |
| scidocs_rel_bool | laya | 360 | 24.6 | 6.00 |
| scidocs_rel_bool | qwen_zeroshot | 60 | 259.1 | 6.00 |
| scidocs_rel_bool | qwen_trained | 60 | 258.1 | 6.00 |
| scidocs_rel_choice | laya | 60 | 24.1 | 1.00 |
| scidocs_rel_choice | qwen_zeroshot | 60 | 55.1 | 1.00 |
| scidocs_rel_choice | qwen_trained | 60 | 54.5 | 1.00 |
| scidocs_rel_score | laya | 360 | 24.3 | 6.00 |
| scidocs_rel_score | qwen_zeroshot | 60 | 271.5 | 6.00 |
| scidocs_rel_score | qwen_trained | 60 | 268.0 | 6.00 |
| scifact_rel_bool | laya | 306 | 24.7 | 5.10 |
| scifact_rel_bool | qwen_zeroshot | 60 | 238.0 | 5.10 |
| scifact_rel_bool | qwen_trained | 60 | 235.7 | 5.10 |
| scifact_rel_choice | laya | 60 | 24.7 | 1.00 |
| scifact_rel_choice | qwen_zeroshot | 60 | 59.1 | 1.00 |
| scifact_rel_choice | qwen_trained | 60 | 58.6 | 1.00 |
| scifact_rel_score | laya | 306 | 23.9 | 5.10 |
| scifact_rel_score | qwen_zeroshot | 60 | 248.2 | 5.10 |
| scifact_rel_score | qwen_trained | 60 | 244.9 | 5.10 |
| snli | laya | 1184 | 23.7 | 2.96 |
| snli | qwen_zeroshot | 400 | 91.9 | 2.96 |
| snli | qwen_trained | 400 | 90.7 | 2.96 |
| trec-covid_rel_bool | laya | 300 | 29.7 | 6.00 |
| trec-covid_rel_bool | qwen_zeroshot | 50 | 284.9 | 6.00 |
| trec-covid_rel_bool | qwen_trained | 50 | 282.1 | 6.00 |
| trec-covid_rel_choice | laya | 50 | 25.3 | 1.00 |
| trec-covid_rel_choice | qwen_zeroshot | 50 | 63.8 | 1.00 |
| trec-covid_rel_choice | qwen_trained | 50 | 63.3 | 1.00 |
| trec-covid_rel_score | laya | 300 | 24.8 | 6.00 |
| trec-covid_rel_score | qwen_zeroshot | 50 | 293.0 | 6.00 |
| trec-covid_rel_score | qwen_trained | 50 | 288.6 | 6.00 |
| truthfulqa | laya | 240 | 24.7 | 1.00 |
| truthfulqa | qwen_zeroshot | 240 | 41.9 | 1.00 |
| truthfulqa | qwen_trained | 240 | 40.8 | 1.00 |
| mnli_ood | laya | 1186 | 26.2 | 2.96 |
| mnli_ood | qwen_zeroshot | 400 | 118.2 | 2.96 |
| mnli_ood | qwen_trained | 400 | 117.0 | 2.96 |
