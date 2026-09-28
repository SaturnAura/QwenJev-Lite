# Dataset results

- items per task: all
- variants: `laya`, `qwen_zeroshot`, `qwen_trained`

## Accuracy and calibration

| task | type | n | majority | laya acc | qwen_zeroshot acc | qwen_trained acc | laya ECE | qwen_zeroshot ECE | qwen_trained ECE |
|---|---|---|---|---|---|---|---|---|---|
| arguana_rel_bool | bool | 300 | 0.800 | 0.450 | 0.330 | 0.200 | 0.345 | 0.250 | 0.797 |
| arguana_rel_choice | choice | 60 | 0.367 | 0.650 | 0.900 | 0.600 | 0.115 | 0.189 | 0.322 |
| arguana_rel_score | score | 300 | 0.800 | 0.190 | 0.647 | 0.800 | 0.558 | 0.153 | 0.192 |
| banking77 | choice | 400 | 0.025 | 0.247 | err | 0.040 | 0.589 | err | 0.853 |
| banking77_top15 | choice | 150 | 0.067 | 0.507 | 0.587 | 0.273 | 0.409 | 0.097 | 0.635 |
| clinc150 | choice | 400 | 0.013 | 0.512 | err | 0.013 | 0.343 | err | 0.831 |
| clinc150_top15 | choice | 75 | 0.067 | 0.893 | 0.893 | 0.453 | 0.088 | 0.193 | 0.444 |
| fever | choice | 400 | 0.352 | 0.307 | 0.415 | 0.432 | 0.674 | 0.329 | 0.506 |
| fever_support | score | 400 | 0.352 | 0.355 | 0.515 | 0.480 | 0.224 | 0.164 | 0.432 |
| goemotions | bool | 11200 | 0.958 | 0.907 | 0.722 | 0.249 | 0.013 | 0.101 | 0.634 |
| goemotions_sentiment | score | 400 | 0.400 | 0.425 | 0.605 | 0.420 | 0.385 | 0.103 | 0.465 |
| hwu64 | choice | 400 | 0.025 | 0.270 | err | 0.075 | 0.389 | err | 0.890 |
| hwu64_top15 | choice | 150 | 0.067 | 0.600 | 0.613 | 0.300 | 0.267 | 0.141 | 0.624 |
| intentgrasp | choice | 400 | 0.212 | 0.315 | 0.323 | 0.310 | 0.214 | 0.266 | 0.595 |
| jigsaw | bool | 2400 | 0.962 | 0.897 | 0.869 | 0.619 | 0.039 | 0.148 | 0.245 |
| jigsaw_severity | score | 400 | 0.905 | 0.385 | 0.860 | 0.917 | 0.153 | 0.067 | 0.062 |
| mmlu_pro | choice | 400 | 0.142 | 0.133 | 0.393 | 0.265 | 0.209 | 0.145 | 0.646 |
| mnli | choice | 1178 | 0.366 | 0.582 | 0.792 | 0.733 | 0.198 | 0.073 | 0.177 |
| nfcorpus_rel_bool | bool | 355 | 0.676 | 0.349 | 0.701 | 0.330 | 0.509 | 0.130 | 0.636 |
| nfcorpus_rel_choice | choice | 60 | 0.283 | 0.333 | 0.617 | 0.467 | 0.215 | 0.132 | 0.450 |
| nfcorpus_rel_score | score | 355 | 0.676 | 0.262 | 0.555 | 0.673 | 0.412 | 0.113 | 0.265 |
| scidocs_rel_bool | bool | 360 | 0.667 | 0.361 | 0.800 | 0.350 | 0.461 | 0.224 | 0.565 |
| scidocs_rel_choice | choice | 60 | 0.367 | 0.467 | 0.950 | 0.633 | 0.191 | 0.278 | 0.302 |
| scidocs_rel_score | score | 360 | 0.667 | 0.217 | 0.622 | 0.700 | 0.448 | 0.220 | 0.243 |
| scifact_rel_bool | bool | 306 | 0.784 | 0.258 | 0.889 | 0.216 | 0.598 | 0.298 | 0.748 |
| scifact_rel_choice | choice | 60 | 0.383 | 0.417 | 0.967 | 0.683 | 0.174 | 0.199 | 0.222 |
| scifact_rel_score | score | 306 | 0.784 | 0.144 | 0.781 | 0.817 | 0.513 | 0.199 | 0.157 |
| snli | choice | 1184 | 0.382 | 0.617 | 0.861 | 0.828 | 0.131 | 0.070 | 0.104 |
| trec-covid_rel_bool | bool | 300 | 0.667 | 0.420 | 0.707 | 0.333 | 0.412 | 0.136 | 0.655 |
| trec-covid_rel_choice | choice | 50 | 0.420 | 0.420 | 0.640 | 0.320 | 0.155 | 0.145 | 0.634 |
| trec-covid_rel_score | score | 300 | 0.667 | 0.187 | 0.697 | 0.667 | 0.459 | 0.109 | 0.329 |
| truthfulqa | choice | 240 | 0.237 | 0.242 | 0.604 | 0.433 | 0.276 | 0.070 | 0.470 |
| vihealthqa_rel_bool | bool | 359 | 0.669 | 0.379 | 0.599 | 0.432 | 0.425 | 0.017 | 0.369 |
| vihealthqa_rel_choice | choice | 60 | 0.350 | 0.233 | 0.233 | 0.233 | 0.292 | 0.429 | 0.701 |
| vihealthqa_rel_score | score | 359 | 0.669 | 0.270 | 0.666 | 0.696 | 0.355 | 0.196 | 0.287 |
| mnli_ood | choice | 1186 | 0.378 | 0.596 | 0.824 | 0.728 | 0.183 | 0.093 | 0.181 |

## Speed

| task | variant | requests | ms/request | decisions/request |
|---|---|---|---|---|
| arguana_rel_bool | laya | 300 | 26.0 | 5.00 |
| arguana_rel_bool | qwen_zeroshot | 60 | 574.3 | 5.00 |
| arguana_rel_bool | qwen_trained | 60 | 572.1 | 5.00 |
| arguana_rel_choice | laya | 60 | 25.5 | 1.00 |
| arguana_rel_choice | qwen_zeroshot | 60 | 146.9 | 1.00 |
| arguana_rel_choice | qwen_trained | 60 | 148.7 | 1.00 |
| arguana_rel_score | laya | 300 | 24.4 | 5.00 |
| arguana_rel_score | qwen_zeroshot | 60 | 584.1 | 5.00 |
| arguana_rel_score | qwen_trained | 60 | 582.6 | 5.00 |
| banking77 | laya | 400 | 28.7 | 1.00 |
| banking77 | qwen_trained | 400 | 145.9 | 1.00 |
| banking77_top15 | laya | 150 | 24.5 | 1.00 |
| banking77_top15 | qwen_zeroshot | 150 | 46.1 | 1.00 |
| banking77_top15 | qwen_trained | 150 | 43.7 | 1.00 |
| clinc150 | laya | 400 | 29.6 | 1.00 |
| clinc150 | qwen_trained | 400 | 205.1 | 1.00 |
| clinc150_top15 | laya | 75 | 24.3 | 1.00 |
| clinc150_top15 | qwen_zeroshot | 75 | 41.0 | 1.00 |
| clinc150_top15 | qwen_trained | 75 | 38.1 | 1.00 |
| fever | laya | 400 | 23.5 | 1.00 |
| fever | qwen_zeroshot | 400 | 33.3 | 1.00 |
| fever | qwen_trained | 400 | 32.9 | 1.00 |
| fever_support | laya | 400 | 23.0 | 1.00 |
| fever_support | qwen_zeroshot | 400 | 27.2 | 1.00 |
| fever_support | qwen_trained | 400 | 26.6 | 1.00 |
| goemotions | laya | 11200 | 22.3 | 28.00 |
| goemotions | qwen_zeroshot | 400 | 491.7 | 28.00 |
| goemotions | qwen_trained | 400 | 472.9 | 28.00 |
| goemotions_sentiment | laya | 400 | 21.4 | 1.00 |
| goemotions_sentiment | qwen_zeroshot | 400 | 28.9 | 1.00 |
| goemotions_sentiment | qwen_trained | 400 | 27.9 | 1.00 |
| hwu64 | laya | 400 | 24.8 | 1.00 |
| hwu64 | qwen_trained | 400 | 100.6 | 1.00 |
| hwu64_top15 | laya | 150 | 23.5 | 1.00 |
| hwu64_top15 | qwen_zeroshot | 150 | 41.6 | 1.00 |
| hwu64_top15 | qwen_trained | 150 | 39.3 | 1.00 |
| intentgrasp | laya | 400 | 23.6 | 1.00 |
| intentgrasp | qwen_zeroshot | 400 | 98.8 | 1.00 |
| intentgrasp | qwen_trained | 400 | 97.3 | 1.00 |
| jigsaw | laya | 2400 | 22.9 | 6.00 |
| jigsaw | qwen_zeroshot | 400 | 251.2 | 6.00 |
| jigsaw | qwen_trained | 400 | 248.3 | 6.00 |
| jigsaw_severity | laya | 400 | 22.4 | 1.00 |
| jigsaw_severity | qwen_zeroshot | 400 | 83.8 | 1.00 |
| jigsaw_severity | qwen_trained | 400 | 85.0 | 1.00 |
| mmlu_pro | laya | 400 | 25.3 | 1.00 |
| mmlu_pro | qwen_zeroshot | 400 | 100.9 | 1.00 |
| mmlu_pro | qwen_trained | 400 | 98.7 | 1.00 |
| mnli | laya | 1178 | 24.5 | 2.94 |
| mnli | qwen_zeroshot | 400 | 115.0 | 2.94 |
| mnli | qwen_trained | 400 | 113.6 | 2.94 |
| nfcorpus_rel_bool | laya | 355 | 25.5 | 5.92 |
| nfcorpus_rel_bool | qwen_zeroshot | 60 | 251.2 | 5.92 |
| nfcorpus_rel_bool | qwen_trained | 60 | 251.7 | 5.92 |
| nfcorpus_rel_choice | laya | 60 | 25.1 | 1.00 |
| nfcorpus_rel_choice | qwen_zeroshot | 60 | 55.2 | 1.00 |
| nfcorpus_rel_choice | qwen_trained | 60 | 54.6 | 1.00 |
| nfcorpus_rel_score | laya | 355 | 24.6 | 5.92 |
| nfcorpus_rel_score | qwen_zeroshot | 60 | 264.4 | 5.92 |
| nfcorpus_rel_score | qwen_trained | 60 | 259.6 | 5.92 |
| scidocs_rel_bool | laya | 360 | 26.2 | 6.00 |
| scidocs_rel_bool | qwen_zeroshot | 60 | 259.2 | 6.00 |
| scidocs_rel_bool | qwen_trained | 60 | 255.6 | 6.00 |
| scidocs_rel_choice | laya | 60 | 25.9 | 1.00 |
| scidocs_rel_choice | qwen_zeroshot | 60 | 55.1 | 1.00 |
| scidocs_rel_choice | qwen_trained | 60 | 54.4 | 1.00 |
| scidocs_rel_score | laya | 360 | 25.8 | 6.00 |
| scidocs_rel_score | qwen_zeroshot | 60 | 270.7 | 6.00 |
| scidocs_rel_score | qwen_trained | 60 | 267.5 | 6.00 |
| scifact_rel_bool | laya | 306 | 24.8 | 5.10 |
| scifact_rel_bool | qwen_zeroshot | 60 | 238.0 | 5.10 |
| scifact_rel_bool | qwen_trained | 60 | 234.2 | 5.10 |
| scifact_rel_choice | laya | 60 | 29.0 | 1.00 |
| scifact_rel_choice | qwen_zeroshot | 60 | 59.0 | 1.00 |
| scifact_rel_choice | qwen_trained | 60 | 58.3 | 1.00 |
| scifact_rel_score | laya | 306 | 24.8 | 5.10 |
| scifact_rel_score | qwen_zeroshot | 60 | 247.3 | 5.10 |
| scifact_rel_score | qwen_trained | 60 | 243.3 | 5.10 |
| snli | laya | 1184 | 24.6 | 2.96 |
| snli | qwen_zeroshot | 400 | 91.4 | 2.96 |
| snli | qwen_trained | 400 | 90.0 | 2.96 |
| trec-covid_rel_bool | laya | 300 | 25.2 | 6.00 |
| trec-covid_rel_bool | qwen_zeroshot | 50 | 264.3 | 6.00 |
| trec-covid_rel_bool | qwen_trained | 50 | 259.8 | 6.00 |
| trec-covid_rel_choice | laya | 50 | 28.4 | 1.00 |
| trec-covid_rel_choice | qwen_zeroshot | 50 | 59.4 | 1.00 |
| trec-covid_rel_choice | qwen_trained | 50 | 58.6 | 1.00 |
| trec-covid_rel_score | laya | 300 | 24.6 | 6.00 |
| trec-covid_rel_score | qwen_zeroshot | 50 | 272.6 | 6.00 |
| trec-covid_rel_score | qwen_trained | 50 | 268.4 | 6.00 |
| truthfulqa | laya | 240 | 24.5 | 1.00 |
| truthfulqa | qwen_zeroshot | 240 | 41.2 | 1.00 |
| truthfulqa | qwen_trained | 240 | 40.4 | 1.00 |
| vihealthqa_rel_bool | laya | 359 | 24.6 | 5.98 |
| vihealthqa_rel_bool | qwen_zeroshot | 60 | 316.6 | 5.98 |
| vihealthqa_rel_bool | qwen_trained | 60 | 304.9 | 5.98 |
| vihealthqa_rel_choice | laya | 60 | 24.3 | 1.00 |
| vihealthqa_rel_choice | qwen_zeroshot | 60 | 78.2 | 1.00 |
| vihealthqa_rel_choice | qwen_trained | 60 | 78.3 | 1.00 |
| vihealthqa_rel_score | laya | 359 | 24.4 | 5.98 |
| vihealthqa_rel_score | qwen_zeroshot | 60 | 315.7 | 5.98 |
| vihealthqa_rel_score | qwen_trained | 60 | 311.9 | 5.98 |
| mnli_ood | laya | 1186 | 24.4 | 2.96 |
| mnli_ood | qwen_zeroshot | 400 | 117.7 | 2.96 |
| mnli_ood | qwen_trained | 400 | 116.2 | 2.96 |
