# QwenJev-lite reproduction report

| field | value |
|---|---|
| backbone | C:\qwen3.5-4B |
| readout (probes) | reserved_label |
| readout (training) | slot_head |
| device | cuda:0 |
| limits | <= 255 options, 32768 per branch, 65536 per request |

## The essay's opening example

```json
{
  "results": {
    "queue": {
      "type": "choice",
      "probabilities": {
        "payments": 0.95108,
        "account": 0.005083,
        "other": 0.043837
      },
      "answer": "payments",
      "confidence": 0.92662
    },
    "escalate": {
      "type": "bool",
      "probabilities": {
        "yes": 0.931245,
        "no": 0.068755
      },
      "answer": true,
      "probability": 0.931245,
      "confidence": 0.86249
    }
  },
  "usage": {
    "state_tokens": 37,
    "question_tokens": 84,
    "input_tokens": 121,
    "output_tokens": 37,
    "questions": 2,
    "branches": 2
  },
  "timing": {
    "prefill_ms": 565.71,
    "branch_ms": 304.92,
    "total_ms": 871.87,
    "state_cache_hit": false,
    "shared_state": true,
    "readout": "reserved_label"
  }
}
```

## Probes

### visibility

| condition | mean p(secret) | max p(secret) | n |
|---|---|---|---|
| secret_in_sibling | 0.203 | 0.203 | 5 |
| secret_absent | 0.203 | 0.203 | 5 |
| secret_in_state | 0.740 | 0.740 | 5 |

### reference_card

| condition | correct | n | mean p(correct) |
|---|---|---|---|
| card in options: first | 0 | 4 | 0.180 |
| card in options: middle | 2 | 4 | 0.457 |
| card in options: last | 0 | 4 | 0.203 |
| card in shared state | 12 | 12 | 0.891 |

### option_interaction

| quantity | value |
|---|---|
| mean log-odds, 4 options | -3.034 |
| mean log-odds, 5 options | -3.343 |
| mean change | -0.309 |
| 95% paired interval | [-0.309, -0.309] |
| decreased in all blocks | True |
| duplicate-request control (noise) | 0.000 |
| mean change with a different extra option | -0.060 |
| blocks | 10 |

### option_order

| condition | mean p(payments) | mean p(top) |
|---|---|---|
| declared | 0.913 | 0.913 |
| reversed | 0.890 | 0.890 |

Mean shift in p(payments): **-0.023**

### fake_option

| condition | slots | answer | p(account) |
|---|---|---|---|
| base | 3 | account | 0.932 |
| injected | 3 | account | 0.918 |

### accounting

| quantity | value |
|---|---|
| shared state tokens | 23 |
| suffix tokens, one bool | 32 |
| suffix tokens, two identical bools | 64 |
| suffix tokens, three-option choice | 41 |
| additive | True |
| mixed request equals the sum | True |
| billing figure, short identifier | 20 |
| billing figure, long identifier | 26 |

### latency

State length sweep (one question, state cache disabled):

| state tokens | median ms |
|---|---|
| 261 | 376.0 |
| 1031 | 458.1 |
| 4097 | 1437.3 |

Question count sweep (short state):

| questions | median ms |
|---|---|
| 1 | 398.3 |
| 8 | 264.9 |
| 64 | 724.6 |
| 256 | 2898.6 |

One shared-state request vs Q separate requests:

| questions | shared ms | separate ms | speedup |
|---|---|---|---|
| 1 | 274.0 | 274.5 | 1.0 |
| 4 | 271.9 | 1108.1 | 4.07 |
| 16 | 302.9 | 4290.5 | 14.16 |

State-length medians are encoding times with the state cache disabled; question-count medians are whole requests against one cached state.

## RLCD: training the readout against outcomes

| metric | before | after RLCD | after temperature | shifted data |
|---|---|---|---|---|
| accuracy | 0.870 | 0.938 | 0.938 | 0.766 |
| mean_top_probability | 0.786 | 0.962 | 0.897 | 0.921 |
| ece | 0.094 | 0.027 | 0.047 | 0.161 |
| brier | 0.230 | 0.093 | 0.101 | 0.391 |
| nll | 0.406 | 0.224 | 0.206 | 1.011 |

- steps: 256, objective: `log_loss`, mean loss: 0.256, wall clock: 78.56s
- fitted temperature: 2.217
