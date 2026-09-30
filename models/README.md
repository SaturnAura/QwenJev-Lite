# Readouts

Everything inference needs is in this folder, and the shipped head can be rebuilt from
the other three **bit for bit** (`torch.equal`, verified - the composed table and weights
come out identical).

| folder | rows | what it is | how it was made |
|---|---|---|---|
| `qwenjev-multitask-v2/` | 512 = 27 + 485 | **the shipped readout.** Rows 0-26 are the pretrained label rows (the fallback for an answer the head never saw); the rest are private blocks, one per `(question, label space, answer)`, 90 label spaces in total. `card.json` carries the benchmark of the run it shipped with. | `scripts/compose_wide_rows.py` |
| `shared-rows/` | 256 | the *shared* trained rows: the RLCD head blended half-way back to its initialisation. This is what every label space the reserved rows cover reads from. | `scripts/interpolate.py --checkpoint raw-trained-rows --alphas 0.5` |
| `raw-trained-rows/` | 256 | the same head before blending (alpha = 1), so the blend can be re-swept. | `train.py` (RLCD: log loss on outcomes, readout only) |
| `prototype-rows/` | 1024 | closed-form rows for label spaces **wider than the 26 reserved rows** (CLINC150 150-way, HWU64 64-way): each row is the mean state of the samples that chose that answer, one forward pass, no gradient steps. | `train.py --prototype-init` |

## Load it

```python
from qwenjev.config import QwenJevConfig
from qwenjev.engine import QwenJevLite

engine = QwenJevLite.from_pretrained(config=QwenJevConfig(
    model_path=r"C:\qwen3.5-4B",
    readout="slot_head",
    readout_checkpoint="models/qwenjev-multitask-v2/readout.pt",
))
print(engine.decide("My payouts have failed three times.", {
    "queue": {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"payments": "Payouts", "account": "Login", "other": "Else"}},
    "escalate": {"type": "bool", "instructions": "Does this require urgent attention?"},
}).to_dict())
```

## Rebuild the shipped head

```bash
python scripts/interpolate.py --checkpoint models/raw-trained-rows/readout.pt \
  --alphas 0.5 --tasks scifact_rel_bool arguana_rel_bool jigsaw goemotions mnli snli \
  clinc150_top15 scifact_rel_score jigsaw_severity \
  --out models/shared-rows/readout.pt

python scripts/compose_wide_rows.py \
  --base models/shared-rows/readout.pt \
  --wide models/prototype-rows/readout.pt \
  --data-dir data/ready \
  --out models/qwenjev-multitask-v2/readout.pt
```

The prototypes themselves are reproduced with
`train.py --prototype-init --items-per-label 12 --min-items-per-task 200` (about 20
minutes on a 3090); the shared rows in the same way with `train.py` and then one
`interpolate.py` run. Why this shape, and what was tried before it, is in
[`artifacts/REPORT_V8_CN.md`](../artifacts/REPORT_V8_CN.md).
