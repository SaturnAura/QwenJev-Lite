"""Reinforcement Learning for Calibrated Decisions, in its supervised form.

"TypeSafe calls its training method Reinforcement Learning for Calibrated Decisions,
or RLCD. ... My proposed training recipe adapts the transformer and readout to typed
decision tasks using an outcome-based objective." Log loss and the Brier score are
both proper scoring rules, so minimising either in expectation recovers the true
conditional distribution (essay section 5).

This module implements that objective on typed decisions. The backbone is frozen and
the readout is trained; the same loss works with backbone gradients if the caller has
the memory for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import os

import numpy as np
import torch
import torch.nn.functional as F

from .calibration import (
    TemperatureScaler,
    multiclass_brier,
    negative_log_likelihood,
    summarise_reliability,
)
from .prompt import render_branch, render_state
from .schema import FAMILY_SEP, SLOT_RESERVE, build_question, question_option_ids
from .tokenize import tokenize_branch


def log_loss(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``-log p(y)`` for the observed outcome (proper scoring rule)."""

    return F.nll_loss(log_probs, targets)


def brier_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Squared distance between the predicted distribution and a one-hot outcome."""

    one_hot = F.one_hot(targets, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    return ((probabilities - one_hot) ** 2).sum(dim=-1).mean()


def _balance_samples(samples, *, seed: int = 0, cap: int = 1200) -> list:
    """Oversample so every class within a label space contributes equally.

    Log loss on imbalanced binary/ordinal tasks learns the class prior ("answer no"
    everywhere). Grouping by (option keys, target) and equalising group sizes fixes that
    without touching the loss function. ``cap`` bounds a single group, so a many-label
    task such as CLINC150 does not blow up the batch count.
    """

    from collections import defaultdict

    groups: dict[tuple, list] = defaultdict(list)
    for sample in samples:
        spec = getattr(sample, "spec", None)
        criteria = (spec or {}).get("criteria") if isinstance(spec, dict) else None
        keys = tuple(sorted(criteria)) if isinstance(criteria, dict) else ("-",)
        groups[(keys, sample.target)].append(sample)
    if not groups:
        return samples
    target_size = min(max(len(group) for group in groups.values()), cap)
    balanced: list = []
    for group in groups.values():
        if not group:
            continue
        if len(group) >= target_size:
            balanced.extend(group[:target_size])
        else:
            repeats = (target_size + len(group) - 1) // len(group)
            balanced.extend((group * repeats)[:target_size])
    import random

    random.Random(seed).shuffle(balanced)
    return balanced


@dataclass
class TrainingReport:
    steps: int = 0
    epochs: int = 0
    loss: float = float("nan")
    objective: str = "log_loss"
    seconds: float = 0.0
    history: list[float] = field(default_factory=list)
    temperature: float | None = None


@dataclass
class Evaluation:
    """Per-item predictions, ready for the reliability helpers."""

    #: probability the model gave the answer it returned (the essay's "probability
    #: it gave its chosen answer")
    top_probabilities: np.ndarray
    #: probability the model gave the outcome that actually happened
    true_probabilities: np.ndarray
    correct: np.ndarray
    targets: list[str]
    answers: list[str]
    logits: list[list[float]]
    probability_rows: list[list[float]]
    target_indices: np.ndarray

    def reliability(self, n_bins: int = 10) -> dict:
        summary = summarise_reliability(self.top_probabilities, self.correct, n_bins=n_bins)
        summary["mean_probability_of_correct_answer"] = float(self.true_probabilities.mean())
        summary["nll"] = negative_log_likelihood(self.true_probabilities)
        summary["brier"] = multiclass_brier(self.probability_rows, self.target_indices)
        summary["balanced_accuracy"] = self.balanced_accuracy()
        return summary

    def balanced_accuracy(self) -> float:
        """Mean per-class recall: the honest metric when one class dominates."""

        from collections import defaultdict

        total: dict[str, int] = defaultdict(int)
        hit: dict[str, int] = defaultdict(int)
        for target, answer in zip(self.targets, self.answers):
            total[target] += 1
            if target == answer:
                hit[target] += 1
        recalls = [hit[key] / total[key] for key in total if total[key]]
        return float(sum(recalls) / len(recalls)) if recalls else float("nan")

    @property
    def probabilities(self) -> np.ndarray:
        """Alias kept for readability: probability of the correct answer."""

        return self.true_probabilities

    def confusion(self) -> dict[str, dict[str, int]]:
        table: dict[str, dict[str, int]] = {}
        for target, answer in zip(self.targets, self.answers):
            table.setdefault(target, {})
            table[target][answer] = table[target].get(answer, 0) + 1
        return table


class RLCDFineTune:
    """Train the readout against outcomes with a proper scoring rule."""

    def __init__(
        self,
        engine,
        *,
        question_specs: dict[str, dict] | None = None,
        objective: str = "log_loss",
        lr: float = 2e-3,
        weight_decay: float = 0.0,
        batch_size: int = 8,
        max_grad_norm: float = 1.0,
        seed: int = 0,
        balance: str = "none",
        anchor: float = 0.0,
        row_norm_cap: float | None = None,
        optimizer: str = "adamw",
        momentum: float = 0.0,
    ):
        if objective not in {"log_loss", "brier"}:
            raise ValueError("objective must be 'log_loss' or 'brier'")
        if balance not in {"none", "class"}:
            raise ValueError("balance must be 'none' or 'class'")
        if anchor < 0:
            raise ValueError("anchor must be >= 0")
        if optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        self.engine = engine
        self.question_specs = question_specs
        self.objective = objective
        self.balance = balance
        # L2 distance to the initial readout. The slot head starts as the reserved
        # vocabulary rows, so anchoring keeps the trained head a *correction* of the
        # pretrained readout instead of a replacement that has to relearn every task.
        self.anchor = anchor
        # The trained rows are projected back onto the logit scale of the reserved
        # label rows after every step. Qwen3.5-4B's decision states have a norm of
        # ~157 while those rows have a norm of ~0.74, so one step at lr 1e-3 moves a
        # fresh row by a fifth of its whole useful length: without this the rows drift
        # to a scale where the logits explode (we measured a mean training loss of 33
        # on the many-option tasks). ``None`` leaves the rows unconstrained.
        self.row_norm_cap = row_norm_cap
        self._capped_rows: list[int] = []
        if row_norm_cap:
            table = getattr(engine.readout, "slot_table", None) or {}
            self._capped_rows = sorted(set(table.values()))
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)
        for param in engine.model.parameters():
            param.requires_grad_(False)
        engine.readout.train()
        trainable = [p for p in engine.readout.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                f"readout {engine.readout.name!r} has no trainable parameters; use the "
                "'slot_head' or 'pointer' readout to train with RLCD"
            )
        self.trainable = trainable
        self._anchor_params = [p.detach().clone() for p in trainable] if anchor else []
        # A row of a linear readout should end up along the mean of the states it has
        # to answer for, which is exactly what SGD builds. Adam rescales every
        # coordinate by its own second moment, so with a handful of samples per row
        # the rows all drift toward ``sign(h)`` and become nearly parallel - measuring
        # that cost us a mean loss of 15 on the many-option tasks against 5.0 for an
        # untouched head. ``sgd`` is therefore the default for a fresh head.
        if optimizer == "sgd":
            # No momentum by default: with a row norm cap the row reaches the cap in a
            # few steps, and momentum then drives it straight past a good direction at
            # ~10x the gradient step (measured: the loss rose above the uniform loss).
            self.optimizer = torch.optim.SGD(
                trainable, lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        else:
            self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        self.optimizer_name = optimizer

    # -- data plumbing ---------------------------------------------------------
    def _specs(self, question_specs: dict[str, dict] | None) -> dict[str, dict]:
        # Samples may carry their own spec (dataset adapters); the mapping is only a
        # fallback for generators that reuse one question definition.
        specs = question_specs or self.question_specs or {}
        self.question_specs = specs
        return specs

    def _prepare(self, samples, question_specs: dict[str, dict] | None = None):
        specs = self._specs(question_specs)
        engine = self.engine
        prepared = []
        for sample in samples:
            spec = sample.spec if getattr(sample, "spec", None) else specs.get(sample.question_id)
            if spec is None:
                raise ValueError(f"no question spec for {sample.question_id!r}")
            question = build_question(sample.question_id, spec)
            branch = tokenize_branch(engine.tokenizer, render_branch(question))
            state_ids = list(
                engine.tokenizer(render_state(sample.state), add_special_tokens=False)["input_ids"]
            )
            keys = [o.key for o in question.options]
            if sample.target not in keys:
                raise ValueError(
                    f"target {sample.target!r} is not a criterion of {sample.question_id!r}"
                )
            prepared.append(
                (
                    state_ids,
                    branch,
                    keys.index(sample.target),
                    keys,
                    question_option_ids(question),
                )
            )
        return prepared

    def _forward(self, rows, *, return_hidden: bool = False):
        """One padded batch of ``[state + branch]`` sequences.

        Returns the slot logits, or ``(logits, decision states)`` when ``hidden`` is set
        (the prototype initialiser needs the representation the readout sees).
        """

        engine = self.engine
        device = engine.device
        pad_id = engine.tokenizer.pad_token_id or 0
        length = max(len(s) + len(b) for s, b, _, _, _ in rows)
        input_ids = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), length), dtype=torch.long, device=device)
        for i, (state_ids, branch, _, _, _) in enumerate(rows):
            s = len(state_ids)
            input_ids[i, :s] = torch.tensor(state_ids, device=device)
            input_ids[i, s : s + len(branch)] = torch.tensor(branch.ids, device=device)
            mask[i, : s + len(branch)] = 1
        out = engine.model.model(
            input_ids=input_ids, attention_mask=mask, return_dict=True, use_cache=False
        )
        hidden = out.last_hidden_state
        decision = torch.stack(
            [
                hidden[i, len(state_ids) + branch.decision_index]
                for i, (state_ids, branch, _, _, _) in enumerate(rows)
            ]
        )
        option_hidden = [
            hidden[i, [len(state_ids) + j for j in branch.option_end_indices], :]
            for i, (state_ids, branch, _, _, _) in enumerate(rows)
        ]
        logits = engine.readout.score(
            decision,
            labels=[list(b.labels) for _, b, _, _, _ in rows],
            option_hidden=option_hidden if engine.readout.uses_option_states else None,
            option_ids=[ids for _, _, _, _, ids in rows],
        )
        return (logits.raw_logits, decision) if return_hidden else logits.raw_logits

    # -- initialisation --------------------------------------------------------
    @torch.no_grad()
    def prototypes(
        self,
        samples,
        *,
        question_specs: dict[str, dict] | None = None,
        batch_size: int | None = None,
        progress: bool = False,
    ) -> dict[str, torch.Tensor]:
        """One row per answer: the mean representation of the states that chose it.

        A linear readout trained by gradient descent needs many passes to find those
        directions: the loss is averaged over the options, so the part of every step
        that actually separates the answers is a small correction to a large shared
        term (measured: a mean training loss of 4.9 after 124 steps on a 150-option
        label space, against 5.0 for an untouched head). The mean representation of an
        answer's outcomes is the closed-form answer to the same question, needs a
        single forward pass, and is exactly the shape the head already has. Each
        row is centred within its own label space - the competition the row appears in
        - and then rescaled, so it keeps the logit scale of the reserved rows.
        """

        from collections import defaultdict

        rows = self._prepare(list(samples), question_specs)
        batch_size = batch_size or self.batch_size
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
        batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
        bar = None
        if progress:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(batches), desc="prototypes", unit="step")
        answer_sum: dict[str, torch.Tensor] = defaultdict(lambda: None)
        answer_count: dict[str, int] = defaultdict(int)
        space_sum: dict[str, torch.Tensor] = defaultdict(lambda: None)
        space_count: dict[str, int] = defaultdict(int)
        for batch_indices in batches:
            batch = [rows[i] for i in batch_indices]
            _logits, decision = self._forward(batch, return_hidden=True)
            for i, (_state, _branch, _target, _keys, ids) in enumerate(batch):
                vector = decision[i].float().cpu()
                # The row of an answer is the mean state of the samples that *chose*
                # it; the space mean is over every sample that offered it.
                chosen = ids[_target]
                space = chosen.rsplit(FAMILY_SEP, 1)[0]
                answer_sum[chosen] = (
                    vector if answer_sum[chosen] is None else answer_sum[chosen] + vector
                )
                answer_count[chosen] += 1
                space_sum[space] = vector if space_sum[space] is None else space_sum[space] + vector
                space_count[space] += 1
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        return {
            answer: answer_sum[answer] / answer_count[answer]
            - space_sum[answer.rsplit(FAMILY_SEP, 1)[0]] / space_count[answer.rsplit(FAMILY_SEP, 1)[0]]
            for answer in answer_sum
        }

    # -- training --------------------------------------------------------------
    def train(
        self,
        samples,
        *,
        epochs: int = 1,
        question_specs: dict[str, dict] | None = None,
        progress: bool = False,
    ) -> TrainingReport:
        samples = list(samples)
        if self.balance == "class":
            samples = _balance_samples(samples, seed=self.seed)
        rows = self._prepare(samples, question_specs)
        t0 = time.perf_counter()
        history: list[float] = []
        steps = 0
        import random as _random

        timeline: list[list[int]] = []
        for epoch in range(epochs):
            order = list(range(len(rows)))
            # Length bucketing: shuffle, then sort by sequence length so a batch pads
            # to a similar size instead of to the longest sample in the batch.
            _random.Random(self.seed + epoch).shuffle(order)
            order.sort(key=lambda i: len(rows[i][0]) + len(rows[i][1]))
            batches = [
                order[start : start + self.batch_size]
                for start in range(0, len(order), self.batch_size)
            ]
            _random.Random(self.seed + epoch).shuffle(batches)
            timeline.extend(batches)

        bar = None
        if progress:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(timeline), desc="RLCD training", unit="step")
        for batch_indices in timeline:
            batch = [rows[i] for i in batch_indices]
            logits = self._forward(batch)
            targets = torch.tensor([r[2] for r in batch], device=logits.device)
            if self.objective == "brier":
                loss = brier_loss(torch.softmax(logits, dim=-1), targets)
            else:
                loss = log_loss(torch.log_softmax(logits, dim=-1), targets)
            if self.anchor and self._anchor_params:
                penalty = sum(
                    ((param - start) ** 2).mean()
                    for param, start in zip(self.trainable, self._anchor_params)
                )
                loss = loss + self.anchor * penalty
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.engine.readout.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
            if os.environ.get("QWENJEV_DEBUG_GRAD") and steps < 4:
                flat = torch.cat(
                    [
                        p.grad.detach().flatten()
                        for p in self.engine.readout.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                )
                options = max(len(r[3]) for r in batch)
                print(
                    f"  step {steps}: options={options:<4} |grad|={float(flat.norm()):12.1f} "
                    f"|penalty|={float(loss.detach()):.3f}"
                )
            self.optimizer.step()
            if self.row_norm_cap and self._capped_rows:
                with torch.no_grad():
                    weight = self.engine.readout.proj.weight
                    index = torch.tensor(self._capped_rows, device=weight.device)
                    capped = weight.index_select(0, index)
                    norms = capped.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    capped.mul_((self.row_norm_cap / norms).clamp(max=1.0))
                    weight.index_copy_(0, index, capped)
            history.append(float(loss.detach()))
            steps += 1
            if bar is not None:
                window = history[-20:]
                bar.set_postfix(loss=f"{sum(window)/len(window):.3f}", refresh=False)
                bar.update(1)
        if bar is not None:
            bar.close()
        self.engine.readout.eval()
        return TrainingReport(
            steps=steps,
            epochs=epochs,
            loss=float(np.mean(history)) if history else float("nan"),
            objective=self.objective,
            seconds=time.perf_counter() - t0,
            history=history,
        )

    # -- evaluation ------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        samples,
        *,
        batch_size: int | None = None,
        question_specs: dict[str, dict] | None = None,
    ) -> Evaluation:
        rows = self._prepare(list(samples), question_specs)
        batch_size = batch_size or self.batch_size
        top_probabilities: list[float] = []
        true_probabilities: list[float] = []
        correct: list[bool] = []
        targets: list[str] = []
        answers: list[str] = []
        logits_out: list[list[float]] = []
        probability_rows: list[list[float]] = []
        target_indices: list[int] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            logits = self._forward(batch)
            probs = torch.softmax(logits / self.engine.readout_temperature, dim=-1)
            for i, (_, _, target_idx, keys, _family) in enumerate(batch):
                row = probs[i, : len(keys)]
                pred = int(row.argmax())
                top_probabilities.append(float(row[pred]))
                true_probabilities.append(float(row[target_idx]))
                correct.append(pred == target_idx)
                targets.append(keys[target_idx])
                answers.append(keys[pred])
                logits_out.append([float(x) for x in logits[i, : len(keys)].tolist()])
                probability_rows.append([float(x) for x in row.tolist()])
                target_indices.append(target_idx)
        return Evaluation(
            top_probabilities=np.asarray(top_probabilities, dtype=float),
            true_probabilities=np.asarray(true_probabilities, dtype=float),
            correct=np.asarray(correct, dtype=bool),
            targets=targets,
            answers=answers,
            logits=logits_out,
            probability_rows=probability_rows,
            target_indices=np.asarray(target_indices, dtype=int),
        )

    # -- persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        table = getattr(self.engine.readout, "slot_table", None)
        torch.save(
            {
                "readout": self.engine.readout.state_dict(),
                "readout_name": self.engine.readout.name,
                "temperature": float(self.engine.readout_temperature),
                "objective": self.objective,
                "slot_table": dict(table) if table else None,
            },
            path,
        )


def build_slot_table(
    samples,
    *,
    question_specs: dict[str, dict] | None = None,
    max_slots: int = 1024,
    reserve: int = SLOT_RESERVE,
) -> tuple[dict[str, int], dict[str, int]]:
    """Give every answer in ``samples`` its own row.

    Returns ``(table, decisions_per_answer)``. Rows ``0 .. reserve-1`` are left alone:
    they stay the pretrained label rows, which is where an answer that is *not* in the
    table is read from. Without this, every task shares the rows at the front of the
    matrix and the last task trained wins - which is exactly why a trained head used
    to be worse than the pretrained one on the label spaces it never saw.
    """

    from collections import Counter

    counts: "Counter[str]" = Counter()
    for sample in samples:
        spec = sample.spec if getattr(sample, "spec", None) else (question_specs or {}).get(
            sample.question_id
        )
        if spec is None:
            raise ValueError(f"no question spec for {sample.question_id!r}")
        counts.update(question_option_ids(build_question(sample.question_id, spec)))

    needed = reserve + len(counts)
    if needed > max_slots:
        raise ValueError(
            f"this mix needs {needed} rows but the head has {max_slots}; drop a task or "
            "raise JevLimits.max_slots"
        )
    table = {answer: reserve + index for index, answer in enumerate(sorted(counts))}
    return table, dict(counts)


def fit_temperature(engine, evaluation: Evaluation) -> float:
    """Fit the readout temperature by minimising NLL on the given outcomes."""

    scaler = TemperatureScaler().fit(list(evaluation.logits), evaluation.target_indices)
    return float(scaler.temperature)
