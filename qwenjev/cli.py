"""Command line entry point: ``python -m qwenjev.cli <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import QwenJevConfig


def _engine(args, *, readout: str | None = None, checkpoint: str | None = None):
    from .engine import QwenJevLite

    chosen = readout or getattr(args, "readout", "reserved_label")
    if getattr(args, "fake", False):
        from .testing import build_tiny_engine

        return build_tiny_engine(readout=chosen)
    config = QwenJevConfig(
        model_path=args.model,
        readout=chosen,
        readout_checkpoint=checkpoint,
        device=getattr(args, "device", "cuda:0"),
    )
    return QwenJevLite.from_pretrained(config=config)


EXAMPLE_STATE = (
    "My payouts have failed three times. The bank says everything is fine. "
    "The customer has been waiting nine days and is threatening a chargeback."
)
EXAMPLE_QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "payments": "Payout failures and payment processing",
            "account": "Login and account access",
            "other": "Something else",
        },
    },
    "escalate": {
        "type": "bool",
        "instructions": "Does this message require urgent human attention?",
    },
}


def cmd_demo(args) -> int:
    engine = _engine(args)
    response = engine.decide(args.state or EXAMPLE_STATE, json.loads(args.questions) if args.questions else EXAMPLE_QUESTIONS)
    print(json.dumps(response.to_dict(), indent=2))
    return 0


def cmd_serve(args) -> int:
    from .api import create_app

    import uvicorn

    config = QwenJevConfig(model_path=args.model, device=args.device, readout=args.readout)
    kwargs = {"config": config, "fake": args.fake}
    app = create_app(**kwargs)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_probe(args) -> int:
    from .probes import run_all

    engine = _engine(args)
    names = None if args.probe in (None, "all") else [args.probe]
    results = run_all(engine, quick=args.quick, names=names)
    payload = [r.to_dict() for r in results]
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def cmd_account(args) -> int:
    engine = _engine(args)
    state = args.state or EXAMPLE_STATE
    questions = json.loads(args.questions) if args.questions else EXAMPLE_QUESTIONS
    accounting = engine.account(state, questions)
    print(json.dumps(accounting, indent=2))
    engine.assert_within_limits(accounting)
    print("within API limits", file=sys.stderr)
    return 0


def cmd_datasets(args) -> int:
    """Show which datasets are present on disk, and how to get the rest."""

    from .datasets import status
    from .normalize import catalog, load_manifest_notes

    report = status(args.root)
    if args.json:
        print(json.dumps({"raw": report, "normalized": catalog(args.normalized)}, indent=2))
        return 0
    normalized = catalog(args.normalized)
    if normalized:
        notes = load_manifest_notes(args.normalized)
        manifest_path = Path(args.normalized) / "manifest.json"
        counts = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for task, entry in (manifest.get("tasks") or {}).items():
                counts[task] = entry.get("splits", {})
        print(f"normalised tasks in {args.normalized}:")
        for task in sorted(normalized):
            splits = counts.get(task, {})
            detail = ", ".join(
                f"{name}:{info['items']}i/{info['decisions']}d" for name, info in splits.items()
            )
            print(f"  {task:<13} {detail or ', '.join(normalized[task])}")
            if notes.get(task):
                print(f"                note: {notes[task]}")
        print()
    print("raw dataset folders:")
    for entry in report:
        mark = "OK     " if entry["available"] else "MISSING"
        print(f"[{mark}] {entry['dataset']:<10} {entry['title']}")
        print(f"          task    : {entry['task']}")
        print(f"          readout : {entry['readout']}")
        print(f"          root    : {entry['root']}")
        if entry["files"]:
            print(f"          files   : {', '.join(entry['files'])}")
        if not entry["available"]:
            for line in entry["hint"].strip().splitlines():
                print(f"          {line}")
        print()
    return 0


def _dataset_kwargs(args) -> dict:
    kwargs: dict = {}
    name = args.dataset
    if name == "clinc150" and getattr(args, "max_intents", None):
        kwargs["max_intents"] = args.max_intents
    if name == "hwu64":
        kwargs["include_domain"] = bool(getattr(args, "domains", False))
        if getattr(args, "max_intents", None):
            kwargs["max_intents"] = args.max_intents
    if name == "mmlu" and getattr(args, "state_mode", None):
        kwargs["state_mode"] = args.state_mode
    if name == "fever" and getattr(args, "evidence", None):
        from pathlib import Path as _Path

        kwargs["evidence"] = _Path(args.evidence)
    if name == "synthetic":
        kwargs["seed"] = getattr(args, "seed", 0)
        kwargs["question"] = getattr(args, "question", None)
    return kwargs


def cmd_eval(args) -> int:
    """Zero-shot (or checkpoint) evaluation of a decision model on a dataset."""

    from .datasets import DatasetUnavailable, load_items
    from .evaluation import evaluate_items

    backend = _backend(args)
    try:
        items = load_items(
            args.dataset,
            root=args.root,
            split=args.split,
            limit=args.limit,
            normalized=args.data_root,
            **_dataset_kwargs(args),
        )
    except (DatasetUnavailable, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    result = evaluate_items(
        backend,
        items,
        dataset=args.dataset,
        split=args.split or "default",
        share_state=None if args.share_state is None else args.share_state,
        example_count=args.examples,
        progress=args.progress,
    )
    payload = result.summary()
    payload["backend"] = backend.name
    payload["readout"] = getattr(backend, "readout", None) and backend.readout.name or backend.name
    payload["temperature"] = getattr(backend, "readout_temperature", None)
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _backend(args):
    """Pick the decision model: our Qwen engine, or a Laya checkpoint."""

    kind = getattr(args, "backend", None) or "qwenjev"
    if kind == "laya":
        from .backends import LayaBackend

        return LayaBackend(
            model_path=getattr(args, "laya_path", None) or r"C:\laya",
            device=getattr(args, "device", None),
        )
    from .backends import QwenBackend

    return QwenBackend(_engine(args, readout=args.readout, checkpoint=args.checkpoint))


def cmd_normalize(args) -> int:
    """Turn the raw datasets into the one canonical decision-record format."""

    from .normalize import normalize_all, refresh_manifest

    if getattr(args, "manifest_only", False):
        manifest = refresh_manifest(args.out, args.src, seed=args.seed)
    else:
        manifest = normalize_all(
            args.src,
            args.out,
            tasks=args.tasks,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            seed=args.seed,
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    tasks = manifest["tasks"]
    ok = [k for k, v in tasks.items() if v.get("status") == "ok"]
    skipped = [k for k, v in tasks.items() if v.get("status") != "ok"]
    print(
        f"\n{len(ok)} tasks normalised into {args.out}: {', '.join(sorted(ok))}",
        file=sys.stderr,
    )
    for key in skipped:
        print(f"  skipped {key}: {tasks[key].get('warning') or tasks[key].get('status')}", file=sys.stderr)
    return 0


def cmd_relevance(args) -> int:
    """Turn query/document collections into relevance-judgement tasks."""

    from .relevance import normalize_relevance

    manifest = normalize_relevance(
        args.src,
        args.out,
        collections=args.collections,
        modes=args.modes,
        queries=args.queries,
        train_queries=args.train_queries,
        positives=args.positives,
        negatives=args.negatives,
        doc_chars=args.doc_chars,
        option_chars=args.option_chars,
        seed=args.seed,
        progress=not args.quiet,
    )
    if not args.quiet:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    done = [key for key, entry in manifest["tasks"].items() if entry.get("status") == "ok"]
    print(
        f"\n{len(done)} collections written to {args.out}: {', '.join(sorted(done))}",
        file=sys.stderr,
    )
    return 0


def _variant_from_spec(spec: str, args):
    """Variant grammar::

        name:laya
        name:qwenjev                          # the --readout default
        name:qwenjev@reserved_label           # pick the readout explicitly
        name:qwenjev@slot_head:checkpoint.pt  # readout + RLCD checkpoint
        name:qwenjev@checkpoint.pt            # checkpoint (implies slot_head)
    """

    from .backends import LayaBackend, QwenBackend
    from .benchmark import Variant
    from .config import QwenJevConfig
    from .engine import QwenJevLite

    name, _, kind = spec.partition(":")
    if not kind:
        name, kind = kind or "model", name
    checkpoint = None
    readout = args.readout
    if "@" in kind:
        kind, _, checkpoint = kind.partition("@")
        if ":" in checkpoint:
            readout, _, checkpoint = checkpoint.partition(":")
        elif checkpoint.endswith(".pt") or "/" in checkpoint or "\\" in checkpoint:
            readout = "slot_head"
        else:
            readout, checkpoint = checkpoint, None
    if kind == "laya":
        backend = LayaBackend(model_path=getattr(args, "laya_path", None) or r"C:\laya")
        return Variant(name=name, backend=backend, note="Laya checkpoint, zero-shot")
    config = QwenJevConfig(
        model_path=args.model,
        device=args.device,
        readout=readout,
        readout_checkpoint=checkpoint,
        branch_batch_size=args.branch_batch_size,
    )
    engine = QwenJevLite.from_pretrained(config=config)
    note = "Qwen3.5-4B, zero-shot readout" if not checkpoint else f"RLCD readout {checkpoint}"
    return Variant(name=name, backend=QwenBackend(engine), note=note)


def cmd_benchmark(args) -> int:
    """Score one or more backends over the normalised tasks, with speed."""

    from .benchmark import SPLITS, run_benchmark
    from .normalize import catalog

    available = catalog(args.data_root)
    tasks = args.tasks or [t for t in SPLITS if t in available or t == "mnli_ood"]
    variants = [_variant_from_spec(spec, args) for spec in args.variants]
    for variant in variants:
        print(f"variant {variant.name}: {variant.note}", file=sys.stderr)

    report = run_benchmark(
        variants,
        data_root=args.data_root,
        tasks=tasks,
        test_limit=args.test_limit,
        progress=not args.quiet,
        out_path=args.out,
    )
    payload = report.to_dict()
    payload["task_splits"] = {task: SPLITS.get(task, "test") for task in tasks}
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    if args.markdown:
        from .benchmark import render_markdown
        from .normalize import load_manifest_notes

        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(
            render_markdown(report, notes=load_manifest_notes(args.data_root)), encoding="utf-8"
        )
        print(f"wrote {args.markdown}", file=sys.stderr)
    return 0


def cmd_train(args) -> int:
    from .rlcd import RLCDFineTune, fit_temperature

    if getattr(args, "multitask", False):
        return _train_multitask(args)

    dataset = getattr(args, "dataset", None) or "synthetic"
    if dataset == "synthetic":
        from .synth import generate, question_specs

        specs = question_specs()
        train = generate(args.train_size, seed=args.seed, question=args.question)
        held_out = generate(args.eval_size, seed=args.seed + 1000, question=args.question)
        shifted = generate(args.eval_size, seed=args.seed + 2000, shift=True, question=args.question)
        calibration = generate(args.calib_size, seed=args.seed + 3000, question=args.question)
    else:
        specs, train, held_out, calibration, shifted = _dataset_training_data(args, dataset)

    engine = _engine(args, readout="slot_head", checkpoint=None)

    tuner = RLCDFineTune(
        engine,
        question_specs=specs,
        objective=args.objective,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
    )
    before = tuner.evaluate(held_out)
    report = tuner.train(train, epochs=args.epochs)
    after = tuner.evaluate(held_out)
    temperature = fit_temperature(engine, tuner.evaluate(calibration))
    engine.readout_temperature = temperature
    calibrated = tuner.evaluate(held_out)
    engine.readout_temperature = 1.0
    shifted_metrics = tuner.evaluate(shifted).reliability() if shifted else None

    summary = {
        "steps": report.steps,
        "mean_loss": report.loss,
        "seconds": round(report.seconds, 2),
        "objective": report.objective,
        "temperature": temperature,
        "before": {k: v for k, v in before.reliability().items() if k != "bins"},
        "after": {k: v for k, v in after.reliability().items() if k != "bins"},
        "after_temperature": {k: v for k, v in calibrated.reliability().items() if k != "bins"},
        "shifted": (
            {k: v for k, v in shifted_metrics.items() if k != "bins"} if shifted_metrics else None
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tuner.save(args.out)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.report}", file=sys.stderr)
    return 0


def _dataset_training_data(args, dataset: str):
    """Fit the training loop to a real dataset: train / calibration / held-out."""

    from .datasets import DatasetUnavailable, items_to_samples, load_items

    kwargs = _dataset_kwargs(args)
    train_split = getattr(args, "train_split", None) or "train"
    test_split = args.split or None
    try:
        train_items = load_items(
            dataset, root=args.root, split=train_split, limit=args.train_size, **kwargs
        )
        test_items = load_items(
            dataset, root=args.root, split=test_split, limit=args.eval_size * 2, **kwargs
        )
    except DatasetUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    half = len(test_items) // 2
    calibration_items, held_out_items = test_items[:half], test_items[half:]
    print(
        f"{dataset}: {len(train_items)} train items, {len(held_out_items)} held out, "
        f"{len(calibration_items)} for the temperature fit",
        file=sys.stderr,
    )
    # Samples carry their own question spec, so no shared mapping is needed.
    return (
        {},
        items_to_samples(train_items),
        items_to_samples(held_out_items),
        items_to_samples(calibration_items),
        [],
    )


def _train_multitask(args) -> int:
    """One readout trained on every task's train split at once, class-balanced."""

    import random
    from pathlib import Path as _Path

    from .datasets import items_to_samples
    from .normalize import load_normalized
    from .rlcd import RLCDFineTune

    root = Path(args.train_data_root)
    budget = args.decisions_per_task
    rng = random.Random(args.seed)
    tasks = list(args.tasks) if args.tasks else sorted(
        path.stem[: -len("_train")] for path in root.glob("*_train.jsonl")
    )
    excluded = set(getattr(args, "exclude_tasks", None) or [])
    if excluded:
        skipped = [task for task in tasks if task in excluded]
        tasks = [task for task in tasks if task not in excluded]
        print(f"excluding {', '.join(skipped)}", file=sys.stderr)
    samples = []
    per_task: dict[str, dict] = {}
    for task in tasks:
        path = root / f"{task}_train.jsonl"
        if not path.is_file():
            print(f"skipping {task}: no {path}", file=sys.stderr)
            continue
        items = load_normalized(path)
        rng.shuffle(items)
        taken: list = []
        decisions = 0
        for entry in items:
            taken.append(entry)
            decisions += len(entry.questions)
            if budget and decisions >= budget:
                break
        samples.extend(items_to_samples(taken))
        per_task[task] = {"items": len(taken), "decisions": decisions}
        print(f"{task:<14} {len(taken):>5} items  {decisions:>6} decisions", file=sys.stderr)
    rng.shuffle(samples)
    print(f"total {len(samples)} training decisions before balancing", file=sys.stderr)

    engine = _engine(args, readout="slot_head", checkpoint=None)
    from .rlcd import _balance_samples

    samples = _balance_samples(samples, seed=args.seed)
    rng.shuffle(samples)
    if args.max_train_samples and len(samples) > args.max_train_samples:
        samples = samples[: args.max_train_samples]
    print(f"{len(samples)} training decisions after class balancing", file=sys.stderr)
    tuner = RLCDFineTune(
        engine,
        objective=args.objective,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        balance="none",
    )
    report = tuner.train(samples, epochs=args.epochs)

    model_dir = _Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "readout.pt"
    tuner.save(str(checkpoint_path))
    summary = {
        "model_dir": str(model_dir),
        "tasks": per_task,
        "samples": len(samples),
        "steps": report.steps,
        "epochs": report.epochs,
        "mean_loss": report.loss,
        "seconds": round(report.seconds, 2),
        "objective": report.objective,
        "readout": engine.readout.name,
        "balance": "class",
        "probes_loss": report.history[:5],
        "probes_loss_last": report.history[-5:],
    }
    (model_dir / "card.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.report}", file=sys.stderr)
    print(f"model saved to {checkpoint_path}", file=sys.stderr)
    return 0


def cmd_reproduce(args) -> int:
    """Run the whole reproduction: demo, probes, RLCD, report."""

    from .probes import run_all
    from .report import render_report
    from .rlcd import RLCDFineTune, fit_temperature
    from .synth import generate, question_specs

    import torch

    engine = _engine(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo = engine.decide(EXAMPLE_STATE, EXAMPLE_QUESTIONS).to_dict()
    probes = [r.to_dict() for r in run_all(engine, quick=args.quick)]
    torch.cuda.empty_cache()

    training = None
    if not args.no_train:
        specs = question_specs()
        train = generate(args.train_size, seed=1, question=args.question)
        held_out = generate(args.eval_size, seed=2, question=args.question)
        shifted = generate(args.eval_size, seed=3, shift=True, question=args.question)
        calibration = generate(args.calib_size, seed=4, question=args.question)
        engine.config.readout = "slot_head"
        from .readout import build_readout

        engine.readout = build_readout(
            "slot_head",
            engine.tokenizer,
            engine.model.get_output_embeddings(),
            int(engine.model.config.text_config.hidden_size),
            max_slots=engine.limits.max_slots,
        ).to(engine.device)
        engine.readout_temperature = 1.0
        tuner = RLCDFineTune(
            engine,
            question_specs=specs,
            objective=args.objective,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        before = tuner.evaluate(held_out)
        report = tuner.train(train, epochs=args.epochs)
        after = tuner.evaluate(held_out)
        temperature = fit_temperature(engine, tuner.evaluate(calibration))
        engine.readout_temperature = temperature
        calibrated = tuner.evaluate(held_out)
        engine.readout_temperature = 1.0
        tuner.save(str(out_dir / "readout.pt"))
        training = {
            "steps": report.steps,
            "mean_loss": report.loss,
            "seconds": round(report.seconds, 2),
            "objective": report.objective,
            "temperature": temperature,
            "before": {k: v for k, v in before.reliability().items() if k != "bins"},
            "after": {k: v for k, v in after.reliability().items() if k != "bins"},
            "after_temperature": {k: v for k, v in calibrated.reliability().items() if k != "bins"},
            "shifted": {
                k: v for k, v in tuner.evaluate(shifted).reliability().items() if k != "bins"
            },
        }

    info = {
        "backbone": engine.config.model_path,
        "readout (probes)": "reserved_label",
        "readout (training)": "slot_head" if training else "n/a",
        "device": str(engine.device),
        "limits": f"<= {engine.limits.max_options} options, "
        f"{engine.limits.max_branch_tokens} per branch, {engine.limits.max_request_tokens} per request",
    }
    markdown = render_report(demo=demo, probes=probes, training=training, engine_info=info)
    (out_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps({"demo": demo, "probes": probes, "training": training, "engine": info}, indent=2),
        encoding="utf-8",
    )
    print(markdown)
    print(f"\nwrote {out_dir / 'REPORT.md'} and {out_dir / 'report.json'}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .config import DEFAULT_MODEL_PATH

    parser = argparse.ArgumentParser(prog="qwenjev", description="QwenJev-lite")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--readout", default="reserved_label", choices=["reserved_label", "slot_head", "pointer"])
    parser.add_argument("--fake", action="store_true", help="use the tiny random backbone")
    parser.add_argument("--branch-batch-size", type=int, default=64)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the essay's opening example")
    demo.add_argument("--state", default=None)
    demo.add_argument("--questions", default=None, help="JSON questions object")
    demo.set_defaults(func=cmd_demo)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8300)
    serve.set_defaults(func=cmd_serve)

    probe = sub.add_parser("probe", help="run the essay's experiments")
    probe.add_argument("probe", nargs="?", default="all")
    probe.add_argument("--quick", action="store_true")
    probe.add_argument("--out", default=None)
    probe.set_defaults(func=cmd_probe)

    account = sub.add_parser("account", help="token accounting and limit checks")
    account.add_argument("--state", default=None)
    account.add_argument("--questions", default=None)
    account.set_defaults(func=cmd_account)

    train = sub.add_parser("train", help="RLCD: train the readout on outcomes")
    train.add_argument("--dataset", default="synthetic", help="synthetic, fever, mmlu, clinc150, hwu64, jigsaw")
    train.add_argument("--root", default=None, help="dataset directory (default: data/<dataset>)")
    train.add_argument("--split", default=None, help="evaluation split (default: the dataset's test split)")
    train.add_argument("--train-split", default=None, help="training split (default: train)")
    train.add_argument("--limit", type=int, default=None, help="cap the evaluation items")
    train.add_argument("--max-intents", type=int, default=None, help="keep only the N most frequent intents")
    train.add_argument("--domains", action="store_true", help="HWU64: also ask the domain question")
    train.add_argument("--state-mode", default=None, choices=[None, "question", "context"])
    train.add_argument("--evidence", default=None, help="FEVER: evidence.jsonl path")
    train.add_argument("--tasks", nargs="*", default=None,
                       help="multi-task RLCD over normalised train files")
    train.add_argument("--multitask", action="store_true",
                       help="train one readout on every normalised train split")
    train.add_argument("--exclude-tasks", nargs="*", default=None,
                       help="tasks to leave out of the combined training set")
    train.add_argument("--train-data-root", default="data/ready")
    train.add_argument("--decisions-per-task", type=int, default=600, help="0 = no cap")
    train.add_argument("--max-train-samples", type=int, default=10000, help="0 = no cap after balancing")
    train.add_argument("--model-dir", default="models/qwenjev-multitask-v2",
                       help="folder for readout.pt + card.json")
    train.add_argument("--question", default="queue", choices=["queue", "escalate"])
    train.add_argument("--train-size", type=int, default=1024)
    train.add_argument("--eval-size", type=int, default=192)
    train.add_argument("--calib-size", type=int, default=192)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--objective", default="log_loss", choices=["log_loss", "brier"])
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--out", default=None, help="write the readout checkpoint here")
    train.add_argument("--report", default=None, help="write the metrics JSON here")
    train.set_defaults(func=cmd_train)

    datasets_cmd = sub.add_parser("datasets", help="show which datasets are present, and how to get the rest")
    datasets_cmd.add_argument("--root", default=None, help="override the data root")
    datasets_cmd.add_argument("--normalized", default="data/ready")
    datasets_cmd.add_argument("--json", action="store_true")
    datasets_cmd.set_defaults(func=cmd_datasets)

    evaluate = sub.add_parser("eval", help="evaluate a decision model on a dataset")
    evaluate.add_argument("--dataset", required=True, help="fever, mmlu, clinc150, hwu64, jigsaw, synthetic")
    evaluate.add_argument("--root", default=None, help="raw dataset directory (default data/<dataset>)")
    evaluate.add_argument("--data-root", default=None, help="formatted dataset directory (default data/ready)")
    evaluate.add_argument("--backend", default="qwenjev", choices=["qwenjev", "laya"])
    evaluate.add_argument("--laya-path", default=None, help="Laya checkpoint directory")
    evaluate.add_argument("--split", default=None)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--checkpoint", default=None, help="readout checkpoint from `qwenjev train`")
    evaluate.add_argument("--max-intents", type=int, default=None)
    evaluate.add_argument("--domains", action="store_true")
    evaluate.add_argument("--state-mode", default=None, choices=[None, "question", "context"])
    evaluate.add_argument("--evidence", default=None)
    evaluate.add_argument("--examples", type=int, default=3)
    evaluate.add_argument("--share-state", dest="share_state", action="store_true", default=None)
    evaluate.add_argument("--no-share-state", dest="share_state", action="store_false")
    evaluate.add_argument("--progress", action="store_true")
    evaluate.add_argument("--out", default=None)
    evaluate.set_defaults(func=cmd_eval)

    normalize = sub.add_parser("normalize", help="convert the raw data into data/ready/*.jsonl")
    normalize.add_argument("--src", default="datasets", help="root holding the raw dataset folders")
    normalize.add_argument("--out", default="data/ready")
    normalize.add_argument("--tasks", nargs="*", default=None)
    normalize.add_argument("--train-limit", type=int, default=2000, help="0 = no cap")
    normalize.add_argument("--test-limit", type=int, default=400, help="0 = no cap")
    normalize.add_argument("--seed", type=int, default=0)
    normalize.add_argument("--manifest-only", action="store_true",
                           help="rebuild manifest.json from the JSONL files already on disk")
    normalize.set_defaults(func=cmd_normalize)

    benchmark = sub.add_parser("benchmark", help="score backends over the normalised tasks")
    benchmark.add_argument("--variants", nargs="+", required=True,
                           help="name:qwenjev | name:qwenjev@checkpoint.pt | name:laya")
    benchmark.add_argument("--tasks", nargs="*", default=None)
    benchmark.add_argument("--data-root", default="data/ready")
    benchmark.add_argument("--test-limit", type=int, default=150, help="0 = the whole split")
    benchmark.add_argument("--laya-path", default=None)
    benchmark.add_argument("--out", default=None)
    benchmark.add_argument("--markdown", default=None)
    benchmark.add_argument("--quiet", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark)

    relevance = sub.add_parser(
        "relevance",
        help="build bool/choice/score relevance-judgement tasks from query/document collections",
    )
    relevance.add_argument("--src", default=r"D:\AutoBM25\dataset_en")
    relevance.add_argument("--out", default="data/ready")
    relevance.add_argument("--collections", nargs="*", default=None)
    relevance.add_argument("--modes", nargs="+", default=["bool", "choice", "score"], choices=["bool", "choice", "score"])
    relevance.add_argument("--queries", type=int, default=60, help="queries judged per task")
    relevance.add_argument("--train-queries", type=int, default=150)
    relevance.add_argument("--positives", type=int, default=2, help="judged passages per query (bool/score)")
    relevance.add_argument("--negatives", type=int, default=4, help="unjudged passages per query (bool/score)")
    relevance.add_argument("--doc-chars", type=int, default=500)
    relevance.add_argument("--option-chars", type=int, default=160)
    relevance.add_argument("--seed", type=int, default=0)
    relevance.add_argument("--quiet", action="store_true")
    relevance.set_defaults(func=cmd_relevance)


    reproduce = sub.add_parser("reproduce", help="run the whole reproduction and write a report")
    reproduce.add_argument("--out-dir", default="artifacts")
    reproduce.add_argument("--quick", action="store_true")
    reproduce.add_argument("--no-train", action="store_true")
    reproduce.add_argument("--question", default="queue", choices=["queue", "escalate"])
    reproduce.add_argument("--train-size", type=int, default=1024)
    reproduce.add_argument("--eval-size", type=int, default=192)
    reproduce.add_argument("--calib-size", type=int, default=192)
    reproduce.add_argument("--epochs", type=int, default=2)
    reproduce.add_argument("--objective", default="log_loss", choices=["log_loss", "brier"])
    reproduce.add_argument("--lr", type=float, default=5e-4)
    reproduce.add_argument("--batch-size", type=int, default=8)
    reproduce.add_argument("--weight-decay", type=float, default=1e-2)
    reproduce.set_defaults(func=cmd_reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
