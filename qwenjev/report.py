"""Render probe and training results as a markdown report."""

from __future__ import annotations

from typing import Any


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _probe_section(result: dict) -> str:
    name = result["name"]
    summary = result["summary"]
    out = [f"### {name}", ""]

    if name == "visibility":
        out.append(
            _table(
                ["condition", "mean p(secret)", "max p(secret)", "n"],
                [
                    [k, _fmt(v["mean_p_secret"]), _fmt(v["max_p_secret"]), v["n"]]
                    for k, v in summary.items()
                ],
            )
        )
    elif name == "reference_card":
        rows = [
            [f"card in options: {pos}", v["correct"], v["n"], _fmt(v["mean_p_correct"])]
            for pos, v in summary["by_position"].items()
        ]
        state = summary["state_reference"]
        rows.append(["card in shared state", state["correct"], state["n"], _fmt(state["mean_p_correct"])])
        out.append(_table(["condition", "correct", "n", "mean p(correct)"], rows))
    elif name == "option_interaction":
        out.append(
            _table(
                ["quantity", "value"],
                [
                    ["mean log-odds, 4 options", _fmt(summary["base4_log_odds"])],
                    ["mean log-odds, 5 options", _fmt(summary["append5_log_odds"])],
                    ["mean change", _fmt(summary["mean_change_append"])],
                    [
                        "95% paired interval",
                        f"[{_fmt(summary['interval95_append'][0])}, {_fmt(summary['interval95_append'][1])}]",
                    ],
                    ["decreased in all blocks", summary["decreased_in_all_blocks"]],
                    ["duplicate-request control (noise)", _fmt(summary["null_change"])],
                    ["mean change with a different extra option", _fmt(summary["mean_change_replace"])],
                    ["blocks", summary["blocks"]],
                ],
            )
        )
    elif name == "option_order":
        out.append(
            _table(
                ["condition", "mean p(payments)", "mean p(top)"],
                [
                    [k, _fmt(v["mean_p_payments"]), _fmt(v["mean_p_top"])]
                    for k, v in summary.items()
                    if isinstance(v, dict)
                ],
            )
        )
        out.append("")
        out.append(f"Mean shift in p(payments): **{_fmt(summary['mean_shift_p_payments'])}**")
    elif name == "fake_option":
        out.append(
            _table(
                ["condition", "slots", "answer", "p(account)"],
                [
                    [k, summary["slots"][k], summary["answer"][k], _fmt(summary["p_account"][k])]
                    for k in summary["slots"]
                ],
            )
        )
    elif name == "accounting":
        out.append(
            _table(
                ["quantity", "value"],
                [
                    ["shared state tokens", summary["state_tokens"]],
                    ["suffix tokens, one bool", summary["suffix_tokens"]["one_bool"]],
                    ["suffix tokens, two identical bools", summary["suffix_tokens"]["two_bool"]],
                    ["suffix tokens, three-option choice", summary["suffix_tokens"]["choice3"]],
                    ["additive", summary["additive"]],
                    ["mixed request equals the sum", summary["mixed_matches_sum"]],
                    ["billing figure, short identifier", summary["billing_short_id"]],
                    ["billing figure, long identifier", summary["billing_long_id"]],
                ],
            )
        )
    elif name == "latency":
        out.append("State length sweep (one question, state cache disabled):")
        out.append("")
        out.append(
            _table(
                ["state tokens", "median ms"],
                [[k, round(v, 1)] for k, v in summary["state_length_ms"].items()],
            )
        )
        out.append("")
        out.append("Question count sweep (short state):")
        out.append("")
        out.append(
            _table(
                ["questions", "median ms"],
                [[k, round(v, 1)] for k, v in summary["question_count_ms"].items()],
            )
        )
        out.append("")
        out.append("One shared-state request vs Q separate requests:")
        out.append("")
        out.append(
            _table(
                ["questions", "shared ms", "separate ms", "speedup"],
                [
                    [k, v["shared_ms"], v["separate_ms"], v["speedup"]]
                    for k, v in summary["shared_vs_separate"].items()
                ],
            )
        )
        if "note" in summary:
            out += ["", summary["note"]]
    else:  # pragma: no cover - future probes
        out.append("```json")
        out.append(str(summary))
        out.append("```")
    return "\n".join(out)


def render_report(
    *,
    demo: dict | None,
    probes: list[dict],
    training: dict | None = None,
    engine_info: dict | None = None,
    title: str = "QwenJev-lite reproduction report",
) -> str:
    parts = [f"# {title}", ""]
    if engine_info:
        parts.append(
            _table(
                ["field", "value"],
                [[k, v] for k, v in engine_info.items()],
            )
        )
        parts.append("")
    if demo is not None:
        parts += ["## The reference example", "", "```json", _pretty(demo), "```", ""]
    if probes:
        parts += ["## Probes", ""]
        for result in probes:
            parts.append(_probe_section(result))
            parts.append("")
    if training:
        parts += ["## RLCD: training the readout against outcomes", ""]
        parts.append(
            _table(
                ["metric", "before", "after RLCD", "after temperature", "shifted data"],
                [
                    [
                        metric,
                        _fmt(training["before"][metric]),
                        _fmt(training["after"][metric]),
                        _fmt(training["after_temperature"][metric]),
                        _fmt(training["shifted"][metric]),
                    ]
                    for metric in ("accuracy", "mean_top_probability", "ece", "brier", "nll")
                ],
            )
        )
        parts += [
            "",
            f"- steps: {training['steps']}, objective: `{training['objective']}`, "
            f"mean loss: {_fmt(training['mean_loss'])}, wall clock: {training['seconds']}s",
            f"- fitted temperature: {_fmt(training['temperature'])}",
            "",
        ]
    return "\n".join(parts)


def _pretty(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)
