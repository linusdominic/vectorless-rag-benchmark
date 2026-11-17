"""Turns evaluation records into the tables in the README."""

import json
import math
import statistics
from pathlib import Path

CUTOFFS = (1, 3, 5)


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Honest about small samples."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denominator = 1 + z * z / n
    return (centre - spread) / denominator, (centre + spread) / denominator


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for paired outcomes on the same questions."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(records: list[dict]) -> dict:
    """Metrics for one method over a set of records."""
    n = len(records)
    ranks = [r.get("rank") for r in records]
    section_ranks = [r.get("section_rank") for r in records]
    correct = sum(r["verdict"] == "correct" for r in records)
    low, high = wilson(correct, n)
    scored = [r for r in records if "retrieval_latency_s" in r]
    return {
        "n": n,
        **{f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in CUTOFFS},
        "mrr@5": sum(1 / r for r in ranks if r is not None and r <= 5) / n,
        "section_hit@3": sum(r is not None and r <= 3 for r in section_ranks) / n,
        "answer_accuracy": correct / n,
        "answer_ci": (low, high),
        "not_found": sum(r["verdict"] == "not_found" for r in records) / n,
        "errors": sum(r["verdict"] == "error" for r in records),
        "latency_p50_s": statistics.median(r["retrieval_latency_s"] for r in scored),
        "latency_p90_s": sorted(r["retrieval_latency_s"] for r in scored)[int(len(scored) * 0.9)],
        "retrieval_cost_usd": statistics.mean(r["retrieval_cost_usd"] for r in scored),
        "retrieval_tokens": statistics.mean(
            r["retrieval_input_tokens"] + r["retrieval_output_tokens"] for r in scored
        ),
        "context_chars": statistics.mean(r["context_chars"] for r in scored),
    }


def by_method(records: list[dict]) -> dict[str, dict]:
    methods = list(dict.fromkeys(r["method"] for r in records))
    return {m: summarize([r for r in records if r["method"] == m]) for m in methods}


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _latency(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def _cost(usd: float) -> str:
    return "$0" if usd == 0 else f"${usd:.5f}"


def main_table(summary: dict[str, dict]) -> str:
    lines = [
        "| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Answer accuracy (95% CI) | "
        "Retrieval latency p50 | Retrieval cost per query |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method, s in summary.items():
        low, high = s["answer_ci"]
        lines.append(
            f"| {method} | {_pct(s['hit@1'])} | {_pct(s['hit@3'])} | {_pct(s['hit@5'])} | "
            f"{s['mrr@5']:.2f} | {_pct(s['answer_accuracy'])} ({_pct(low)} to {_pct(high)}) | "
            f"{_latency(s['latency_p50_s'])} | {_cost(s['retrieval_cost_usd'])} |"
        )
    return "\n".join(lines)


def detail_table(summary: dict[str, dict]) -> str:
    lines = [
        '| Method | Right section in top 3 | Said "not found" | Latency p90 | '
        "Retrieval tokens per query | Context sent to the answer model |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method, s in summary.items():
        lines.append(
            f"| {method} | {_pct(s['section_hit@3'])} | {_pct(s['not_found'])} | "
            f"{_latency(s['latency_p90_s'])} | {s['retrieval_tokens']:,.0f} | "
            f"{s['context_chars']:,.0f} chars |"
        )
    return "\n".join(lines)


def style_table(records: list[dict]) -> str:
    styles = list(dict.fromkeys(r["style"] for r in records))
    methods = list(dict.fromkeys(r["method"] for r in records))
    header = "| Method | " + " | ".join(
        f"{style}: Hit@3 | {style}: answer accuracy" for style in styles
    )
    lines = [header + " |", "| --- |" + " --- | --- |" * len(styles)]
    for method in methods:
        cells = []
        for style in styles:
            s = summarize([r for r in records if r["method"] == method and r["style"] == style])
            cells.append(f"{_pct(s['hit@3'])} | {_pct(s['answer_accuracy'])}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def navigation_table(records: list[dict]) -> str:
    """For the vectorless methods: at which of the two steps a miss happened."""
    lines = [
        "| Method | Found the gold paragraph | Missed at step 1 (wrong sections) | "
        "Missed at step 2 (right section, wrong paragraphs) |",
        "| --- | --- | --- | --- |",
    ]
    for method in dict.fromkeys(r["method"] for r in records):
        rows = [r for r in records if r["method"] == method and "sections" in r.get("trace", {})]
        if not rows:
            continue
        found = sum(r["rank"] is not None for r in rows)
        step_one = sum(
            r["rank"] is None and r["gold_paragraph"].split("(")[0] not in r["trace"]["sections"]
            for r in rows
        )
        lines.append(f"| {method} | {found} | {step_one} | {len(rows) - found - step_one} |")
    return "\n".join(lines)


def paired_table(records: list[dict], baseline: str = "hybrid") -> str:
    """Each method against the baseline, question by question."""
    by_method: dict[str, dict[str, dict]] = {}
    for r in records:
        by_method.setdefault(r["method"], {})[r["question_id"]] = r
    if baseline not in by_method:
        return ""
    outcomes = {
        "Hit@1": lambda r: r.get("rank") == 1,
        "Answer": lambda r: r["verdict"] == "correct",
    }
    header = " | ".join(f"{name}: better / worse | {name}: p" for name in outcomes)
    lines = [f"| Method vs {baseline} | {header} |", "| --- |" + " --- | --- |" * len(outcomes)]
    base = by_method[baseline]
    for method, rows in by_method.items():
        if method == baseline:
            continue
        cells = []
        for passed in outcomes.values():
            shared = [q for q in rows if q in base]
            wins = sum(passed(rows[q]) and not passed(base[q]) for q in shared)
            losses = sum(passed(base[q]) and not passed(rows[q]) for q in shared)
            cells.append(f"{wins} / {losses} | {mcnemar_exact(wins, losses):.3f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(records: list[dict]) -> str:
    summary = by_method(records)
    questions = len({r["question_id"] for r in records})
    return "\n\n".join(
        [
            f"Results over {questions} questions.",
            main_table(summary),
            detail_table(summary),
            "By question style:",
            style_table(records),
            "Where the vectorless methods missed:",
            navigation_table(records),
            "Paired comparison with hybrid search (exact McNemar test on the same questions):",
            paired_table(records),
        ]
    )
