"""Question generation, the evaluation loop and the report."""

import json

import pytest

from tests.conftest import FakeLLM
from vrbench import report
from vrbench.evaluate import NOT_FOUND, Models, evaluate_one, rank_of, run
from vrbench.llm import ModelRefused
from vrbench.questions import Question, generate, is_grounded, load, sample_paragraphs, save
from vrbench.retrievers.base import Passage, Retrieval

QUESTION = Question(
    id="q001",
    style="direct",
    question="How long must an escrow account be kept?",
    answer="At least five years.",
    gold_paragraph="1026.35(a)",
    gold_section="1026.35",
    evidence="must be kept for at least five years",
)


class FixedRetriever:
    name = "fixed"

    def __init__(self, paragraph_ids, cost=0.002) -> None:
        self.paragraph_ids, self.cost = paragraph_ids, cost

    async def retrieve(self, question):
        passages = [Passage(p, f"text of {p}") for p in self.paragraph_ids[:2]]
        return Retrieval(
            self.paragraph_ids, passages, latency_s=1.5, llm_calls=2, input_tokens=900,
            output_tokens=100, cost_usd=self.cost, trace={"sections": ["1026.35"]},
        )  # fmt: skip


def grader(answer="Five years (1026.35(a)).", verdict="correct"):
    def handler(system, prompt):
        if "You grade" in system:
            return {"reasoning": "matches the reference", "verdict": verdict}
        return answer

    return FakeLLM(handler)


# --- question generation -------------------------------------------------------------


def test_evidence_must_appear_in_the_paragraph():
    text = "(a) Requirement. An escrow account must be kept for at least five years."
    assert is_grounded("must be kept for at least FIVE years", text)
    assert not is_grounded("must be kept for at least ten years", text)
    assert not is_grounded("five years", text)  # too short to count as evidence


def test_sampling_is_seeded_and_spread_across_sections(corpus):
    first = sample_paragraphs(corpus, 4, seed=1, min_words=10, per_section=1)
    again = sample_paragraphs(corpus, 4, seed=1, min_words=10, per_section=1)
    assert first == again
    assert sorted(p.section for p in first) == ["1026.1", "1026.2", "1026.35"]
    assert sample_paragraphs(corpus, 4, seed=1, min_words=500) == []


async def test_generate_keeps_grounded_questions_and_balances_styles(corpus, tmp_path):
    def handler(system, prompt):
        text = prompt.split("\n\n", 1)[1]
        evidence = "this is not in the paragraph at all" if "1026.2 " in prompt else text[4:60]
        return {"question": " A question? ", "answer": "An answer.", "evidence": evidence}

    llm = FakeLLM(handler)
    # count=4 samples all six paragraphs; the two from § 1026.2 come back ungrounded.
    kept, rejected = await generate(corpus, llm, "claude-sonnet-5-5", 4, seed=3, min_words=10)

    assert rejected == 2
    assert {q.style for q in kept} == {"direct", "scenario"}
    assert [q.id for q in kept] == [f"q{i:03d}" for i in range(1, len(kept) + 1)]
    assert all(q.gold_section != "1026.2" and q.question == "A question?" for q in kept)
    assert any("Style: scenario" in call["system"] for call in llm.calls)

    save(kept, tmp_path / "questions.jsonl")
    assert load(tmp_path / "questions.jsonl") == kept


# --- evaluation ----------------------------------------------------------------------


def test_rank_of():
    assert rank_of(["a", "b", "c"], lambda x: x == "b") == 2
    assert rank_of(["a"], lambda x: x == "z") is None


async def test_record_holds_ranks_costs_answer_and_verdict(corpus):
    retriever = FixedRetriever(["1026.35(b)", "1026.35(a)", "1026.1(a)"])
    llm = grader()

    record = await evaluate_one(QUESTION, retriever, corpus, llm, Models())

    assert (record["rank"], record["section_rank"]) == (2, 1)
    assert record["verdict"] == "correct" and record["answer"].startswith("Five years")
    assert record["retrieval_cost_usd"] == 0.002 and record["retrieval_llm_calls"] == 2
    assert record["context_chars"] == len("text of 1026.35(b)") + len("text of 1026.35(a)")
    # The answer model saw the retrieved text; the judge saw the reference, not the text.
    assert "[1026.35(b)]\ntext of 1026.35(b)" in llm.calls[0]["prompt"]
    assert "Reference answer: At least five years." in llm.calls[1]["prompt"]
    assert "text of" not in llm.calls[1]["prompt"]


async def test_not_found_answer_is_scored_without_calling_the_judge(corpus):
    llm = grader(answer=NOT_FOUND)
    record = await evaluate_one(QUESTION, FixedRetriever(["1026.1(a)"]), corpus, llm, Models())

    assert (record["rank"], record["verdict"]) == (None, "not_found")
    assert len(llm.calls) == 1


async def test_empty_retrieval_makes_no_model_calls(corpus):
    llm = grader()
    record = await evaluate_one(QUESTION, FixedRetriever([]), corpus, llm, Models())

    assert record["verdict"] == "not_found" and record["context_chars"] == 0
    assert llm.calls == []


async def test_model_failure_is_recorded_as_a_miss(corpus):
    llm = FakeLLM(lambda system, prompt: ModelRefused("stopped with refusal"))
    record = await evaluate_one(QUESTION, FixedRetriever(["1026.35(a)"]), corpus, llm, Models())

    assert (record["verdict"], record["rank"]) == ("error", None)
    assert "ModelRefused" in record["error"]


async def test_run_writes_one_record_per_method_and_question(corpus, tmp_path):
    out = tmp_path / "records.jsonl"
    retrievers = [FixedRetriever(["1026.35(a)"]), FixedRetriever(["1026.1(a)"])]
    retrievers[1].name = "other"

    records = await run([QUESTION], retrievers, corpus, grader(), Models(), out)

    assert [r["method"] for r in records] == ["fixed", "other"]
    assert [json.loads(line)["rank"] for line in out.read_text().splitlines()] == [1, None]


# --- report --------------------------------------------------------------------------


def record(method, rank, verdict, style="direct", latency=0.5, cost=0.0, question="q1"):
    return {
        "question_id": question, "style": style, "method": method, "rank": rank,
        "section_rank": rank, "verdict": verdict, "retrieval_latency_s": latency,
        "retrieval_cost_usd": cost, "retrieval_input_tokens": 900,
        "retrieval_output_tokens": 100, "context_chars": 4000,
    }  # fmt: skip


def test_wilson_interval():
    low, high = report.wilson(8, 10)
    assert (round(low, 3), round(high, 3)) == (0.49, 0.943)
    assert report.wilson(0, 0) == (0.0, 0.0)
    assert report.wilson(10, 10)[1] == pytest.approx(1.0)


def test_summary_metrics():
    records = [
        record("m", 1, "correct"),
        record("m", 3, "correct"),
        record("m", 6, "incorrect"),
        record("m", None, "not_found"),
    ]
    summary = report.summarize(records)

    assert (summary["hit@1"], summary["hit@3"], summary["hit@5"]) == (0.25, 0.5, 0.5)
    assert summary["mrr@5"] == pytest.approx((1 + 1 / 3) / 4)
    assert (summary["answer_accuracy"], summary["not_found"]) == (0.5, 0.25)


def test_errors_count_as_misses_and_do_not_break_the_report():
    failed = {"question_id": "q2", "style": "direct", "method": "m", "rank": None,
              "section_rank": None, "verdict": "error", "error": "ModelRefused"}  # fmt: skip
    summary = report.summarize([record("m", 1, "correct"), failed])

    assert (summary["n"], summary["hit@1"], summary["errors"]) == (2, 0.5, 1)


def test_render_has_a_row_per_method_and_a_style_breakdown():
    records = [
        record("bm25", 1, "correct", latency=0.004),
        record("vectorless", 2, "incorrect", style="scenario", latency=2.4, cost=0.0012),
        record("bm25", None, "not_found", style="scenario", latency=0.004, question="q2"),
        record("vectorless", 1, "correct", latency=2.4, cost=0.0012, question="q2"),
    ]
    text = report.render(records)

    assert "Results over 2 questions." in text
    assert "| bm25 | 50% | 50% | 50% | 0.50 | 50% (9% to 91%) | 4 ms | $0 |" in text
    assert "| vectorless | 50% | 100% | 100% | 0.75 |" in text and "2.4 s | $0.00120 |" in text
    assert "direct: Hit@3 | direct: answer accuracy | scenario: Hit@3" in text


def test_navigation_table_separates_the_two_kinds_of_miss():
    def nav(rank, sections):
        return {**record("vectorless", rank, "correct"), "gold_paragraph": "1026.35(a)",
                "trace": {"sections": sections}}  # fmt: skip

    records = [
        nav(1, ["1026.35"]),
        nav(None, ["1026.2"]),  # never opened the right section
        nav(None, ["1026.35", "1026.2"]),  # opened it, picked the wrong paragraphs
        record("bm25", 1, "correct"),  # no trace, so not listed
    ]
    table = report.navigation_table(records)

    assert "| vectorless | 1 | 1 | 1 |" in table
    assert "bm25" not in table


def test_mcnemar_exact():
    assert report.mcnemar_exact(13, 1) == pytest.approx(0.00183, abs=1e-5)
    assert report.mcnemar_exact(5, 5) == 1.0
    assert report.mcnemar_exact(0, 0) == 1.0


def test_paired_table_counts_wins_and_losses_against_the_baseline():
    records = [
        record("hybrid", 2, "correct", question="q1"),
        record("hybrid", 1, "incorrect", question="q2"),
        record("vectorless", 1, "correct", question="q1"),  # better on Hit@1, same on answer
        record("vectorless", 3, "correct", question="q2"),  # worse on Hit@1, better on answer
    ]
    table = report.paired_table(records)

    assert "| vectorless | 1 / 1 | 1.000 | 1 / 0 | 1.000 |" in table
    assert report.paired_table([record("bm25", 1, "correct")]) == ""
