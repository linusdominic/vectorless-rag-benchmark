"""Runs every retrieval method over the question set and scores it.

For each question and method this records where the gold paragraph ranked, what the
retrieval cost and how long it took, then has a model answer from the retrieved text and a
second model grade that answer against the reference.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from vrbench.corpus import Corpus
from vrbench.llm import LLM, ModelRefused
from vrbench.questions import Question
from vrbench.retrievers.base import Retrieval, Retriever

NOT_FOUND = "NOT FOUND"

ANSWER_SYSTEM = f"""\
You answer questions about Regulation Z, the Truth in Lending regulation (12 CFR Part 1026), \
using only the excerpts you are given. Answer in at most three sentences and name the \
paragraph ids you relied on in parentheses. If the excerpts do not contain the answer, \
reply with exactly: {NOT_FOUND}"""

JUDGE_SYSTEM = """\
You grade an answer to a question about Regulation Z against a reference answer.

correct: the answer states the same rule, figure, deadline or condition as the reference. \
Extra detail that does not contradict the reference is fine.
incorrect: the answer contradicts the reference, misses its key point, or is too vague to \
tell.

Give one sentence of reasoning, then the verdict."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
    },
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Models:
    answer: str = "claude-haiku-5-5"
    judge: str = "claude-sonnet-5-5"  # the grader is a stronger model than the answerer


def rank_of(items: list[str], wanted) -> int | None:
    """1-based position of the first item that satisfies ``wanted``, or None."""
    return next((i for i, item in enumerate(items, start=1) if wanted(item)), None)


async def answer_question(
    llm: LLM, model: str, question: str, retrieval: Retrieval
) -> tuple[str, float]:
    if not retrieval.passages:
        return NOT_FOUND, 0.0
    excerpts = "\n\n".join(f"[{p.paragraph_id}]\n{p.text}" for p in retrieval.passages)
    result = await llm.complete(
        model=model,
        system=ANSWER_SYSTEM,
        prompt=f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}",
        max_tokens=1500,
    )
    return result.text.strip(), result.cost_usd


async def _judge(llm: LLM, model: str, question: Question, answer: str) -> tuple[str, str, float]:
    if NOT_FOUND in answer:
        return "not_found", "The system said the answer was not in the retrieved text.", 0.0
    result = await llm.complete(
        model=model,
        system=JUDGE_SYSTEM,
        prompt=(
            f"Question: {question.question}\n\nReference answer: {question.answer}\n\n"
            f"Answer to grade: {answer}"
        ),
        schema=JUDGE_SCHEMA,
        max_tokens=1000,
    )
    graded = result.json()
    return graded["verdict"], graded["reasoning"], result.cost_usd


async def evaluate_one(
    question: Question, retriever: Retriever, corpus: Corpus, llm: LLM, models: Models
) -> dict:
    record = {
        "question_id": question.id,
        "style": question.style,
        "method": retriever.name,
        "gold_paragraph": question.gold_paragraph,
    }
    try:
        retrieval = await retriever.retrieve(question.question)
        answer, answer_cost = await answer_question(
            llm, models.answer, question.question, retrieval
        )
        verdict, reasoning, judge_cost = await _judge(llm, models.judge, question, answer)
    except (ModelRefused, json.JSONDecodeError, KeyError) as error:
        # A method that fails to produce a usable result has missed the question.
        failure = {"rank": None, "section_rank": None, "verdict": "error"}
        return {**record, **failure, "error": f"{type(error).__name__}: {error}"}

    def in_gold_section(paragraph_id: str) -> bool:
        paragraph = corpus.paragraph(paragraph_id)
        return paragraph is not None and paragraph.section == question.gold_section

    return {
        **record,
        "retrieved": retrieval.paragraph_ids[:10],
        "rank": rank_of(retrieval.paragraph_ids, lambda p: p == question.gold_paragraph),
        "section_rank": rank_of(retrieval.paragraph_ids, in_gold_section),
        "retrieval_latency_s": round(retrieval.latency_s, 4),
        "retrieval_llm_calls": retrieval.llm_calls,
        "retrieval_input_tokens": retrieval.input_tokens,
        "retrieval_output_tokens": retrieval.output_tokens,
        "retrieval_cost_usd": round(retrieval.cost_usd, 6),
        "context_chars": sum(len(p.text) for p in retrieval.passages),
        "answer": answer,
        "answer_cost_usd": round(answer_cost, 6),
        "verdict": verdict,
        "judge_reasoning": reasoning,
        "judge_cost_usd": round(judge_cost, 6),
        "trace": retrieval.trace,
    }


async def run(
    questions: list[Question],
    retrievers: list[Retriever],
    corpus: Corpus,
    llm: LLM,
    models: Models,
    out_path: Path,
) -> list[dict]:
    records = await asyncio.gather(
        *(
            evaluate_one(question, retriever, corpus, llm, models)
            for retriever in retrievers
            for question in questions
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return list(records)
