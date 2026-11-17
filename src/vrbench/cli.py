"""Command line: fetch the corpus, build the index and questions, run and report."""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from vrbench import questions as question_set
from vrbench import report
from vrbench.chunking import chunk_corpus
from vrbench.corpus import Corpus, fetch_xml, parse
from vrbench.evaluate import Models, answer_question, run
from vrbench.llm import LLM
from vrbench.retrievers.chunked import build_chunk_retrievers
from vrbench.retrievers.vectorless import VectorlessRetriever
from vrbench.summaries import summarize

DATA = Path("data")
RESULTS = Path("results")
CORPUS = DATA / "regz.json"
SUMMARIES = DATA / "summaries.json"
QUESTIONS = DATA / "questions.jsonl"
RECORDS = RESULTS / "records.jsonl"
REPORT = RESULTS / "report.md"

NAVIGATOR = os.environ.get("VRBENCH_NAVIGATOR_MODEL", "claude-haiku-5-5")
WRITER = os.environ.get("VRBENCH_WRITER_MODEL", "claude-sonnet-5-5")
METHODS = ["bm25", "dense", "hybrid", "hybrid+parents", "vectorless", "vectorless+summaries"]


def _llm(args: argparse.Namespace) -> LLM:
    return LLM(budget_usd=args.budget, ledger_path=RESULTS / "spend.json")


def _retrievers(corpus: Corpus, llm: LLM, names: list[str]) -> list:
    chunked = build_chunk_retrievers(
        chunk_corpus(corpus),
        cache_dir=Path(".cache"),
        with_dense=bool({"dense", "hybrid", "hybrid+parents"} & set(names)),
        corpus=corpus,
    )
    available = {r.name: r for r in chunked}
    available["vectorless"] = VectorlessRetriever(corpus, llm, NAVIGATOR)
    if SUMMARIES.exists():
        available["vectorless+summaries"] = VectorlessRetriever(
            corpus,
            llm,
            NAVIGATOR,
            name="vectorless+summaries",
            summaries=json.loads(SUMMARIES.read_text()),
        )
    missing = [name for name in names if name not in available]
    if missing:
        raise SystemExit(f"Unavailable methods: {missing}. Run `vrbench summarize` first?")
    return [available[name] for name in names]


def cmd_fetch(args: argparse.Namespace) -> None:
    corpus = parse(fetch_xml())
    DATA.mkdir(exist_ok=True)
    corpus.save(CORPUS)
    words = sum(len(p.text.split()) for p in corpus.paragraphs)
    print(f"{len(corpus.sections)} sections, {len(corpus.paragraphs)} paragraphs, {words:,} words")


def cmd_summarize(args: argparse.Namespace) -> None:
    llm = _llm(args)
    entries = asyncio.run(summarize(Corpus.load(CORPUS), llm, NAVIGATOR))
    SUMMARIES.write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    print(f"{len(entries)} summaries written. Total spend so far: ${llm.ledger.total:.4f}")


def cmd_questions(args: argparse.Namespace) -> None:
    llm = _llm(args)
    kept, rejected = asyncio.run(
        question_set.generate(Corpus.load(CORPUS), llm, WRITER, args.count, args.seed)
    )
    question_set.save(kept, QUESTIONS)
    print(f"{len(kept)} questions kept, {rejected} rejected as ungrounded.")
    print(f"Total spend so far: ${llm.ledger.total:.4f}")


def cmd_run(args: argparse.Namespace) -> None:
    llm = _llm(args)
    corpus = Corpus.load(CORPUS)
    items = question_set.load(QUESTIONS)[: args.limit]
    retrievers = _retrievers(corpus, llm, args.methods)
    out = Path(args.out)
    records = asyncio.run(run(items, retrievers, corpus, llm, Models(), out))
    print(report.render(records))
    print(f"\n{len(records)} records in {out}. Total spend so far: ${llm.ledger.total:.4f}")


def cmd_report(args: argparse.Namespace) -> None:
    text = report.render(report.load_records(RECORDS))
    if args.write:
        REPORT.write_text(text + "\n")
    print(text)


def cmd_ask(args: argparse.Namespace) -> None:
    llm = _llm(args)
    corpus = Corpus.load(CORPUS)
    retriever = _retrievers(corpus, llm, [args.method])[0]

    async def go() -> None:
        retrieval = await retriever.retrieve(args.question)
        for key, value in retrieval.trace.items():
            print(f"{key}: {value}")
        print(f"retrieved: {retrieval.paragraph_ids[:5]}")
        print(f"retrieval: {retrieval.latency_s:.2f} s, ${retrieval.cost_usd:.5f}\n")
        answer, _ = await answer_question(llm, Models().answer, args.question, retrieval)
        print(answer)

    asyncio.run(go())


def cmd_spend(args: argparse.Namespace) -> None:
    path = RESULTS / "spend.json"
    print(path.read_text() if path.exists() else "Nothing spent yet.")


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(prog="vrbench", description=__doc__)
    parser.add_argument("--budget", type=float, default=5.0, help="spending cap in USD")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("fetch", help="download and parse Regulation Z").set_defaults(
        func=cmd_fetch
    )
    commands.add_parser("summarize", help="write outline summaries (paid)").set_defaults(
        func=cmd_summarize
    )

    make_questions = commands.add_parser("questions", help="generate the question set (paid)")
    make_questions.add_argument("--count", type=int, default=60)
    make_questions.add_argument("--seed", type=int, default=7)
    make_questions.set_defaults(func=cmd_questions)

    bench = commands.add_parser("run", help="run the benchmark (paid)")
    bench.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    bench.add_argument("--limit", type=int, default=None, help="only the first N questions")
    bench.add_argument("--out", default=str(RECORDS))
    bench.set_defaults(func=cmd_run)

    tables = commands.add_parser("report", help="print the tables from saved records")
    tables.add_argument("--write", action="store_true", help="also update results/report.md")
    tables.set_defaults(func=cmd_report)

    ask = commands.add_parser("ask", help="answer one question and show the trace (paid)")
    ask.add_argument("question")
    ask.add_argument("--method", default="vectorless", choices=METHODS)
    ask.set_defaults(func=cmd_ask)

    commands.add_parser("spend", help="show the spending ledger").set_defaults(func=cmd_spend)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
