"""Shared test doubles: a scripted model and a small synthetic regulation."""

import json

import pytest

from vrbench.corpus import Corpus, parse
from vrbench.llm import LLMResult


class FakeLLM:
    """Stands in for the model client. ``handler(system, prompt)`` returns text or a dict."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict] = []

    async def complete(self, *, model, system, prompt, schema=None, max_tokens=2000, effort="low"):
        self.calls.append({"model": model, "system": system, "prompt": prompt, "schema": schema})
        reply = self.handler(system, prompt)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResult(text, input_tokens=100, output_tokens=20, cost_usd=0.001, latency_s=0.5)


def section_xml(number: str, title: str, paragraphs: list[str]) -> str:
    body = "".join(f"<P>{p}</P>" for p in paragraphs)
    return f'<DIV8 N="{number}" TYPE="SECTION"><HEAD>§ {number} {title}.</HEAD>{body}</DIV8>'


def part_xml(subparts: dict[str, tuple[str, list[str]]]) -> str:
    divisions = "".join(
        f'<DIV6 N="{letter}" TYPE="SUBPART"><HEAD>Subpart {letter}—{title}</HEAD>'
        f"{''.join(sections)}</DIV6>"
        for letter, (title, sections) in subparts.items()
    )
    appendix = '<DIV9 N="Appendix A" TYPE="APPENDIX"><HEAD>Appendix A</HEAD><P>Ignored.</P></DIV9>'
    return (
        '<DIV5 N="1026" TYPE="PART"><HEAD>PART 1026—TRUTH IN LENDING (REGULATION Z)</HEAD>'
        f"{divisions}{appendix}</DIV5>"
    )


@pytest.fixture
def corpus() -> Corpus:
    """Three sections in two subparts, about refunds, late fees and escrow."""
    xml = part_xml(
        {
            "A": (
                "General",
                [
                    section_xml(
                        "1026.1",
                        "Refunds",
                        [
                            "(a) <I>Refund deadline.</I> A creditor must refund a credit "
                            "balance within seven business days of a written request.",
                            "(b) <I>Unclaimed balances.</I> After six months the creditor "
                            "must make a good faith effort to return the balance.",
                        ],
                    ),
                    section_xml(
                        "1026.2",
                        "Late fees",
                        [
                            "(a) <I>Limit.</I> A late payment penalty may not exceed the "
                            "minimum payment that was due.",
                            "(b) <I>Timing.</I> A fee may be charged only after the "
                            "payment is fifteen days overdue.",
                        ],
                    ),
                ],
            ),
            "E": (
                "Home Mortgages",
                [
                    section_xml(
                        "1026.35",
                        "Escrow accounts",
                        [
                            "(a) <I>Requirement.</I> An escrow account for property taxes "
                            "and insurance must be kept for at least five years.",
                            "(b) <I>Exemptions.</I> Loans secured by shares in a "
                            "cooperative are exempt from the escrow requirement.",
                        ],
                    )
                ],
            ),
        }
    )
    return parse(xml)
