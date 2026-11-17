"""What every retrieval method returns, so they can be scored the same way."""

from dataclasses import dataclass, field
from typing import Protocol

from vrbench.corpus import Corpus


@dataclass(frozen=True)
class Passage:
    paragraph_id: str
    text: str


@dataclass
class Retrieval:
    paragraph_ids: list[str]  # ranked best first, no duplicates
    passages: list[Passage]  # the text handed to the answering model
    latency_s: float
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    trace: dict = field(default_factory=dict)


class Retriever(Protocol):
    name: str

    async def retrieve(self, question: str) -> Retrieval: ...


def unique(items: list[str]) -> list[str]:
    """Drop repeats, keeping the first occurrence of each item."""
    return list(dict.fromkeys(items))


def whole_paragraphs(
    corpus: Corpus, paragraph_ids: list[str], limit: int = 3, max_chars: int = 24_000
) -> list[Passage]:
    """The full text of the top paragraphs, in rank order, until the character budget is used."""
    passages: list[Passage] = []
    remaining = max_chars
    for paragraph_id in paragraph_ids[:limit]:
        if remaining <= 0:
            break
        paragraph = corpus.paragraph(paragraph_id)
        section = corpus.section(paragraph.section)
        text = f"§ {section.id} {section.title} {paragraph.label}\n{paragraph.text}"
        passages.append(Passage(paragraph_id, text[:remaining]))
        remaining -= len(text)
    return passages
