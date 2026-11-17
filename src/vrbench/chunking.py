"""Chunks for the BM25, dense and hybrid baselines.

Each top-level paragraph is split on its own sub-paragraph boundaries, and every chunk
carries a header naming its section and paragraph, so a chunk is never anonymous text.
"""

import re
from dataclasses import dataclass

from vrbench.corpus import Corpus, Paragraph, Section

_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


@dataclass(frozen=True)
class Chunk:
    paragraph_id: str
    text: str


def _split_long(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(line):
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def header(section: Section, paragraph: Paragraph) -> str:
    parts = [f"§ {section.id} {section.title}"]
    if paragraph.label:
        parts.append(f"{paragraph.label} {paragraph.heading}".strip())
    return " | ".join(parts)


def chunk_paragraph(section: Section, paragraph: Paragraph, max_chars: int = 1200) -> list[Chunk]:
    """Pack the paragraph's lines into chunks of at most ``max_chars`` of body text."""
    head = header(section, paragraph)
    bodies: list[str] = []
    current: list[str] = []
    size = 0
    for line in paragraph.text.split("\n"):
        for piece in _split_long(line, max_chars):
            if current and size + len(piece) > max_chars:
                bodies.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        bodies.append("\n".join(current))
    return [Chunk(paragraph.id, f"{head}\n{body}") for body in bodies]


def chunk_corpus(corpus: Corpus, max_chars: int = 1200) -> list[Chunk]:
    return [
        chunk
        for section in corpus.sections
        for paragraph in section.paragraphs
        for chunk in chunk_paragraph(section, paragraph, max_chars)
    ]
