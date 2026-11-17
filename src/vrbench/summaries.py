"""Optional indexing step: a short model-written summary for each outline entry.

This is the only indexing cost of the vectorless method, and the benchmark measures
whether it is worth paying by running the method with and without it.
"""

import asyncio

from vrbench.corpus import Corpus, Paragraph, Section
from vrbench.llm import LLM, ModelRefused

PARAGRAPH_SYSTEM = """\
You write index entries for Regulation Z (12 CFR Part 1026) so that a reader scanning an \
outline can tell which paragraph answers a question. In at most 45 words, list the specific \
topics, rules, thresholds, deadlines and defined terms the paragraph covers, as noun \
phrases separated by semicolons. No preamble and no full sentences."""

SECTION_SYSTEM = """\
You write index entries for Regulation Z (12 CFR Part 1026). Given a section's title and \
the index entries of its paragraphs, write one entry of at most 40 words that says what \
the section covers, as noun phrases separated by semicolons. No preamble."""

MIN_WORDS = 40  # shorter paragraphs are shown by their opening words instead
MAX_CHARS = 24_000
MAX_TOKENS = 3000  # thinking counts toward the limit, so leave headroom over the ~60 needed


async def _entry(llm: LLM, model: str, system: str, prompt: str) -> str:
    """One index entry, or an empty string if the model gave no usable reply."""
    try:
        result = await llm.complete(
            model=model, system=system, prompt=prompt, max_tokens=MAX_TOKENS
        )
    except ModelRefused:
        return ""
    return " ".join(result.text.split())


async def _paragraph(llm: LLM, model: str, section: Section, paragraph: Paragraph) -> str:
    prompt = f"§ {section.id} {section.title} {paragraph.label}\n\n{paragraph.text[:MAX_CHARS]}"
    return await _entry(llm, model, PARAGRAPH_SYSTEM, prompt)


async def _section(llm: LLM, model: str, section: Section, entries: dict[str, str]) -> str:
    lines = [
        f"{p.label} {p.heading}: {entries.get(p.id, p.text[:200])}" for p in section.paragraphs
    ]
    prompt = f"§ {section.id} {section.title}\n\n" + "\n".join(lines)
    return await _entry(llm, model, SECTION_SYSTEM, prompt)


async def summarize(corpus: Corpus, llm: LLM, model: str) -> dict[str, str]:
    """Summaries keyed by paragraph id and by section id."""
    targets = [
        (section, paragraph)
        for section in corpus.sections
        for paragraph in section.paragraphs
        if len(paragraph.text.split()) >= MIN_WORDS
    ]
    texts = await asyncio.gather(*(_paragraph(llm, model, s, p) for s, p in targets))
    entries = {
        paragraph.id: text for (_, paragraph), text in zip(targets, texts, strict=True) if text
    }

    section_texts = await asyncio.gather(
        *(_section(llm, model, section, entries) for section in corpus.sections)
    )
    entries.update(
        {
            section.id: text
            for section, text in zip(corpus.sections, section_texts, strict=True)
            if text
        }
    )
    return entries
