"""Builds the evaluation set: questions with a known gold paragraph.

Questions are written by a model from one sampled paragraph each, in two styles, and kept
only when the evidence the model quotes really appears in that paragraph.
"""

import asyncio
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from vrbench.corpus import Corpus, Paragraph
from vrbench.llm import LLM, ModelRefused

STYLES = {
    "direct": (
        "Style: direct. Ask the way a compliance analyst would, using the regulation's own terms."
    ),
    "scenario": (
        "Style: scenario. Describe a concrete situation in plain everyday language, the way "
        "a borrower or a loan officer would. Avoid the regulation's defined terms and "
        "distinctive phrases wherever a plain-language paraphrase exists."
    ),
}

SYSTEM = """\
You write evaluation questions for a retrieval benchmark over Regulation Z, the Truth in \
Lending regulation (12 CFR Part 1026). You are given one paragraph of the regulation. Write \
one question that this paragraph answers, and a reference answer.

Rules:
- The question has one specific, checkable answer found in this paragraph: a number, a \
deadline, a threshold, a condition or a requirement.
- The question stands on its own. Do not mention section or paragraph numbers and do not \
refer to "this paragraph" or "the regulation above".
- The reference answer is one or two sentences and states only what the paragraph says.
- evidence is the sentence or clause from the paragraph that contains the answer, copied \
word for word.

{style}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["question", "answer", "evidence"],
    "additionalProperties": False,
}

MAX_CHARS = 12_000


@dataclass(frozen=True)
class Question:
    id: str
    style: str
    question: str
    answer: str
    gold_paragraph: str
    gold_section: str
    evidence: str


def sample_paragraphs(
    corpus: Corpus, count: int, seed: int, min_words: int = 80, per_section: int = 2
) -> list[Paragraph]:
    """A seeded sample of substantive paragraphs, spread out so no section dominates."""
    eligible = [
        p
        for p in corpus.paragraphs
        if p.label.startswith("(")
        and len(p.text.split()) >= min_words
        and "[Reserved]" not in p.text[:80]
    ]
    random.Random(seed).shuffle(eligible)
    picked: list[Paragraph] = []
    used: dict[str, int] = {}
    for paragraph in eligible:
        if used.get(paragraph.section, 0) < per_section:
            picked.append(paragraph)
            used[paragraph.section] = used.get(paragraph.section, 0) + 1
        if len(picked) == count:
            break
    return picked


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_grounded(evidence: str, paragraph_text: str) -> bool:
    """True when the quoted evidence appears in the paragraph, ignoring case and punctuation."""
    quote = _normalize(evidence)
    return len(quote.split()) >= 4 and quote in _normalize(paragraph_text)


async def _write(
    llm: LLM, model: str, corpus: Corpus, paragraph: Paragraph, style: str
) -> Question | None:
    section = corpus.section(paragraph.section)
    text = paragraph.text[:MAX_CHARS]
    try:
        result = await llm.complete(
            model=model,
            system=SYSTEM.format(style=STYLES[style]),
            prompt=f"§ {section.id} {section.title} {paragraph.label}\n\n{text}",
            schema=SCHEMA,
            max_tokens=1500,
        )
        item = result.json()
    except (ModelRefused, json.JSONDecodeError):
        return None
    if not is_grounded(item["evidence"], text):
        return None
    return Question(
        id="",
        style=style,
        question=item["question"].strip(),
        answer=item["answer"].strip(),
        gold_paragraph=paragraph.id,
        gold_section=paragraph.section,
        evidence=item["evidence"].strip(),
    )


async def generate(
    corpus: Corpus,
    llm: LLM,
    model: str,
    count: int,
    seed: int,
    spare: float = 1.25,
    min_words: int = 80,
) -> tuple[list[Question], int]:
    """Write ``count`` questions, half in each style. Returns them and the number rejected."""
    paragraphs = sample_paragraphs(corpus, int(count * spare) + 1, seed, min_words=min_words)
    styles = list(STYLES)
    drafts = await asyncio.gather(
        *(
            _write(llm, model, corpus, paragraph, styles[index % len(styles)])
            for index, paragraph in enumerate(paragraphs)
        )
    )
    rejected = sum(draft is None for draft in drafts)
    kept: list[Question] = []
    for style in styles:  # equal numbers of each style
        of_style = [d for d in drafts if d is not None and d.style == style]
        kept += of_style[: count // len(styles)]
    kept.sort(key=lambda q: (q.gold_section, q.gold_paragraph, q.style))
    numbered = [Question(**{**asdict(q), "id": f"q{i:03d}"}) for i, q in enumerate(kept, start=1)]
    return numbered, rejected


def save(questions: list[Question], path: Path) -> None:
    path.write_text("".join(json.dumps(asdict(q), ensure_ascii=False) + "\n" for q in questions))


def load(path: Path) -> list[Question]:
    return [Question(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
