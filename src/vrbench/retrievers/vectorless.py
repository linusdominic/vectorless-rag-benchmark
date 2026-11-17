"""Vectorless retrieval: a model reads the document's outline and navigates to the answer.

There are no embeddings and no chunks. The index is the regulation's own structure:

    step 1  show the table of contents (59 sections)      -> the model picks sections
    step 2  show the outline of the paragraphs in those   -> the model picks paragraphs

The whole text of the chosen paragraphs is then passed to the answering model. Optional
model-written summaries can be attached to each outline entry at indexing time.
"""

from dataclasses import dataclass

from vrbench.corpus import Corpus, Paragraph, Section
from vrbench.llm import LLM, LLMResult
from vrbench.retrievers.base import Retrieval, unique, whole_paragraphs

SECTION_SYSTEM = """\
You are the retrieval step of a question-answering system for Regulation Z, the Truth in \
Lending regulation (12 CFR Part 1026). You are shown its table of contents. Choose the \
sections most likely to contain the answer to the question.

Return up to {limit} section ids, most likely first. Use ids exactly as written, for \
example "1026.19". Give one sentence of reasoning first.

{outline}"""

PARAGRAPH_SYSTEM = """\
You are the retrieval step of a question-answering system for Regulation Z, the Truth in \
Lending regulation (12 CFR Part 1026). You are shown the outline of the sections chosen \
for a question. Choose the paragraphs most likely to contain the answer.

Return up to {limit} paragraph ids, most likely first. Use ids exactly as written, for \
example "1026.19(e)". Give one sentence of reasoning first."""


def _schema(key: str) -> dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            key: {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", key],
        "additionalProperties": False,
    }


def _opening_words(text: str, count: int = 22) -> str:
    words = text.split()
    return " ".join(words[:count]) + ("..." if len(words) > count else "")


def table_of_contents(corpus: Corpus, summaries: dict[str, str] | None = None) -> str:
    lines: list[str] = []
    subpart = None
    for section in corpus.sections:
        if section.subpart != subpart:
            subpart = section.subpart
            lines.append(f"\nSubpart {subpart}: {section.subpart_title}")
        lines.append(f"  {section.id} {section.title}")
        if summaries and section.id in summaries:
            lines.append(f"      {summaries[section.id]}")
    return "\n".join(lines).strip()


def _body(paragraph: Paragraph) -> str:
    """The paragraph's text without its leading "(c) Heading." so the outline does not repeat it."""
    lead = f"{paragraph.label} {paragraph.heading}" if paragraph.heading else paragraph.label
    return paragraph.text.removeprefix(lead).lstrip(" .—-")


def describe(paragraph: Paragraph, summaries: dict[str, str] | None = None) -> str:
    """One outline entry: id, heading, then a summary or the paragraph's opening words."""
    words = len(paragraph.text.split())
    detail = (summaries or {}).get(paragraph.id) or _opening_words(_body(paragraph))
    heading = f"{paragraph.heading}. " if paragraph.heading else ""
    return f"  {paragraph.id} [{words} words] {heading}{detail}"


def section_outline(section: Section, summaries: dict[str, str] | None = None) -> str:
    lines = [f"§ {section.id} {section.title}"]
    lines += [describe(paragraph, summaries) for paragraph in section.paragraphs]
    return "\n".join(lines)


@dataclass
class VectorlessRetriever:
    corpus: Corpus
    llm: LLM
    model: str
    name: str = "vectorless"
    summaries: dict[str, str] | None = None
    max_sections: int = 3
    max_paragraphs: int = 5
    context_paragraphs: int = 3
    context_chars: int = 24_000

    async def retrieve(self, question: str) -> Retrieval:
        calls: list[LLMResult] = []

        first = await self.llm.complete(
            model=self.model,
            system=SECTION_SYSTEM.format(
                limit=self.max_sections, outline=table_of_contents(self.corpus, self.summaries)
            ),
            prompt=f"Question: {question}",
            schema=_schema("sections"),
        )
        calls.append(first)
        chosen = first.json()
        # The model may return an id that does not exist. Keep only real ones.
        sections = [
            section
            for section_id in unique(chosen["sections"])[: self.max_sections]
            if (section := self.corpus.section(section_id.strip().removeprefix("§").strip()))
        ]
        trace = {"section_reasoning": chosen["reasoning"], "sections": [s.id for s in sections]}

        paragraph_ids: list[str] = []
        if sections:
            outline = "\n\n".join(section_outline(s, self.summaries) for s in sections)
            second = await self.llm.complete(
                model=self.model,
                system=PARAGRAPH_SYSTEM.format(limit=self.max_paragraphs),
                prompt=f"{outline}\n\nQuestion: {question}",
                schema=_schema("paragraphs"),
            )
            calls.append(second)
            picked = second.json()
            allowed = {p.id for s in sections for p in s.paragraphs}
            paragraph_ids = [
                paragraph_id
                for paragraph_id in unique([p.strip() for p in picked["paragraphs"]])
                if paragraph_id in allowed
            ][: self.max_paragraphs]
            trace["paragraph_reasoning"] = picked["reasoning"]
        trace["paragraphs"] = paragraph_ids

        return Retrieval(
            paragraph_ids=paragraph_ids,
            passages=whole_paragraphs(
                self.corpus, paragraph_ids, self.context_paragraphs, self.context_chars
            ),
            latency_s=sum(call.latency_s for call in calls),
            llm_calls=len(calls),
            input_tokens=sum(call.input_tokens for call in calls),
            output_tokens=sum(call.output_tokens for call in calls),
            cost_usd=sum(call.cost_usd for call in calls),
            trace=trace,
        )
