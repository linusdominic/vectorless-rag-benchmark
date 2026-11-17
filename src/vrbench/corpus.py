"""The corpus: Regulation Z (12 CFR Part 1026) parsed from eCFR XML into a tree.

    Part -> Subpart -> Section -> top-level paragraph, for example 1026.43(c)

Top-level paragraphs are the unit every retriever is scored on.
"""

import gzip
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

PART = "1026"
AS_OF = "2026-01-01"
SOURCE_URL = f"https://www.ecfr.gov/api/versioner/v1/full/{AS_OF}/title-12.xml?part={PART}"

_DESIGNATOR = re.compile(r"^\(([A-Za-z0-9]+)\)")
_WHITESPACE = re.compile(r"\s+")
# A single lowercase letter that could also be a roman numeral, with the numeral after it.
_ROMAN_NEXT = {"i": "ii", "v": "vi", "x": "xi"}


@dataclass(frozen=True)
class Paragraph:
    id: str  # "1026.43(c)", "1026.3 intro", or "1026.5" for a section with no lettered parts
    section: str
    label: str  # "(c)", "intro" or ""
    heading: str
    text: str


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    subpart: str
    subpart_title: str
    paragraphs: tuple[Paragraph, ...]

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)


@dataclass
class Corpus:
    title: str
    source_url: str
    as_of: str
    sections: list[Section]
    _paragraphs: dict[str, Paragraph] = field(default_factory=dict, repr=False)
    _sections: dict[str, Section] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._sections = {s.id: s for s in self.sections}
        self._paragraphs = {p.id: p for s in self.sections for p in s.paragraphs}

    @property
    def paragraphs(self) -> list[Paragraph]:
        return list(self._paragraphs.values())

    def section(self, section_id: str) -> Section | None:
        return self._sections.get(section_id)

    def paragraph(self, paragraph_id: str) -> Paragraph | None:
        return self._paragraphs.get(paragraph_id)

    def save(self, path: Path) -> None:
        payload = {
            "title": self.title,
            "source_url": self.source_url,
            "as_of": self.as_of,
            "sections": [asdict(s) for s in self.sections],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        raw = json.loads(path.read_text(encoding="utf-8"))
        sections = [
            Section(**{**s, "paragraphs": tuple(Paragraph(**p) for p in s["paragraphs"])})
            for s in raw["sections"]
        ]
        return cls(raw["title"], raw["source_url"], raw["as_of"], sections)


def fetch_xml(url: str = SOURCE_URL) -> str:
    """Download the part from eCFR. The API only answers requests that accept compression."""
    request = urllib.request.Request(
        url, headers={"Accept-Encoding": "gzip", "User-Agent": "vrbench/0.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    return body.decode("utf-8")


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _designator(text: str) -> str | None:
    match = _DESIGNATOR.match(text)
    return match.group(1) if match else None


def _heading(element: ET.Element) -> str:
    """The italic heading that directly follows a paragraph's designator, if it has one."""
    first = next(iter(element), None)
    lead = _clean(element.text or "")
    if first is not None and first.tag == "I" and _DESIGNATOR.fullmatch(lead):
        return _clean("".join(first.itertext())).rstrip(".")
    return ""


def _starts_top_level(designators: list[str | None], index: int, current: str | None) -> bool:
    """Decide whether the paragraph at ``index`` opens a new top-level paragraph such as (c).

    CFR numbering is (a)(1)(i)(A), so a lone lowercase letter is top-level unless it is
    really a roman numeral. "(i)" is ambiguous after "(h)". It is top-level when "(1)"
    comes next, because only a letter has numbered children. Otherwise it is the roman
    numeral when "(ii)" follows before another "(i)" or the next letter does.
    """
    designator = designators[index]
    if designator is None or not re.fullmatch(r"[a-z]", designator):
        return False
    if current is not None and designator <= current:
        return False
    if designator not in _ROMAN_NEXT:
        return True
    expected = "a" if current is None else chr(ord(current) + 1)
    if designator != expected:
        return False
    following_letter = chr(ord(designator) + 1)
    if designators[index + 1 : index + 2] == ["1"]:
        return True
    for later in designators[index + 1 :]:
        if later == _ROMAN_NEXT[designator]:
            return False
        if later in (designator, following_letter):
            return True
    return True


def _parse_section(element: ET.Element, subpart: str, subpart_title: str) -> Section:
    section_id = element.attrib["N"]
    head = _clean("".join(element.find("HEAD").itertext()))
    title = head.removeprefix("§").strip().removeprefix(section_id).strip().rstrip(".")

    rows = [(_clean("".join(p.itertext())), _heading(p)) for p in element.iter("P")]
    rows = [(text, heading) for text, heading in rows if text]
    designators = [_designator(text) for text, _ in rows]

    groups: list[tuple[str | None, str, list[str]]] = []  # (letter, heading, lines)
    current: str | None = None
    for index, (text, heading) in enumerate(rows):
        if _starts_top_level(designators, index, current):
            current = designators[index]
            groups.append((current, heading, []))
        elif not groups:
            groups.append((None, "", []))
        groups[-1][2].append(text)

    lettered = any(letter is not None for letter, _, _ in groups)
    paragraphs = []
    for letter, heading, lines in groups:
        if letter is not None:
            paragraph_id, label = f"{section_id}({letter})", f"({letter})"
        elif lettered:
            paragraph_id, label = f"{section_id} intro", "intro"
        else:
            paragraph_id, label = section_id, ""
        paragraphs.append(Paragraph(paragraph_id, section_id, label, heading, "\n".join(lines)))
    return Section(section_id, title, subpart, subpart_title, tuple(paragraphs))


def parse(xml: str) -> Corpus:
    """Parse the regulation's sections. Appendices and the official commentary are left out."""
    root = ET.fromstring(xml)
    title = _clean("".join(root.find("HEAD").itertext()))
    sections = []
    for subpart in root.iter("DIV6"):
        head = _clean("".join(subpart.find("HEAD").itertext()))
        subpart_title = head.split("—", 1)[-1].strip() if "—" in head else head
        for element in subpart.iter("DIV8"):
            sections.append(_parse_section(element, subpart.attrib["N"], subpart_title))
    return Corpus(title, SOURCE_URL, AS_OF, sections)
