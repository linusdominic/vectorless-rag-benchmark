from tests.conftest import part_xml, section_xml
from vrbench.corpus import Corpus, parse


def parse_section(paragraphs: list[str]):
    xml = part_xml({"A": ("General", [section_xml("1026.9", "Example rules", paragraphs)])})
    return parse(xml).sections[0]


def letters_up_to(last: str) -> list[str]:
    return [
        f"({chr(code)}) <I>Rule {chr(code)}.</I> Text." for code in range(ord("a"), ord(last) + 1)
    ]


def test_structure_titles_and_ids(corpus):
    assert [s.id for s in corpus.sections] == ["1026.1", "1026.2", "1026.35"]
    escrow = corpus.section("1026.35")
    assert (escrow.title, escrow.subpart, escrow.subpart_title) == (
        "Escrow accounts",
        "E",
        "Home Mortgages",
    )
    paragraph = corpus.paragraph("1026.35(a)")
    assert (paragraph.label, paragraph.heading) == ("(a)", "Requirement")
    assert paragraph.text.startswith("(a) Requirement. An escrow account")


def test_appendices_are_left_out(corpus):
    assert all("Ignored" not in p.text for p in corpus.paragraphs)


def test_nested_items_stay_inside_their_top_level_paragraph():
    section = parse_section(
        [
            "(a) <I>First.</I> (1) A numbered item:",
            "(i) A roman item.",
            "(ii) Another roman item.",
            "(A) An uppercase item.",
            "(2) A second numbered item.",
            "(b) <I>Second.</I> Text.",
        ]
    )
    assert [p.label for p in section.paragraphs] == ["(a)", "(b)"]
    assert section.paragraphs[0].text.count("\n") == 4


def test_text_before_the_first_letter_becomes_an_intro_paragraph():
    section = parse_section(["The following are exempt:", "(a) <I>First.</I> Text."])
    assert [(p.id, p.label) for p in section.paragraphs] == [
        ("1026.9 intro", "intro"),
        ("1026.9(a)", "(a)"),
    ]


def test_section_without_letters_is_one_paragraph():
    section = parse_section(["Plain line one.", "Plain line two."])
    assert [(p.id, p.label) for p in section.paragraphs] == [("1026.9", "")]


def test_roman_i_under_h_is_not_a_new_paragraph():
    section = parse_section(
        [
            *letters_up_to("g"),
            "(h) <I>Rule h.</I> (1) Items:",
            "(i) roman one",
            "(ii) roman two",
            "(i) <I>Rule i.</I> Text.",
            "(j) <I>Rule j.</I> Text.",
        ]
    )
    assert [p.label for p in section.paragraphs] == [f"({c})" for c in "abcdefghij"]
    assert "roman one" in section.paragraphs[7].text
    assert section.paragraphs[8].heading == "Rule i"


def test_letter_i_followed_by_a_numbered_item_is_top_level():
    # The shape that appears in § 1026.38: "(i)", then "(1) ... (i) ..." on one line, then "(ii)".
    section = parse_section(
        [
            *letters_up_to("h"),
            "(i) <I>Rule i.</I> Text.",
            "(1) <I>Total.</I> (i) first roman",
            "(ii) second roman",
            "(j) <I>Rule j.</I> Text.",
        ]
    )
    assert [p.label for p in section.paragraphs] == [f"({c})" for c in "abcdefghij"]


def test_save_and_load_round_trip(corpus, tmp_path):
    path = tmp_path / "corpus.json"
    corpus.save(path)
    loaded = Corpus.load(path)
    assert loaded.sections == corpus.sections
    assert loaded.paragraph("1026.2(b)") == corpus.paragraph("1026.2(b)")
