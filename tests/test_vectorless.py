from tests.conftest import FakeLLM
from vrbench.retrievers.vectorless import (
    VectorlessRetriever,
    section_outline,
    table_of_contents,
)


def navigator(sections, paragraphs):
    """A model that picks the given sections, then the given paragraphs."""

    def handler(system, prompt):
        if "table of contents" in system:
            return {"reasoning": "escrow is a mortgage topic", "sections": sections}
        return {"reasoning": "the requirement paragraph", "paragraphs": paragraphs}

    return FakeLLM(handler)


def test_table_of_contents_groups_sections_by_subpart(corpus):
    assert table_of_contents(corpus) == (
        "Subpart A: General\n"
        "  1026.1 Refunds\n"
        "  1026.2 Late fees\n"
        "\n"
        "Subpart E: Home Mortgages\n"
        "  1026.35 Escrow accounts"
    )


def test_outline_uses_summaries_when_given_and_opening_words_otherwise(corpus):
    section = corpus.section("1026.35")
    plain = section_outline(section)
    assert "  1026.35(a) [18 words] Requirement. An escrow account" in plain

    summarized = section_outline(section, {"1026.35(a)": "escrow duration; five-year minimum"})
    assert "  1026.35(a) [18 words] Requirement. escrow duration; five-year minimum" in summarized
    assert "  1026.35(b) [15 words] Exemptions. Loans secured" in summarized


async def test_two_step_navigation_returns_ranked_paragraphs(corpus):
    llm = navigator(["1026.35", "1026.1"], ["1026.35(a)", "1026.1(b)"])
    retriever = VectorlessRetriever(corpus, llm, "claude-haiku-5-5")

    retrieval = await retriever.retrieve("How long must escrow be kept?")

    assert retrieval.paragraph_ids == ["1026.35(a)", "1026.1(b)"]
    assert retrieval.passages[0].text.startswith("§ 1026.35 Escrow accounts (a)\n(a) Requirement.")
    assert (retrieval.llm_calls, retrieval.input_tokens, retrieval.output_tokens) == (2, 200, 40)
    assert retrieval.cost_usd == 0.002 and retrieval.latency_s == 1.0
    assert retrieval.trace["sections"] == ["1026.35", "1026.1"]

    # Step 1 saw the whole table of contents; step 2 saw only the chosen sections.
    assert "1026.2 Late fees" in llm.calls[0]["system"]
    assert "§ 1026.35 Escrow accounts" in llm.calls[1]["prompt"]
    assert "Late fees" not in llm.calls[1]["prompt"]


async def test_made_up_and_out_of_scope_ids_are_dropped(corpus):
    # 1026.99 does not exist; 1026.2(a) exists but its section was not chosen.
    llm = navigator(
        ["1026.99", "§ 1026.35"], ["1026.35(z)", "1026.2(a)", "1026.35(b)", "1026.35(b)"]
    )
    retriever = VectorlessRetriever(corpus, llm, "claude-haiku-5-5")

    retrieval = await retriever.retrieve("question")

    assert retrieval.trace["sections"] == ["1026.35"]
    assert retrieval.paragraph_ids == ["1026.35(b)"]


async def test_no_valid_section_ends_the_search_after_one_call(corpus):
    llm = navigator(["1026.99"], ["1026.35(a)"])
    retrieval = await VectorlessRetriever(corpus, llm, "claude-haiku-5-5").retrieve("question")

    assert retrieval.paragraph_ids == [] and retrieval.passages == []
    assert retrieval.llm_calls == 1


async def test_limits_on_sections_paragraphs_and_context(corpus):
    llm = navigator(
        ["1026.1", "1026.2", "1026.35"],
        ["1026.1(a)", "1026.1(b)", "1026.2(a)", "1026.2(b)"],
    )
    retriever = VectorlessRetriever(
        corpus, llm, "m", max_sections=2, max_paragraphs=3, context_paragraphs=2, context_chars=150
    )

    retrieval = await retriever.retrieve("question")

    assert retrieval.trace["sections"] == ["1026.1", "1026.2"]
    assert retrieval.paragraph_ids == ["1026.1(a)", "1026.1(b)", "1026.2(a)"]
    assert [p.paragraph_id for p in retrieval.passages] == ["1026.1(a)", "1026.1(b)"]
    assert sum(len(p.text) for p in retrieval.passages) == 150


async def test_summaries_reach_both_prompts(corpus):
    llm = navigator(["1026.35"], ["1026.35(a)"])
    summaries = {"1026.35": "escrow rules for mortgages", "1026.35(a)": "five-year minimum"}
    retriever = VectorlessRetriever(corpus, llm, "m", summaries=summaries)

    await retriever.retrieve("question")

    assert "escrow rules for mortgages" in llm.calls[0]["system"]
    assert "five-year minimum" in llm.calls[1]["prompt"]
