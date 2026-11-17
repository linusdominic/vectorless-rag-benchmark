"""The chunker and the chunk-based baselines. The dense index is covered by the benchmark."""

from vrbench.chunking import Chunk, chunk_corpus, chunk_paragraph
from vrbench.retrievers.chunked import (
    BM25Index,
    ChunkRetriever,
    build_chunk_retrievers,
    reciprocal_rank_fusion,
    tokenize,
)


def test_every_chunk_names_its_section_and_paragraph(corpus):
    chunks = chunk_corpus(corpus)
    assert len(chunks) == 6
    assert chunks[0].paragraph_id == "1026.1(a)"
    assert chunks[0].text.startswith("§ 1026.1 Refunds | (a) Refund deadline\n(a) Refund deadline.")


def test_long_paragraph_is_split_within_the_limit(corpus):
    section = corpus.section("1026.1")
    long_text = "\n".join(f"({i}) " + "word " * 60 for i in range(1, 9))
    paragraph = type(section.paragraphs[0])("1026.1(a)", "1026.1", "(a)", "Long", long_text)

    chunks = chunk_paragraph(section, paragraph, max_chars=700)

    assert len(chunks) > 1
    assert all(chunk.paragraph_id == "1026.1(a)" for chunk in chunks)
    assert all(len(chunk.text.split("\n", 1)[1]) <= 700 for chunk in chunks)
    assert all(chunk.text.startswith("§ 1026.1 Refunds | (a) Long") for chunk in chunks)


def test_tokenizer_keeps_section_numbers_whole():
    assert tokenize("See § 1026.43(c), the Ability-to-Repay rule.") == [
        "see", "1026.43", "c", "the", "ability", "to", "repay", "rule",
    ]  # fmt: skip


def test_bm25_ranks_the_matching_chunk_first(corpus):
    chunks = chunk_corpus(corpus)
    index = BM25Index(chunks)

    assert chunks[index.search("escrow account for property taxes", 3)[0]].paragraph_id == (
        "1026.35(a)"
    )
    assert chunks[index.search("late payment penalty", 3)[0]].paragraph_id == "1026.2(a)"
    assert index.search("zebra xylophone", 3) == []


def test_bm25_prefers_rare_terms_over_common_ones():
    chunks = [
        Chunk("a", "creditor creditor creditor disclosure"),
        Chunk("b", "creditor rescission"),
        Chunk("c", "creditor disclosure"),
    ]
    assert BM25Index(chunks).search("creditor rescission", 3)[0] == 1


def test_reciprocal_rank_fusion_rewards_agreement():
    # 7 is second in both lists, which beats being first in only one.
    assert reciprocal_rank_fusion([[1, 7, 3], [9, 7, 4]], n=3)[0] == 7
    assert reciprocal_rank_fusion([[1, 2], [1, 2]], n=5) == [1, 2]


async def test_chunk_retriever_ranks_paragraphs_without_duplicates():
    chunks = [Chunk("p1", "one"), Chunk("p1", "two"), Chunk("p2", "three"), Chunk("p3", "four")]
    retriever = ChunkRetriever("fake", chunks, lambda query, n: [1, 0, 3, 2], top_chunks=2)

    retrieval = await retriever.retrieve("anything")

    assert retrieval.paragraph_ids == ["p1", "p3", "p2"]
    assert [p.text for p in retrieval.passages] == ["two", "one"]
    assert retrieval.llm_calls == 0 and retrieval.cost_usd == 0.0


async def test_bm25_retriever_end_to_end(corpus):
    (bm25,) = build_chunk_retrievers(chunk_corpus(corpus), with_dense=False)
    retrieval = await bm25.retrieve("How long must an escrow account be kept?")
    assert retrieval.paragraph_ids[0] == "1026.35(a)"


async def test_parent_expansion_sends_whole_paragraphs(corpus):
    chunks = chunk_corpus(corpus)
    order = [4, 0, 5]  # 1026.35(a), 1026.1(a), 1026.35(b)
    retriever = ChunkRetriever("p", chunks, lambda query, n: order, parents=corpus)

    retrieval = await retriever.retrieve("anything")

    assert retrieval.paragraph_ids == ["1026.35(a)", "1026.1(a)", "1026.35(b)"]
    assert retrieval.passages[0].text == (
        "§ 1026.35 Escrow accounts (a)\n" + corpus.paragraph("1026.35(a)").text
    )
    assert len(retrieval.passages) == 3
