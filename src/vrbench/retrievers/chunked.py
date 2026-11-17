"""The three chunk-based baselines: BM25 (keywords), dense (embeddings) and hybrid."""

import hashlib
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from vrbench.chunking import Chunk
from vrbench.corpus import Corpus
from vrbench.retrievers.base import Passage, Retrieval, unique, whole_paragraphs

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")  # keeps section numbers such as 1026.43 whole


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._lengths = [sum(counts.values()) for counts in self._counts]
        self._average = sum(self._lengths) / max(len(chunks), 1)
        document_frequency = Counter(term for counts in self._counts for term in counts)
        n = len(chunks)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        self._postings: dict[str, list[int]] = {}
        for index, counts in enumerate(self._counts):
            for term in counts:
                self._postings.setdefault(term, []).append(index)

    def search(self, query: str, n: int) -> list[int]:
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index in self._postings[term]:
                tf = self._counts[index][term]
                norm = 1 - self.b + self.b * self._lengths[index] / self._average
                scores[index] = scores.get(index, 0.0) + idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * norm
                )
        return sorted(scores, key=lambda index: (-scores[index], index))[:n]


class DenseIndex:
    """Cosine similarity over local embeddings, by brute force. No vector database."""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Path | None = None,
    ) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        texts = [chunk.text for chunk in chunks]
        digest = hashlib.sha256("\x00".join([model_name, *texts]).encode()).hexdigest()[:16]
        cache = cache_dir / f"embeddings-{digest}.npy" if cache_dir else None
        if cache and cache.exists():
            matrix = np.load(cache)
        else:
            matrix = np.array(list(self._model.embed(texts)), dtype=np.float32)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, matrix)
        self._matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def search(self, query: str, n: int) -> list[int]:
        vector = np.asarray(next(iter(self._model.query_embed(query))), dtype=np.float32)
        scores = self._matrix @ (vector / np.linalg.norm(vector))
        return np.argsort(-scores)[:n].tolist()


def reciprocal_rank_fusion(rankings: list[list[int]], n: int, k: int = 60) -> list[int]:
    """Merge rankings by summing 1 / (k + rank). Items ranked well by several lists win."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:n]


class ChunkRetriever:
    """Turns a chunk search function into a retriever scored on paragraphs.

    By default the answering model gets the top chunks. With ``parents`` set it gets the
    whole paragraphs those chunks came from, the same context the vectorless method sends.
    """

    def __init__(
        self,
        name: str,
        chunks: list[Chunk],
        search: Callable[[str, int], list[int]],
        top_chunks: int = 5,
        candidates: int = 50,
        parents: Corpus | None = None,
    ) -> None:
        self.name = name
        self._chunks = chunks
        self._search = search
        self._top_chunks = top_chunks
        self._candidates = candidates
        self._parents = parents

    async def retrieve(self, question: str) -> Retrieval:
        started = time.perf_counter()
        ranked = [self._chunks[i] for i in self._search(question, self._candidates)]
        latency = time.perf_counter() - started
        paragraph_ids = unique([chunk.paragraph_id for chunk in ranked])
        if self._parents is not None:
            passages = whole_paragraphs(self._parents, paragraph_ids)
        else:
            passages = [Passage(c.paragraph_id, c.text) for c in ranked[: self._top_chunks]]
        return Retrieval(paragraph_ids=paragraph_ids, passages=passages, latency_s=latency)


def build_chunk_retrievers(
    chunks: list[Chunk],
    cache_dir: Path | None = None,
    with_dense: bool = True,
    corpus: Corpus | None = None,
) -> list[ChunkRetriever]:
    bm25 = BM25Index(chunks)
    retrievers = [ChunkRetriever("bm25", chunks, bm25.search)]
    if with_dense:
        dense = DenseIndex(chunks, cache_dir=cache_dir)

        def hybrid(query: str, n: int) -> list[int]:
            return reciprocal_rank_fusion([bm25.search(query, n), dense.search(query, n)], n)

        retrievers += [
            ChunkRetriever("dense", chunks, dense.search),
            ChunkRetriever("hybrid", chunks, hybrid),
        ]
        if corpus is not None:
            retrievers.append(ChunkRetriever("hybrid+parents", chunks, hybrid, parents=corpus))
    return retrievers
