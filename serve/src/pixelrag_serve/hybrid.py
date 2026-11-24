"""Hybrid search: BM25 text index + Reciprocal Rank Fusion with FAISS visual results.

Builds an inverted text index from article text (extracted during indexing or from
OCR/page content). At query time, fuses BM25 text results with FAISS visual results
using Reciprocal Rank Fusion (RRF), improving precision on text-heavy queries while
retaining visual retrieval for tables/charts/diagrams.

Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods"
"""

import json
import logging
import math
import os
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

# BM25 parameters (tuned for short document chunks)
_K1 = 1.2
_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """In-memory BM25 inverted index over article/chunk text.

    Each document is identified by its vector_id (matching FAISS metadata).
    """

    def __init__(self):
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lens: dict[int, int] = {}  # vector_id → doc length
        self.df: dict[str, int] = defaultdict(int)  # term → doc frequency
        self.tf: dict[int, dict[str, int]] = {}  # vector_id → {term: freq}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def build_from_texts(self, texts: dict[int, str]):
        """Build index from {vector_id: text_content} mapping."""
        self.doc_count = len(texts)
        total_len = 0

        for vid, text in texts.items():
            tokens = _tokenize(text)
            self.doc_lens[vid] = len(tokens)
            total_len += len(tokens)

            term_freqs: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1
            self.tf[vid] = dict(term_freqs)

            for term in term_freqs:
                self.df[term] += 1

        self.avg_dl = total_len / max(self.doc_count, 1)
        self._loaded = True
        logger.info("BM25 index built: %d docs, %d terms", self.doc_count, len(self.df))

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        """Return top-k (vector_id, score) pairs ranked by BM25 score."""
        if not self._loaded:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores: dict[int, float] = defaultdict(float)

        for term in tokens:
            if term not in self.df:
                continue
            idf = math.log(
                (self.doc_count - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0
            )
            for vid, term_freqs in self.tf.items():
                if term not in term_freqs:
                    continue
                tf = term_freqs[term]
                dl = self.doc_lens[vid]
                numerator = tf * (_K1 + 1)
                denominator = tf + _K1 * (1 - _B + _B * dl / self.avg_dl)
                scores[vid] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:k]

    def save(self, path: str):
        """Persist to JSON for fast reload."""
        data = {
            "doc_count": self.doc_count,
            "avg_dl": self.avg_dl,
            "doc_lens": self.doc_lens,
            "df": dict(self.df),
            "tf": {str(k): v for k, v in self.tf.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("BM25 index saved: %s (%.1f MB)", path, os.path.getsize(path) / 1e6)

    def load(self, path: str) -> bool:
        """Load from JSON. Returns True on success."""
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.doc_count = data["doc_count"]
        self.avg_dl = data["avg_dl"]
        self.doc_lens = {int(k): v for k, v in data["doc_lens"].items()}
        self.df = defaultdict(int, data["df"])
        self.tf = {int(k): v for k, v in data["tf"].items()}
        self._loaded = True
        logger.info(
            "BM25 index loaded: %d docs, %d terms", self.doc_count, len(self.df)
        )
        return True


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked result lists using RRF.

    Args:
        ranked_lists: List of ranked results, each is [(vector_id, score), ...]
        k: RRF constant (default 60, from the original paper)

    Returns:
        Fused ranked list of (vector_id, rrf_score) sorted descending.
    """
    fused_scores: dict[int, float] = defaultdict(float)

    for ranked in ranked_lists:
        for rank, (vid, _score) in enumerate(ranked):
            fused_scores[vid] += 1.0 / (k + rank + 1)

    return sorted(fused_scores.items(), key=lambda x: -x[1])


def build_text_index_from_articles(
    articles_json: str,
    metadata_path: str | None = None,
) -> BM25Index:
    """Build BM25 index from articles.json titles/URLs.

    For a richer index, pass article page text via metadata. This minimal
    version uses article titles as the text representation — still useful
    for entity/name queries like "Albert Einstein" or "Python programming".
    """
    index = BM25Index()
    texts: dict[int, str] = {}

    with open(articles_json) as f:
        articles = json.load(f)

    for vid, article in enumerate(articles):
        # Use title + URL path as searchable text
        title = article.get("title", "")
        url = article.get("url", "")
        # Extract meaningful text from URL path
        url_text = (
            url.split("/")[-1].replace("_", " ").replace("%20", " ") if url else ""
        )
        text = f"{title} {url_text}"
        if text.strip():
            texts[vid] = text

    index.build_from_texts(texts)
    return index


def load_or_build_bm25(index_dir: str, articles_json: str) -> BM25Index:
    """Load cached BM25 index or build from articles.json."""
    bm25_path = os.path.join(index_dir, "bm25_index.json")
    index = BM25Index()

    if index.load(bm25_path):
        return index

    # Build and cache
    index = build_text_index_from_articles(articles_json)
    if index.loaded:
        index.save(bm25_path)
    return index
