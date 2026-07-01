"""Tests for hybrid search: BM25 index + Reciprocal Rank Fusion."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "serve" / "src"))
from pixelrag_serve.hybrid import BM25Index, reciprocal_rank_fusion, load_or_build_bm25


@pytest.fixture
def bm25():
    idx = BM25Index()
    idx.build_from_texts(
        {
            0: "Python programming language created by Guido van Rossum",
            1: "Machine learning neural networks deep learning",
            2: "Albert Einstein theory of relativity physics Nobel Prize",
            3: "JavaScript React frontend web development",
            4: "Python snake reptile biology animal",
        }
    )
    return idx


def test_bm25_basic_search(bm25):
    """BM25 should rank relevant docs highest."""
    results = bm25.search("Python programming", k=5)
    # Doc 0 (Python programming) should rank first
    assert results[0][0] == 0
    assert results[0][1] > 0


def test_bm25_disambiguates(bm25):
    """'Python snake' should prefer the reptile doc over the programming one."""
    results = bm25.search("python snake reptile", k=5)
    vids = [vid for vid, _ in results]
    # Doc 4 (snake) should rank above doc 0 (programming)
    assert vids.index(4) < vids.index(0)


def test_bm25_empty_query(bm25):
    """Empty or punctuation-only queries should return empty."""
    assert bm25.search("") == []
    assert bm25.search("!!! ???") == []


def test_bm25_unknown_terms(bm25):
    """Query with no matching terms returns empty."""
    assert bm25.search("xyzzy frobnicator") == []


def test_bm25_save_load(bm25, tmp_path):
    """Save and reload should produce identical search results."""
    path = str(tmp_path / "bm25.json")
    bm25.save(path)

    loaded = BM25Index()
    assert loaded.load(path)
    assert loaded.doc_count == bm25.doc_count

    # Same results
    r1 = bm25.search("Einstein physics")
    r2 = loaded.search("Einstein physics")
    assert [vid for vid, _ in r1] == [vid for vid, _ in r2]


def test_rrf_fuses_two_lists():
    """RRF should combine rankings from two sources."""
    # Visual search ranks: doc 5, doc 3, doc 1
    visual = [(5, 0.9), (3, 0.7), (1, 0.5)]
    # Text search ranks: doc 1, doc 5, doc 7
    text = [(1, 3.2), (5, 2.1), (7, 1.0)]

    fused = reciprocal_rank_fusion([visual, text])
    fused_vids = [vid for vid, _ in fused]

    # Doc 5 appears at rank 0 in visual and rank 1 in text — strong signal
    # Doc 1 appears at rank 2 in visual and rank 0 in text — also strong
    assert 5 in fused_vids[:2]
    assert 1 in fused_vids[:2]
    # Doc 7 only in text — should rank lower
    assert fused_vids.index(7) > fused_vids.index(5)


def test_rrf_single_list():
    """RRF with one list should preserve original order."""
    ranked = [(10, 0.9), (20, 0.8), (30, 0.7)]
    fused = reciprocal_rank_fusion([ranked])
    assert [vid for vid, _ in fused] == [10, 20, 30]


def test_load_or_build_from_articles(tmp_path):
    """Integration: build BM25 from articles.json and search."""
    articles = [
        {
            "title": "Python (programming language)",
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        },
        {
            "title": "Albert Einstein",
            "url": "https://en.wikipedia.org/wiki/Albert_Einstein",
        },
        {
            "title": "Machine learning",
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
        },
    ]
    articles_path = tmp_path / "articles.json"
    articles_path.write_text(json.dumps(articles))
    index_dir = str(tmp_path)

    bm25 = load_or_build_bm25(index_dir, str(articles_path))
    assert bm25.loaded
    assert bm25.doc_count == 3

    # Should find Einstein
    results = bm25.search("Einstein")
    assert results[0][0] == 1  # article index 1

    # Cache file should exist
    assert (tmp_path / "bm25_index.json").exists()

    # Reload from cache
    bm25_cached = load_or_build_bm25(index_dir, str(articles_path))
    assert bm25_cached.loaded
