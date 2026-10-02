"""Regression tests for Wikipedia title extraction from URLs."""

from pathlib import Path
from runpy import run_path

RETRIEVAL = run_path(str(Path(__file__).parents[1] / "eval" / "lib" / "retrieval.py"))
extract = RETRIEVAL["WikipediaAPIRetriever"]._extract_wiki_title


def test_plain_url_title_unchanged():
    assert extract(None, "https://en.wikipedia.org/wiki/Albert_Einstein") == (
        "Albert Einstein"
    )


def test_query_string_is_stripped():
    assert extract(
        None, "https://en.wikipedia.org/wiki/Albert_Einstein?wprov=rarw1"
    ) == ("Albert Einstein")


def test_fragment_is_stripped():
    assert extract(
        None, "https://en.wikipedia.org/wiki/Albert_Einstein#Early_life"
    ) == ("Albert Einstein")


def test_percent_encoded_title_is_decoded():
    assert extract(
        None, "https://en.wikipedia.org/wiki/Python_%28programming_language%29"
    ) == ("Python (programming language)")


def test_non_wiki_url_returns_none():
    assert extract(None, "https://example.com/wiki/Albert_Einstein") is None
