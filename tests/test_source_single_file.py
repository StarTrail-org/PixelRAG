"""A source path may point straight at a document, not just a directory.

`pixelrag index build --source ./paper.pdf` is the form the README documents,
but both adapters used to walk the path as a directory — which silently
yielded zero documents and failed several stages later at index build.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "index" / "src"))
from pixelrag_index.sources.local import LocalSource
from pixelrag_index.sources.pdf import PDFSource


@pytest.fixture
def docs_dir(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "notes.md").write_text("# Notes")
    (tmp_path / "ignored.csv").write_text("a,b,c")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.pdf").write_bytes(b"%PDF-1.4\n")
    return tmp_path


def test_pdf_source_accepts_a_single_file(docs_dir):
    source = PDFSource(str(docs_dir / "paper.pdf"))
    assert [d.id for d in source] == ["paper"]
    assert len(source) == 1


def test_local_source_accepts_a_single_file(docs_dir):
    source = LocalSource(str(docs_dir / "notes.md"))
    docs = list(source)
    assert [d.id for d in docs] == ["notes"]
    assert docs[0].metadata["type"] == "text"


def test_local_source_single_file_still_filters_by_extension(docs_dir):
    """An unsupported suffix yields nothing, matching directory behaviour."""
    assert len(LocalSource(str(docs_dir / "ignored.csv"))) == 0


def test_directory_traversal_is_unchanged(docs_dir):
    """The directory path must keep recursing into sub-directories."""
    assert {d.id for d in PDFSource(str(docs_dir))} == {"paper", "deep"}
    assert {d.id for d in LocalSource(str(docs_dir))} == {"paper", "notes", "deep"}
