"""Navigation maps describe materialized PDF pages and web chunks."""

import json
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pixelrag_render import render_pdf
from pixelrag_serve import api


@pytest.fixture
def tiles(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_state", {"tiles_dir": str(tmp_path)})
    api._article_pages.cache_clear()
    yield tmp_path
    api._article_pages.cache_clear()


@pytest.fixture
def pdf_pages(tiles, monkeypatch):
    """Run the real PDF producer with only the external converter stubbed."""
    converter = ModuleType("pdf2image")

    def convert_from_path(**kwargs):
        return [
            Image.new("RGB", (16, 16), "white")
            for _ in range(kwargs.get("first_page", 1), kwargs.get("last_page", 4) + 1)
        ]

    converter.convert_from_path = convert_from_path
    monkeypatch.setitem(sys.modules, "pdf2image", converter)
    source = tiles / "document.pdf"
    source.touch()

    def render(pages):
        return render_pdf(source, tiles, pages=pages, stem="9")[0]

    return render


@pytest.mark.parametrize(
    "pages, expected", [(None, "0:0-0,1:0-0,2:0-0,3:0-0"), ([2, 4], "1:0-0,3:0-0")]
)
def test_pdf_navigation_search_and_get(tiles, pdf_pages, pages, expected):
    article = pdf_pages(pages)
    chunks = json.loads((article / "chunks.json").read_text())["chunks"]
    # PDF pages are the chunk files; no chunk_NNNN_NN files are produced.
    assert all(c["file"] == f"tile_{c['tile_index']:04d}.jpg" for c in chunks)
    first = chunks[0]["tile_index"]
    client = TestClient(api.app)
    assert client.get(f"/tile/9/{first}/0").status_code == 200
    assert api._article_pages(9) == expected

    backend = Mock()
    backend.raw_search.return_value = [
        [
            {
                "score": 1.0,
                "vector_id": 0,
                "article_id": 9,
                "tile_index": first,
                "chunk_index": 0,
                "y_offset": 0,
                "tile_height": 16,
            }
        ]
    ]
    api._state.update(
        backend=backend, dimension=4, articles=[""] * 9 + ["document.pdf"]
    )
    response = client.post("/search", json={"queries": [{"embedding": [1, 0, 0, 0]}]})
    assert response.status_code == 200, response.text
    navigation = response.json()["results"][0]["hits"][0]["article_pages"]
    assert navigation == expected
    for entry in navigation.split(","):
        page, bounds = entry.split(":")
        assert bounds == "0-0"
        image = client.get(f"/tile/9/{page}/0")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert image.content == (article / f"tile_{int(page):04d}.jpg").read_bytes()


@pytest.mark.parametrize("layout", ["", "shard_000/sub"])
def test_web_chunk_map_unchanged(tiles, layout):
    article = tiles / layout / "9.png.tiles"
    article.mkdir(parents=True)
    for page, chunk in [(0, 0), (0, 1), (1, 0)]:
        Image.new("RGB", (16, 16)).save(article / f"chunk_{page:04d}_{chunk:02d}.png")
    assert api._article_pages(9) == "0:0-1,1:0-0"


def test_missing_article_has_no_map(tiles):
    assert api._article_pages(9) is None


def test_materialized_chunks_take_precedence_per_page(tiles, pdf_pages):
    article = pdf_pages([2, 4])
    for chunk in (1, 2):
        Image.new("RGB", (16, 16), "red").save(article / f"chunk_0001_{chunk:02d}.png")
    assert api._article_pages(9) == "1:1-2,3:0-0"
    client = TestClient(api.app)
    for page, chunk, filename in [
        (1, 1, "chunk_0001_01.png"),
        (1, 2, "chunk_0001_02.png"),
        (3, 0, "tile_0003.jpg"),
    ]:
        response = client.get(f"/tile/9/{page}/{chunk}")
        assert response.status_code == 200
        assert response.content == (article / filename).read_bytes()


def test_ondemand_map_remains_unavailable_without_rendering(tiles, pdf_pages):
    pdf_pages(None)
    ondemand = Mock()
    api._state["ondemand"] = ondemand
    assert api._article_pages(9) is None
    assert ondemand.mock_calls == []


def test_only_existing_pdf_page_files_are_advertised(tiles, pdf_pages):
    article = pdf_pages(None)
    (article / "tile_0001.jpg").unlink()
    # Neither stale manifest entries nor directories named like pages qualify.
    (article / "tile_0004.jpg").mkdir()
    assert api._article_pages(9) == "0:0-0,2:0-0,3:0-0"


def test_sharded_pdf_pages_are_not_advertised(tiles):
    # _resolve_path only supports PDF page fallback in the flat layout.
    article = tiles / "shard_000" / "sub" / "9.png.tiles"
    article.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(article / "tile_0000.jpg")
    assert api._article_pages(9) is None
    assert TestClient(api.app).get("/tile/9/0/0").status_code == 404
