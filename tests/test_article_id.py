"""Tests for the article_id manifest contract.

The pipeline writes article_id into tiles.json. chunk.py propagates it into
chunks.json. Both embedders (GPU scan_shard_chunks, CPU scan_chunks) read it
from the manifest, falling back to the directory name only for legacy indexes.
"""

import json
from pathlib import Path

from PIL import Image

from pixelrag_embed.chunk import chunk_article
from pixelrag_embed.embed import scan_shard_chunks
from pixelrag_embed.embed_cpu import scan_chunks


def _make_tile_dir(base: Path, dir_name: str, article_id: int | None = None) -> Path:
    """A tile dir with only tiles.json (as it exists right after rendering)."""
    td = base / f"{dir_name}.png.tiles"
    td.mkdir(parents=True)
    Image.new("RGB", (875, 500)).save(td / "tile_0000.png")
    meta = {"tiles": ["tile_0000.png"], "tile_height": 8192, "complete": True}
    if article_id is not None:
        meta["article_id"] = article_id
    (td / "tiles.json").write_text(json.dumps(meta))
    return td


def _read_chunks_article_id(td: Path):
    return json.loads((td / "chunks.json").read_text()).get("article_id")


# --- chunk.py propagates article_id from tiles.json into chunks.json ----------


def test_chunk_propagates_article_id_to_chunks_json(tmp_path):
    td = _make_tile_dir(tmp_path, "report", article_id=0)
    chunk_article(str(td))
    # This is the real data flow the GPU embedder depends on.
    assert _read_chunks_article_id(td) == 0


def test_chunk_without_article_id_omits_it(tmp_path):
    td = _make_tile_dir(tmp_path, "5", article_id=None)
    chunk_article(str(td))
    assert "article_id" not in json.loads((td / "chunks.json").read_text())


# --- GPU embedder (scan_shard_chunks) reads it end-to-end --------------------


def test_gpu_scan_reads_propagated_article_id(tmp_path):
    # Non-numeric dir name: only the manifest can supply the right id.
    td = _make_tile_dir(tmp_path, "report", article_id=3)
    chunk_article(str(td))
    chunks = scan_shard_chunks(str(tmp_path))
    assert chunks and all(c.article_id == 3 for c in chunks)


def test_gpu_scan_falls_back_to_tiles_json(tmp_path):
    # chunks.json lacks article_id (legacy chunker) but tiles.json has it.
    td = _make_tile_dir(tmp_path, "report", article_id=4)
    chunk_article(str(td))
    chunks_json = td / "chunks.json"
    meta = json.loads(chunks_json.read_text())
    meta.pop("article_id")
    chunks_json.write_text(json.dumps(meta))
    chunks = scan_shard_chunks(str(tmp_path))
    assert chunks and all(c.article_id == 4 for c in chunks)


def test_gpu_scan_falls_back_to_numeric_dir_name(tmp_path):
    td = _make_tile_dir(tmp_path, "42", article_id=None)
    chunk_article(str(td))
    chunks = scan_shard_chunks(str(tmp_path))
    assert chunks and all(c.article_id == 42 for c in chunks)


# --- CPU embedder (scan_chunks) ---------------------------------------------


def test_cpu_scan_reads_article_id_from_manifest(tmp_path):
    td = _make_tile_dir(tmp_path, "report", article_id=7)
    chunk_article(str(td))
    items = scan_chunks(str(tmp_path))
    assert items and all(it["article_id"] == 7 for it in items)


def test_cpu_non_numeric_fallback_is_reproducible(tmp_path):
    # No manifest id, non-numeric dir → must be a *stable* hash, not the salted
    # builtin hash() (which changes per process via PYTHONHASHSEED and would make
    # the index non-reproducible). Assert the exact sha1-derived value so a
    # regression back to builtin hash() fails here.
    import hashlib

    td = _make_tile_dir(tmp_path, "my_report", article_id=None)
    chunk_article(str(td))
    got = scan_chunks(str(tmp_path))[0]["article_id"]
    expected = int(hashlib.sha1(b"my_report").hexdigest()[:8], 16)
    assert got == expected
