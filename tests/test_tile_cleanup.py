"""Source tiles may be removed only after checking the replacement inventory."""

import json
import logging

import pytest
from PIL import Image
from pixelrag_embed.chunk import _delete_tiles_in_shard, process_shard


@pytest.fixture
def article(tmp_path):
    path = tmp_path / "shard_00000" / "0.png.tiles"
    path.mkdir(parents=True)
    Image.new("RGB", (32, 32), "red").save(path / "tile_0000.png")
    (path / "tiles.json").write_text(json.dumps({"tiles": ["tile_0000.png"]}))
    return path


def write_chunks(article, chunks):
    (article / "chunks.json").write_text(json.dumps({"chunks": chunks}))


def replacement(article, tile="tile_0000.png", filename="chunk_0000_00.png"):
    Image.new("RGB", (32, 32), "red").save(article / filename)
    return {"tile": tile, "file": filename}


def test_process_shard_preserves_sources_when_chunking_skips_corrupt_manifests(
    article, tmp_path, caplog
):
    source = (article / "tile_0000.png").read_bytes()
    (article / "tiles.json").write_text("{")
    (article / "chunks.json").write_text("{")
    with caplog.at_level(logging.WARNING):
        result = process_shard(str(tmp_path), delete_tiles=True, progress=False)
    assert result["chunked"] == 0
    assert result["tiles_deleted"] == 0
    assert (article / "tile_0000.png").read_bytes() == source
    assert "Retaining source tiles" in caplog.text


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        "{",
        "",
        "null",
        "[]",
        "{}",
        '{"chunks": []}',
        '{"chunks": null}',
        '{"chunks": {}}',
        '{"chunks": [null]}',
        '{"chunks": [{}]}',
    ],
)
def test_invalid_inventory_preserves_sources(article, tmp_path, caplog, manifest):
    if manifest is not None:
        (article / "chunks.json").write_text(manifest)
    with caplog.at_level(logging.WARNING):
        assert _delete_tiles_in_shard(str(tmp_path)) == 0
    assert (article / "tile_0000.png").is_file()
    assert "Retaining source tiles" in caplog.text


@pytest.mark.parametrize("state", ["missing", "directory", "empty", "symlink"])
def test_unusable_replacement_preserves_sources(article, tmp_path, caplog, state):
    target = article / "chunk_0000_00.png"
    if state == "directory":
        target.mkdir()
    elif state == "empty":
        target.touch()
    elif state == "symlink":
        target.symlink_to("tile_0000.png")
    write_chunks(article, [{"tile": "tile_0000.png", "file": target.name}])
    with caplog.at_level(logging.WARNING):
        assert _delete_tiles_in_shard(str(tmp_path)) == 0
    assert (article / "tile_0000.png").is_file()
    assert "Retaining source tiles" in caplog.text


def test_missing_one_of_multiple_chunks_preserves_source(article, tmp_path):
    chunks = [
        replacement(article),
        {"tile": "tile_0000.png", "file": "chunk_0000_01.png"},
    ]
    write_chunks(article, chunks)
    assert _delete_tiles_in_shard(str(tmp_path)) == 0
    assert (article / "tile_0000.png").is_file()


def test_process_shard_requires_coverage_of_every_source_tile(
    article, tmp_path, caplog
):
    Image.new("RGB", (32, 32), "blue").save(article / "tile_0001.png")
    (article / "tiles.json").write_text(
        json.dumps({"tiles": ["tile_0000.png", "tile_0001.png"]})
    )
    write_chunks(article, [replacement(article)])
    with caplog.at_level(logging.WARNING):
        result = process_shard(str(tmp_path), delete_tiles=True, progress=False)
    assert result["chunked"] == 0  # Existing chunk causes the chunker to skip.
    assert result["tiles_deleted"] == 0
    assert (article / "tile_0000.png").is_file()
    assert (article / "tile_0001.png").is_file()
    assert "Retaining source tiles" in caplog.text


@pytest.mark.parametrize("dry_run", [False, True])
def test_process_shard_success_and_dry_run(article, tmp_path, dry_run):
    before = (article / "tile_0000.png").read_bytes()
    result = process_shard(
        str(tmp_path), delete_tiles=True, dry_run=dry_run, progress=False
    )
    assert result["chunked"] == 1
    assert result["tiles_deleted"] == (0 if dry_run else 1)
    if dry_run:
        assert (article / "tile_0000.png").read_bytes() == before
        assert not (article / "chunks.json").exists()
        assert not (article / "chunk_0000_00.png").exists()
    else:
        assert not (article / "tile_0000.png").exists()
        assert (article / "chunk_0000_00.png").read_bytes() == before
        assert (
            process_shard(str(tmp_path), delete_tiles=True, progress=False)[
                "tiles_deleted"
            ]
            == 0
        )


def test_pdf_self_references_are_preserved(article, tmp_path):
    (article / "tile_0000.png").unlink()
    names = ["tile_0002.jpg", "tile_0004.jpg"]
    chunks = []
    for index, name in zip([2, 4], names):
        Image.new("RGB", (32, 32), "red").save(article / name)
        chunks.append(
            {
                "tile": name,
                "tile_index": index,
                "chunk_index": 0,
                "file": name,
                "y_offset": 0,
                "height": 32,
                "width": 32,
            }
        )
    (article / "tiles.json").write_text(
        json.dumps({"tiles": names, "complete": False, "requested_pages": [3, 5]})
    )
    write_chunks(article, chunks)
    result = process_shard(str(tmp_path), delete_tiles=True, progress=False)
    assert result["chunked"] == 0
    assert result["tiles_deleted"] == 0
    assert all((article / name).is_file() for name in names)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_tile",
        "bad_file_type",
        "outside_file",
        "duplicate_file",
        "count_mismatch",
    ],
)
def test_ambiguous_inventory_preserves_sources(article, tmp_path, problem):
    chunk = replacement(article)
    manifest = {"chunks": [chunk]}
    if problem == "missing_tile":
        del chunk["tile"]
    elif problem == "bad_file_type":
        chunk["file"] = [chunk["file"]]
    elif problem == "outside_file":
        outside = tmp_path / "outside.png"
        Image.new("RGB", (32, 32)).save(outside)
        chunk["file"] = str(outside)
    elif problem == "duplicate_file":
        Image.new("RGB", (32, 32), "blue").save(article / "tile_0001.png")
        manifest["chunks"].append({"tile": "tile_0001.png", "file": chunk["file"]})
    else:
        manifest["num_chunks"] = 2
    (article / "chunks.json").write_text(json.dumps(manifest))
    before = {p.name: p.read_bytes() for p in article.glob("tile_*.png")}
    assert _delete_tiles_in_shard(str(tmp_path)) == 0
    assert {p.name: p.read_bytes() for p in article.glob("tile_*.png")} == before


def test_invalid_article_does_not_block_valid_article_cleanup(article, tmp_path):
    (article / "chunks.json").write_text("{")
    other = article.parent / "1.png.tiles"
    other.mkdir()
    Image.new("RGB", (32, 32)).save(other / "tile_0000.png")
    write_chunks(other, [replacement(other)])
    assert _delete_tiles_in_shard(str(tmp_path)) == 1
    assert (article / "tile_0000.png").is_file()
    assert not (other / "tile_0000.png").exists()
    assert (other / "chunk_0000_00.png").is_file()


@pytest.mark.parametrize("skipped", ["corrupt", "narrow"])
def test_real_chunker_skips_retain_all_sources(article, tmp_path, caplog, skipped):
    omitted = article / "tile_0001.png"
    if skipped == "corrupt":
        omitted.write_bytes(b"not an image")
    else:
        # Too tall for the copy path and too narrow for one model patch.
        Image.new("RGB", (10, 2000), "blue").save(omitted)
    (article / "tiles.json").write_text(
        json.dumps({"tiles": ["tile_0000.png", "tile_0001.png"]})
    )
    before = {p.name: p.read_bytes() for p in article.glob("tile_*.png")}
    with caplog.at_level(logging.WARNING):
        result = process_shard(str(tmp_path), delete_tiles=True, progress=False)
    assert result["chunked"] == 1
    assert result["tiles_deleted"] == 0
    manifest = json.loads((article / "chunks.json").read_text())
    assert {c["tile"] for c in manifest["chunks"]} == {"tile_0000.png"}
    assert {p.name: p.read_bytes() for p in article.glob("tile_*.png")} == before
    assert "tiles without replacement coverage" in caplog.text
