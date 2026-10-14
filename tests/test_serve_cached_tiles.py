"""Coordinate GET serves existing files only, without invoking the renderer."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pixelrag_serve import api


class CachedOnly:
    def __init__(self, cache_dir, path):
        self.cache_dir = cache_dir
        self.path = path

    def cached_chunk_path(self, article_id, tile_index, chunk_index):
        return self.path if (article_id, tile_index, chunk_index) == (7, 0, 0) else None

    def chunk_path(self, *args):
        raise AssertionError("GET must never render")


def png(path, color="red", height=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, height), color).save(path)
    return path


@pytest.fixture
def cached_tile(tmp_path, monkeypatch):
    root = tmp_path / "custom-cache"
    path = png(root / "arbitrary-name.png")
    od = CachedOnly(root, path)
    monkeypatch.setattr(
        api, "_state", {"tiles_dir": str(tmp_path / "store"), "ondemand": od}
    )
    with TestClient(api.app) as client:
        yield client, od


def test_cached_tile_returns_raw_png(cached_tile):
    client, od = cached_tile
    response = client.get("/tile/7/0/0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == od.path.read_bytes()
    with Image.open(io.BytesIO(response.content)) as image:
        image.load()
        assert image.size == (4, 8)  # raw padding is retained; no indexed height lookup


@pytest.mark.parametrize("layout", ["flat", "sharded", "pdf"])
def test_materialized_tile_wins(cached_tile, layout, monkeypatch):
    client, od = cached_tile
    root = api._state["tiles_dir"]
    directory = Path(root)
    if layout == "sharded":
        directory = directory / "shard_000" / "sub"
    name = "tile_0000.jpg" if layout == "pdf" else "chunk_0000_00.png"
    path = png(directory / "7.png.tiles" / name, "blue")

    def unexpected_cache_lookup(*args):
        raise AssertionError("Materialized files must bypass the on-demand cache")

    monkeypatch.setattr(od, "cached_chunk_path", unexpected_cache_lookup)
    response = client.get("/tile/7/0/0")
    assert response.status_code == 200
    assert response.content == path.read_bytes()
    assert response.headers["content-type"] == (
        "image/jpeg" if layout == "pdf" else "image/png"
    )


@pytest.mark.parametrize(
    "coordinates,status",
    [
        ("7/0/1", 404),
        ("-1/0/0", 404),
        ("7/-1/0", 404),
        ("7/0/-1", 404),
        ("bad/0/0", 422),
    ],
)
def test_missing_or_invalid_coordinates_never_render(cached_tile, coordinates, status):
    client, _ = cached_tile
    assert client.get(f"/tile/{coordinates}").status_code == status


@pytest.mark.parametrize("materialized", [False, True])
def test_without_ondemand(cached_tile, materialized):
    client, _ = cached_tile
    api._state.pop("ondemand")
    if materialized:
        png(Path(api._state["tiles_dir"]) / "7.png.tiles" / "chunk_0000_00.png")
    assert client.get("/tile/7/0/0").status_code == (200 if materialized else 404)


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "outside", "symlink", "symlink-loop"]
)
def test_invalid_cached_file(cached_tile, tmp_path, kind):
    client, od = cached_tile
    od.path.unlink()
    if kind == "symlink-loop":
        od.path.symlink_to(od.path)
    elif kind == "directory":
        od.path.mkdir()
    elif kind in ("outside", "symlink"):
        outside = png(tmp_path / "custom-cache-sibling" / "outside.png")
        if kind == "outside":
            od.path = outside
        else:
            od.path.symlink_to(outside)
    assert client.get("/tile/7/0/0").status_code == 404


def test_disappearance_after_resolution_keeps_file_response_contract(
    cached_tile, monkeypatch
):
    client, od = cached_tile
    original = api.FileResponse

    def disappearing_response(path, **kwargs):
        response = original(path, **kwargs)
        od.path.unlink()
        return response

    monkeypatch.setattr(api, "FileResponse", disappearing_response)
    # As for materialized tiles, deletion after the handler's isfile check is
    # reported by Starlette when FileResponse stats the file, before streaming.
    with pytest.raises(RuntimeError, match="does not exist"):
        client.get("/tile/7/0/0")


def test_real_cached_chunk_lookup(cached_tile, monkeypatch):
    from pixelrag_serve.render_ondemand import OnDemandTiles

    client, fake = cached_tile
    ondemand = OnDemandTiles("http://unused.invalid", "book", str(fake.cache_dir))
    path = png(fake.cache_dir / "7.png.tiles" / "chunk_0000_00.png")

    def unexpected_render(*args):
        raise AssertionError("GET must never render")

    monkeypatch.setattr(ondemand, "chunk_path", unexpected_render)
    api._state["ondemand"] = ondemand
    response = client.get("/tile/7/0/0")
    assert response.status_code == 200
    assert response.content == path.read_bytes()
    path.unlink()
    assert client.get("/tile/7/0/0").status_code == 404
