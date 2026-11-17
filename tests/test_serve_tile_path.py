"""Legacy GET /tile?path= serves files under tiles_dir only."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pixelrag_serve import api


def png(path, color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 8), color).save(path)
    return path


@pytest.fixture
def tiles(tmp_path, monkeypatch):
    root = tmp_path / "tiles"
    png(root / "7.png.tiles" / "chunk_0000_00.png")
    monkeypatch.setattr(api, "_state", {"tiles_dir": str(root)})
    with TestClient(api.app) as client:
        yield client, root


def test_file_under_tiles_dir_is_served(tiles):
    client, root = tiles
    response = client.get(
        "/tile", params={"path": str(root / "7.png.tiles" / "chunk_0000_00.png")}
    )
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.content)).size == (4, 8)


@pytest.mark.parametrize("sibling", ["tiles_private", "tiles-old", "tiles.bak"])
def test_sibling_dir_sharing_the_prefix_is_rejected(tiles, sibling):
    client, root = tiles
    secret = png(root.parent / sibling / "secret.png", color="blue")
    response = client.get("/tile", params={"path": str(secret)})
    assert response.status_code == 403


@pytest.mark.parametrize(
    "escape", ["{root}/../outside.png", "{parent}/outside.png", "/etc/hostname"]
)
def test_paths_outside_tiles_dir_are_rejected(tiles, escape):
    client, root = tiles
    png(root.parent / "outside.png")
    path = escape.format(root=root, parent=root.parent)
    assert client.get("/tile", params={"path": path}).status_code == 403


def test_symlink_escaping_tiles_dir_is_rejected(tiles):
    client, root = tiles
    outside = png(root.parent / "outside.png")
    link = root / "link.png"
    link.symlink_to(outside)
    assert client.get("/tile", params={"path": str(link)}).status_code == 403


def test_tiles_dir_itself_is_not_a_file(tiles):
    client, root = tiles
    assert client.get("/tile", params={"path": str(root)}).status_code == 404


def test_missing_file_under_tiles_dir_is_404(tiles):
    client, root = tiles
    path = Path(root) / "7.png.tiles" / "chunk_0099_00.png"
    assert client.get("/tile", params={"path": str(path)}).status_code == 404
