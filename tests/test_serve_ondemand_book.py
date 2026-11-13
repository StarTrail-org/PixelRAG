"""An on-demand serve must know which kiwix book to render from.

The book id is auto-derived from kiwix-serve's /catalog/v2/entries. That
document is kiwix-serve's own; the bundled zim_server/zim_server_async answer
every path out of the ZIM, so the request 404s there and derivation yields "".
Startup used to warn and carry on, which left OnDemandTiles addressing
`/content//<title>` while reporting on-demand rendering as enabled.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="serve extra not installed")
from pixelrag_serve import api


def _args(zim_book=None):
    return SimpleNamespace(zim_book=zim_book, kiwix_url="http://kiwix.invalid")


def test_explicit_book_skips_derivation(monkeypatch):
    monkeypatch.setattr(
        api, "_derive_kiwix_book", lambda url: pytest.fail("should not be called")
    )
    assert api._require_kiwix_book(_args("wikipedia_en_all")) == ("wikipedia_en_all")


def test_derived_book_is_used(monkeypatch):
    monkeypatch.setattr(api, "_derive_kiwix_book", lambda url: "wikipedia_en_all")
    assert api._require_kiwix_book(_args()) == "wikipedia_en_all"


def test_undeterminable_book_stops_startup(monkeypatch):
    monkeypatch.setattr(api, "_derive_kiwix_book", lambda url: "")
    with pytest.raises(SystemExit, match="--zim-book"):
        api._require_kiwix_book(_args())


def test_unreadable_catalog_reports_the_cause(caplog):
    # Nothing is listening on .invalid, which is the shape of pointing this at
    # the bundled zim_server: the catalog request simply fails.
    with caplog.at_level("WARNING"):
        assert api._derive_kiwix_book("http://kiwix.invalid") == ""
    assert "catalog at http://kiwix.invalid unreadable" in caplog.text


def test_catalog_without_a_book_is_reported_separately(monkeypatch, caplog):
    class _Resp:
        def read(self):
            return b"<feed></feed>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=10: _Resp())
    with caplog.at_level("WARNING"):
        assert api._derive_kiwix_book("http://kiwix.invalid") == ""
    assert "no /content/<book> in the catalog" in caplog.text
