"""Bounded wait for the process-global render lock.

Renders are serialized process-wide, so without a bound a caller queues behind
another request's entire render budget (16 renders x 120s worst case) while
holding a thread from the event loop's executor. Giving up yields no image, not
no response.

`threading.Lock` is neither reentrant nor owner-tracked, so a same-thread
acquire exercises the contended path without spawning a thread.
"""

import os
import time

import pytest
from pixelrag_serve import render_ondemand as ro


@pytest.fixture
def tiles(tmp_path, monkeypatch):
    monkeypatch.setattr(ro, "_RENDER_QUEUE_TIMEOUT", 0.05)
    od = ro.OnDemandTiles("http://kiwix.invalid", "book", str(tmp_path / "cache"))
    yield od
    if ro._render_lock.locked():  # never leave a wedged lock behind
        ro._render_lock.release()


def _assert_lock_free():
    assert ro._render_lock.acquire(blocking=False), "render lock was not released"
    ro._render_lock.release()


def test_gives_up_when_the_queue_is_busy(tiles):
    ro._render_lock.acquire()  # stand in for another request mid-render
    try:
        t0 = time.monotonic()
        assert tiles.chunk_path(1, "Title", 0, 0) is None
        # Bounded by the queue timeout, not by _RENDER_TIMEOUT (120s).
        assert time.monotonic() - t0 < 5
    finally:
        ro._render_lock.release()


def test_lock_is_released_after_a_failed_render(tiles):
    def _boom(article_id, title):
        raise RuntimeError("render blew up")

    tiles._render_and_chunk = _boom
    assert tiles.chunk_path(1, "Title", 0, 0) is None
    _assert_lock_free()


def test_in_lock_recheck_skips_a_render_the_race_made_redundant(tiles):
    # The chunk lands between the early cache check and our turn at the lock.
    # Reaching the in-lock re-check needs that early check to miss, so stub it:
    # with the file merely on disk, chunk_path returns before taking the lock
    # and this branch goes untested.
    cpath = tiles._chunk_file(1, 0, 0)
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    with open(cpath, "wb") as fp:
        fp.write(b"png")
    tiles.cached_chunk_path = lambda *a: None  # as if not yet rendered

    def _boom(article_id, title):
        raise AssertionError("must not render when the chunk already exists")

    tiles._render_and_chunk = _boom
    assert tiles.chunk_path(1, "Title", 0, 0) == cpath
    _assert_lock_free()


def test_lock_is_released_after_a_successful_render(tiles):
    cpath = tiles._chunk_file(2, 0, 0)

    def _render(article_id, title):
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        with open(cpath, "wb") as fp:
            fp.write(b"png")

    tiles._render_and_chunk = _render
    assert tiles.chunk_path(2, "Title", 0, 0) == cpath
    _assert_lock_free()


def test_untitled_article_never_touches_the_lock(tiles):
    assert tiles.chunk_path(3, "", 0, 0) is None
    _assert_lock_free()
