"""Render smoke test — the README's `pixelshot <page> --output ./tiles` path.

Renders a local HTML file through the default CDP backend (patched headless
Chrome, auto-downloaded and cached in CI) and asserts tile images are produced.
This is the one user-facing capability the README promises, so it is tested end
to end rather than mocked.
"""

import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pixelrag_render import render_file
from pixelrag_render.backends.page_metrics import truncation_reason


def test_render_local_html_to_tiles(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        "<html><body><h1>PixelRAG render smoke</h1>"
        + "<p>lorem ipsum dolor sit amet. </p>" * 60
        + "</body></html>"
    )
    out = tmp_path / "tiles"

    dirs = render_file(html, out)

    assert dirs, "render_file returned no tile directories"
    tile_dir = Path(dirs[0])
    tiles = sorted(tile_dir.glob("tile_*.jpg"))
    assert tiles, (
        f"no tile images produced in {tile_dir} "
        f"(contents: {[p.name for p in tile_dir.iterdir()]})"
    )


def test_page_taller_than_a_viewport_bounded_body_is_fully_tiled(tmp_path):
    """A page whose body is pinned to the viewport must still tile in full.

    Regression for issue #124. Sites that set ``html, body { height: 100% }``
    (Wikipedia's Vector 2022 skin among them) leave the article content
    overflowing the body box visibly, so ``body.getBoundingClientRect()`` is one
    viewport tall on a 20,000px page. The readiness probe used to clamp the page
    height to that rect, capturing a single tile and reporting the viewport as
    the page height.
    """
    body = "".join(f"<p>line {i:03d}</p>" for i in range(400))
    html = tmp_path / "viewport_bounded_body.html"
    html.write_text(
        '<!DOCTYPE html><html style="height:100%"><body style="height:100%">'
        f"{body}</body></html>"
    )
    out = tmp_path / "tiles"

    dirs = render_file(html, out, tile_height=1000, viewport_width=1280)

    tile_dir = Path(dirs[0])
    manifest = json.loads((tile_dir / "tiles.json").read_text())
    tiles = sorted(tile_dir.glob("tile_*.jpg"))

    # 400 paragraphs are several viewports tall whatever the default font is;
    # assert against the viewport rather than a brittle exact pixel count.
    assert manifest["page_height"] > 3000, (
        f"page_height {manifest['page_height']} is about one viewport — the "
        "content below the fold was never measured"
    )
    assert len(tiles) > 1, f"expected multiple tiles, got {[t.name for t in tiles]}"


def test_hanging_web_font_does_not_stall_render(tmp_path):
    """A font request that never finishes must not block the CDP renderer."""
    release_font = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/hanging.woff2":
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Content-Length", "1000000")
                self.end_headers()
                self.wfile.write(b"\0")
                self.wfile.flush()
                release_font.wait(timeout=60)
                return

            body = "".join(f"<p>line {i:03d}</p>" for i in range(200))
            html = f"""<!doctype html>
                <html><head><script>
                window.addEventListener("load", () => {{
                    const style = document.createElement("style");
                    style.textContent = `@font-face {{
                        font-family: HangingFont;
                        src: url('/hanging.woff2') format('woff2');
                    }} body {{ font-family: HangingFont; }}`;
                    document.head.appendChild(style);
                    document.fonts.load("16px HangingFont");
                }});
                </script></head><body>{body}</body></html>"""
            payload = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    output = tmp_path / "tiles"
    url = f"http://127.0.0.1:{server.server_port}/"
    script = (
        "from pixelrag_render import render_url; import sys; "
        "render_url(sys.argv[1], sys.argv[2], tile_height=300, "
        "viewport_width=400, workers=1, turbo=False)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, url, str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
            pytest.fail(
                "render remained blocked by document.fonts.ready for 30 seconds\n"
                f"stdout:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}"
            )
    finally:
        release_font.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert proc.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    manifests = list(output.rglob("tiles.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["page_height"] > 300
    assert len(manifest["tiles"]) > 1


def test_manifest_records_the_geometry_the_capture_ran_at(tmp_path):
    """A consumer of tiles.json must be able to check the capture itself.

    ``page_height`` alone says nothing: it is only suspicious relative to the
    tile height, which used to be absent from URL manifests entirely, leaving a
    downstream reader guessing which ``--tile-height`` the capture used.
    """
    body = "".join(f"<p>line {i:03d}</p>" for i in range(400))
    html = tmp_path / "long.html"
    html.write_text(f"<!DOCTYPE html><html><body>{body}</body></html>")

    dirs = render_file(html, tmp_path / "tiles", tile_height=1000, viewport_width=1280)
    manifest = json.loads((Path(dirs[0]) / "tiles.json").read_text())

    assert manifest["tile_height"] == 1000
    assert manifest["viewport_width"] == 1280
    assert manifest["complete"] is True, (
        "a page measured well past its viewport is a healthy capture and must "
        f"not be flagged: {manifest}"
    )


def test_capture_stuck_at_the_viewport_height_is_reported_incomplete(tmp_path):
    """The one page height that cannot be trusted must not claim completeness.

    A measured height exactly equal to the tile height (= the emulated viewport)
    is the signature of every truncation so far: the probe tracked the viewport
    instead of the content. The capture still runs — the tile written is real —
    but the manifest has to say the page may not be all there, because nothing
    else downstream can tell.
    """
    html = tmp_path / "one_viewport.html"
    html.write_text(
        '<!DOCTYPE html><html><body style="margin:0">'
        '<div style="height:1000px">content</div></body></html>'
    )

    dirs = render_file(html, tmp_path / "tiles", tile_height=1000, viewport_width=1280)
    tile_dir = Path(dirs[0])
    manifest = json.loads((tile_dir / "tiles.json").read_text())

    assert manifest["page_height"] == manifest["tile_height"] == 1000, (
        f"test setup no longer produces a viewport-height page: {manifest}"
    )
    assert manifest["complete"] is False
    assert sorted(tile_dir.glob("tile_*.jpg")), "the tile itself is still written"


def test_truncation_reason_rules():
    # Healthy: measured, and not pinned to the viewport.
    assert truncation_reason(29184, 1568, measured=True) is None

    # Probe failed — the height is the tile height by fallback, not by measurement.
    assert "fell back" in truncation_reason(1568, 1568, measured=False)

    # Measured, but exactly one viewport: the truncation signature.
    assert "exactly the tile height" in truncation_reason(1568, 1568, measured=True)
