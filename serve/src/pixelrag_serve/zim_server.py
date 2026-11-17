"""Multi-threaded ZIM HTTP server. Replaces kiwix-serve for benchmarking.

kiwix-serve is single-threaded — becomes a bottleneck at 32+ concurrent
Chrome processes. This server uses ThreadPoolExecutor for parallel ZIM reads.

Usage:
    python -m pixelrag_serve.zim_server --zim /path/to/wikipedia.zim --port 9454
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import unquote

from libzim.reader import Archive


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ZimHandler(BaseHTTPRequestHandler):
    archive = None
    book_name = None

    def do_HEAD(self):
        self._handle(head_only=True)

    def do_GET(self):
        self._handle(head_only=False)

    def _handle(self, head_only=False):
        path = unquote(self.path)

        # Strip /content/{book_name}/ prefix
        prefix = f"/content/{self.book_name}/"
        if path.startswith(prefix):
            entry_path = path[len(prefix) :]
        elif path.startswith("/"):
            entry_path = path[1:]
        else:
            entry_path = path

        # Strip query string
        if "?" in entry_path:
            entry_path = entry_path.split("?")[0]

        try:
            if not self.archive.has_entry_by_path(entry_path):
                self.send_error(404, f"Not found: {entry_path}")
                return

            entry = self.archive.get_entry_by_path(entry_path)
            item = entry.get_item()
            content = bytes(item.content)
            mimetype = item.mimetype.split(";")[0].strip()

            self.send_response(200)
            self.send_header("Content-Type", mimetype)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            if not head_only:
                self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        pass  # suppress access logs


def main():
    parser = argparse.ArgumentParser(description="Multi-threaded ZIM HTTP server")
    parser.add_argument("--zim", required=True, help="Path to ZIM file")
    parser.add_argument("--port", type=int, default=9454)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    archive = Archive(args.zim)
    book_name = Path(args.zim).stem

    ZimHandler.archive = archive
    ZimHandler.book_name = book_name

    server = ThreadedHTTPServer((args.host, args.port), ZimHandler)
    print(f"ZIM server: http://{args.host}:{args.port}/content/{book_name}/")
    print(f"ZIM: {args.zim} ({archive.article_count:,} articles)")
    print("Threading: unlimited (one thread per request)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
