"""pixelshot: Document to image tiles.

Renders web pages, PDFs, and local files as tiled screenshots.
"""

from .render import render_file, render_pdf, render_url

__all__ = ["render_file", "render_pdf", "render_url"]
