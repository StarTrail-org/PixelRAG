"""Shared in-page measurement JS for the capture backends.

Kept in one place because both backends measure page height the same way and
must agree: the standard (``cdp``) and turbo (``fast_cdp``) paths each embed
this snippet, and a divergence between them silently changes how much of a page
gets captured depending on which Chrome binary is installed.

No imports — a plain string constant, so either backend can use it without a
dependency edge between them.
"""

# How tall is the document's *content*?
#
# `documentElement.scrollHeight` alone over-reports: padding on the root element
# or a trailing margin inflates it, buying a run of blank tiles at the bottom of
# every such page. So it is clamped to where the content actually ends.
#
# That bound has to be measured from the content, not from the body box. A body
# is only as tall as the document when the page lets it size to its content;
# sites that pin it to the viewport (`html, body { height: 100% }` — Wikipedia's
# Vector 2022 skin among them) leave the content overflowing a one-viewport box,
# and clamping to that box truncates a 20,000px article to a single tile
# (issue #124). Taking the lowest edge among the body and its element children
# reads the same on a self-sizing body and survives a pinned one.
#
# Coordinates are viewport-relative, so scroll offset is added back: a page
# navigated to a `#fragment` lands scrolled down, where a raw rect bottom would
# under-report by exactly the scrolled distance.
CONTENT_BOTTOM_JS = """
    function contentBottom(body) {
        const offset = window.scrollY || window.pageYOffset || 0;
        let bottom = body.getBoundingClientRect().bottom;
        for (let el = body.firstElementChild; el; el = el.nextElementSibling) {
            const r = el.getBoundingClientRect();
            // Skip elements with no box at all (display:none, empty <script>);
            // theirs is a zero rect at the origin and would not move `bottom`,
            // but skipping keeps the intent explicit.
            if (r.width > 0 || r.height > 0) {
                bottom = Math.max(bottom, r.bottom);
            }
        }
        return Math.ceil(bottom + offset);
    }
"""
