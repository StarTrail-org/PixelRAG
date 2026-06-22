#!/usr/bin/env python3
"""
repro_bug1.py — Minimal reproduction of the int() crash on non-numeric local IDs.

Reproduce WITHOUT needing a full PixelRAG installation:
    python repro_bug1.py

Expected on upstream v0.2.1 (before fix):
    ValueError: invalid literal for int() with base 10: 'my_photo'

Expected after fix:
    PASS — article_entries has 2 entries with display titles.
"""

# Simulate the upstream article list from the pipeline (strings from filename stems)
articles = [
    {"id": "my_photo",        "url": "",  "path": "/tmp/my_photo.jpg",  "metadata": {}},
    {"id": "chart_btcusdt",   "url": "",  "path": "/tmp/chart.png",     "metadata": {}},
]

print("Testing upstream behavior (BEFORE fix):")
print("=" * 50)
try:
    max_idx = max(int(a["id"]) for a in articles) + 1 if articles else 0
    print(f"  max_idx = {max_idx}")
    article_entries = [{"title": "", "url": ""}] * max_idx
    for a in articles:
        idx = int(a["id"])            # <-- crashes here on non-numeric IDs
        article_entries[idx] = {"title": str(idx), "url": a.get("url", "")}
    print("  PASS (unexpected — upstream has numeric IDs in this path)")
except ValueError as e:
    print(f"  FAIL (expected on upstream): {e}")

print()
print("Testing FIXED behavior (enumerate-based):")
print("=" * 50)
article_entries = []
for enum_idx, a in enumerate(articles):
    title = a.get("metadata", {}).get("title", "")
    if not title and a.get("url"):
        title = a["url"].split("/")[-1]
    if not title:
        title = a.get("id", str(enum_idx))     # filename stem as fallback title
    url = a.get("url", "") or a.get("path", "")
    article_entries.append({"title": title, "url": url})

for i, entry in enumerate(article_entries):
    print(f"  [{i}] title={entry['title']!r:20s}  url={entry['url']!r}")

assert len(article_entries) == 2
assert article_entries[0]["title"] == "my_photo"
assert article_entries[1]["title"] == "chart_btcusdt"
print()
print("PASS — all assertions OK")
