# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub:
[**Report a vulnerability**](https://github.com/StarTrail-org/PixelRAG/security/advisories/new)
(repository → **Security** → **Advisories** → *Report a vulnerability*).

That opens a private advisory visible only to you and the maintainers, needs no
infrastructure on either side, and gives us somewhere to discuss a fix and issue
a CVE if one is warranted.

If the advisory form is unavailable to you, open a public issue saying only that
you have a security report and how to reach you — no details — and a maintainer
will arrange a private channel.

## What to include

The more of this you can provide, the faster a fix lands:

- what an attacker achieves, and what access they need to start
- affected versions, plus the platform and Python version you saw it on
- a reproduction — a script, or the commands you ran
- anything you already know about the cause or a possible fix

## Scope

PixelRAG renders untrusted documents and web pages and serves retrieval over the
result, so the interesting boundaries are where untrusted input meets execution:

- **`pixelshot` capture** — a headless Chrome/CDP process pointed at attacker-controlled
  pages. Sandbox escapes, navigation to local or internal addresses, and anything that
  escalates from page content to host access.
- **Tiles as model input** — a tile is a screenshot of a page we did not write. Text inside
  it reaching a model as *instructions* rather than data is in scope, including via the
  `pixelbrowse` plugin, which grants `Bash`.
- **The index pipeline** — path traversal, symlink following, and archive extraction when
  ingesting local files or crawled content.
- **The serve API** — `/search`, `/reconstruct`, `/tile` and the on-demand render path:
  unbounded work per request, reads outside the tile directory, or leaking host paths.
- **Credentials** — API keys and endpoints for third-party readers configured in `eval/`.

Out of scope: findings against a deployment you have deliberately exposed to a network
(for example an embedding server bound beyond loopback and reachable from the internet),
and vulnerabilities in third-party dependencies that are already public — report those
upstream, though we do want to know if PixelRAG pins an affected version.

## Supported versions

Fixes land on `main` and ship in the next release. If you are on an older release,
please confirm the issue reproduces on current `main` where you can.

## Disclosure

We will confirm receipt, keep you updated as we investigate, and credit you in the
advisory unless you would rather stay anonymous. Please give us a reasonable window
to ship a fix before disclosing publicly.
