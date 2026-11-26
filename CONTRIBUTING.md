# Contributing to PixelRAG

Contributions are welcome. This page covers the things that are easy to get
wrong here and expensive to find out late.

## Development setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
git clone https://github.com/StarTrail-org/PixelRAG.git
cd PixelRAG
uv sync --extra dev
```

Rendering needs a patched headless Chrome, which the test suite downloads on
first use:

```bash
uv run pixelshot install-chrome
```

## What CI runs

Six jobs. The main lint and test commands are quick to reproduce locally:

```bash
uvx ruff check .            # Python lint
uvx ruff format --check .   # formatting
uv run pytest tests/        # Tests
```

The lockfile job checks every uv project discovered from its `pyproject.toml`:

```bash
set -euo pipefail
while IFS= read -r -d '' manifest; do
    uv lock --check --project "${manifest%/pyproject.toml}"
done < <(find . -name pyproject.toml -not -path '*/.venv/*' -print0)
```

`demos/` and `train/` are **in** lint scope (the exclude list is `tmp`, `docs`,
`skill`). They were excluded until recently, so older files in those trees may
still need a `ruff check --fix` and `ruff format` pass when you touch them.

The other three — Demo tests, Frontend build (`npm ci && npm test && npm run
build`) and Plugin validate — run from the same workflow.

## If your PR shows no checks at all

GitHub holds workflow runs from first-time contributors until a maintainer
approves them. A new PR with only a Vercel entry and no CI jobs is waiting for
that approval, not failing. You don't need to do anything; ping the PR if it
sits for more than a day or two.

The Vercel check fails on every fork PR (`Authorization required to deploy`)
and is not a signal about your change.

## Adding support for a hosted inference provider

**Read the "Hosted reader integration policy" in `eval/README.md` before
writing code.** The short version: an OpenAI-compatible endpoint is reachable
today with no changes to this repository.

```bash
uv run eval/run_bench.py --task simpleqa \
    --model openai/<provider-model-id> \
    --api-base https://provider.example/v1 \
    --api-key "$YOUR_KEY" \
    --litellm --num-examples 1 --max-concurrent 1 --max-tokens 256
```

The `openai/` prefix picks LiteLLM's OpenAI-compatible transport explicitly, so
a model id containing `gemini` or `claude` will not be routed to a native SDK.

A dedicated `--provider` flag is reserved for a capability that path genuinely
cannot express, and such a PR must document and test the missing capability.
Several careful, fully tested PRs have been closed under this rule — not
because the code was poor, but because each one added a flag, a config
function, a dispatch branch and a docs row to express three constants we
already accept on the command line.

If a provider is worth documenting, a row in `eval/README.md` is welcome.

## Adding a demo

Demos live in their own directory under `demos/` and should not change anything
outside it. A demo should show something about PixelRAG — if the PixelRAG part
could be swapped for any other retrieval library without changing the point, it
belongs in your own repository with a link from ours.

Demos are linted and their test suites run in CI.

## Commits and PRs

- Conventional commits (`fix(serve): ...`, `feat(index): ...`, `docs: ...`).
- One subject per PR. A dependency change and a lint-scope change are two PRs,
  even when they touch the same file — they fail for different reasons and get
  reviewed by different eyes.
- If you change dependencies in `pyproject.toml`, run `uv lock` and commit the
  result. Without it `uv sync` re-resolves from scratch in CI, which both
  masks what you changed and can fail on platforms you never tested.
- A fix should come with a test that fails without it. That is the difference
  between a reviewer believing you and a reviewer verifying you.
