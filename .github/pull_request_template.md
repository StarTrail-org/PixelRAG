## What does this PR do?

<!-- Brief description of your changes, and the symptom if this is a fix. -->

## Related Issues

Fixes #

## Checklist

- [ ] Lint passes (`uvx ruff check .` and `uvx ruff format --check .`) — note this covers `demos/` and `train/` as well
- [ ] Tests pass (`uv sync --extra dev && uv run pytest tests/`)
- [ ] New behaviour has a test that fails without the change

<!--
A few things worth knowing before you spend time on this:

**CI will not start on its own for a first-time contributor.** GitHub holds the
workflow run until a maintainer approves it, so "no checks" on a new PR means
it is queued, not broken. Nothing is needed from you.

**Adding support for a hosted inference provider?** Please read the "Hosted
reader integration policy" in `eval/README.md` first. An OpenAI-compatible
endpoint needs no code at all:

    uv run eval/run_bench.py --model openai/<model-id> \
        --api-base https://provider.example/v1 --api-key "$KEY" --litellm

A dedicated provider flag is reserved for a capability that path cannot
express, and the PR has to document and test that capability. We have closed
several well-written PRs under this rule; we would rather you know now.

**Adding a demo under `demos/`?** Keep it to its own directory and make sure it
shows something about PixelRAG rather than only about the third-party service.
-->
