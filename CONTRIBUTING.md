# Contributing

Keep the research code easy for newcomers to read: one public class per file, simple names, meaningful type annotations, no import-time network/model downloads, and no duplicated notebook algorithms.

## Local setup

```bash
uv python install 3.11
uv sync --group dev
uv run pre-commit install
uv run dar smoke --offline --config configs/experiments/smoke.yaml
make check
```

If `uv.lock` is unavailable, generate it **with network access** via `uv lock`, commit the generated file, and then use `uv sync --locked`. Never hand-edit a lockfile. Full training dependencies are optional: `uv sync --extra train --extra benchmark --group dev`.

A PR should describe its issue, reason, dataset/benchmark impact, sample commands, tests, and any licensing or documentation impact. Tests must be offline by default. Tag real network/CUDA tests with `network`, `gpu`, or `slow` markers and include a real hardware report when training GPU code changes.

Training changes need: an isolated regression test, a tiny negative-mining test, evidence of validation-only checkpoint selection, and an explicit CUDA preflight. A benchmark change requires demonstrating same first-stage candidates for S3/S4 and S5/S6. Never publish fabricated improvements or treat synthetic-smoke metrics as real scores.
