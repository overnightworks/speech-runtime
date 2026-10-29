Audience: every agent that edits this repository.

Code policy and coordination live once, in the global contract. This file adds
only this repository's facts.

## Projects

The root is the speech service (`src/`, `tests/`, `scripts/`). Each
`providers/<name>` is its own uv project with its own `uv.lock`;
`providers/provider-contract` is the shared source the others install from.
The GPU providers are frozen under overnightworks/speech-runtime#2: no
hardening, no new tests, and their advisories are allowlisted against that item.

## Checks

CI calls the shared quality gate from the `quality` job of
`.github/workflows/ci.yml`, pinned by SHA. Its required check is
`quality / verdict`; `quality/config.toml` lists the jobs that must be green
for it, so a new mandatory CI job is added there by its display name. Secret
scanning and the SonarCloud scan run inside the gate, and
`sonar-project.properties` may carry only the keys the gate allows.

Each speech job writes a Cobertura report (`coverage-<job>.xml`, paths relative
to the repository root, set by `[tool.coverage.run]` in the root `pyproject.toml`);
the `coverage` job gathers the five into the artifact the gate hands to SonarCloud.

The gate's root-layout rule holds the root entries: a new root entry is declared
in `quality/config.toml` with a reason, or lives in its owner's directory.

Targeted local commands, run in the project's own directory:

```bash
uv run --locked ruff check
uv run --locked ruff format --check
uv run --locked pytest -q <named tests>
```

`scripts/prove_speech.py` starts the providers and takes `/tmp/probe-stack.lock`.
