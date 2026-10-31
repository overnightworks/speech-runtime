Audience: every agent that edits this repository.

Code policy and coordination live once, in the global contract. This file adds
only this repository's facts.

## Projects

The root is the speech service (`src/`, `tests/`, `scripts/`). Each
`providers/<name>` is its own uv project with its own `uv.lock`;
`providers/provider-contract` is the shared source the others install from.
`packages/client` is the speech wire's client distribution, a uv workspace member of the root: linted and tested from the root, and the one owner of the wire's types.
The GPU providers are live code: the models are kept and used. An advisory no
relock reaches is allowlisted in `quality/config.toml` against
overnightworks/speech-runtime#2, with the model-library cap that holds it.

## Checks

CI calls the shared quality gate from the `quality` job of
`.github/workflows/ci.yml`, pinned by SHA. Its required check is
`quality / verdict`; `quality/config.toml` lists the jobs that must be green
for it, so a new mandatory CI job is added there by its display name. Secret
scanning and the SonarCloud scan run inside the gate, and
`sonar-project.properties` may carry only the keys the gate allows.

The `keycloak-token-shapes` job serves the throwaway realm of
`tests/fixtures/keycloak/realm.json` from Keycloak and runs the tests marked
`keycloak` (`tests/test_callers_keycloak.py`) from the root's locked environment;
the default run deselects them. Run them locally only with a Keycloak you start
yourself on a free port, under the probe-stack lock, and `KEYCLOAK_URL` naming it.

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

`providers/provider-contract` has no ruff and no lock of its own. Lint it with
Chatterbox's locked ruff, from `providers/provider-contract`, and never `uv run ruff` there, which leaves a stray `uv.lock`:

```bash
uv run --locked --project ../chatterbox ruff check --config pyproject.toml .
```

`scripts/prove_speech.py` starts the providers and takes `/tmp/probe-stack.lock`.
