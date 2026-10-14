# Contributing to muse-gateway

muse-gateway is an unofficial Muse Code provider gateway forked from
[muse-shim](https://github.com/luckeyfaraday/muse-shim). Keep changes focused,
preserve upstream attribution, and document user-visible behavior.

## Development and tests

Python 3.10+ is required. The runtime has one dependency, protobuf; the tests
use the standard-library unittest runner. From the repository root:

```bash
python -m pip install -e .
python -m unittest -v
python -m muse_gateway --help
```

Use `python3` if that is your Python executable. You can create a project-local
environment with `python -m venv .venv` and activate it before installing.
The installed `muse-gateway launch` command and source wrappers use the same
subscription launcher; see [README.md](README.md).

The test suite uses local HTTP fakes and mocked provider calls. It does not
prove live account access or full compatibility with every Muse release.
Never make ordinary tests depend on personal credentials or paid inference.

## Areas of the codebase

- `muse_gateway.py`: model routing/catalogs, request transforms, HTTP server,
  diagnostics, the unified `launch` command, and the inherited low-level
  `run` launcher.
- `muse_providers.py`: provider authentication and protocol adapters.
- `muse-gateway` and `muse-gateway.ps1`: thin wrappers forwarding to
  `python muse_gateway.py launch`.
- `test_muse_shim.py` and `test_native_providers.py`: provider regression tests.
- `test_launchers.py`: launcher contract tests (wrapper and direct `launch`
  startup, argument splitting, env-file loading).

The `muse_shim` module alias, the `muse-shim` command alias, `MUSE_SHIM_*`
settings, and legacy protocol identifiers are compatibility interfaces.
Avoid cosmetic renames that break existing setups. Use **muse-gateway** in new user-facing documentation and messages.

## Change guidelines

- Preserve native Meta namespace tools and reasoning/cache fields.
- Keep flattened tool schemas and historical function-call names consistent
  on non-Meta routes.
- Codex inference uses HTTP. Claude deliberately uses official `claude -p`
  by default, with its built-in tools disabled; Muse executes returned tools.
- Prefer extending an existing behavioral contract test over adding a test for
  each small change. Cover successful and degraded outcomes together. A mocked
  response, log-line assertion, or source-string check is not evidence of a
  working Muse tool loop; validate multi-turn claims through Muse itself.
- Update documentation with behavior changes, not session transcripts or
  development runlogs. Put personal handoff notes in ignored local files.
- Avoid new dependencies unless their value warrants the maintenance cost.

## Pull requests

Describe the user-visible change, tests run, and any compatibility implications.
Keep the patch small and reviewable. Do not include `.env`, credential files,
`CARRYOVER.md`, private logs, personal paths, or generated caches.

Report vulnerabilities as described in [SECURITY.md](SECURITY.md), not in public
issues. Participation follows [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
