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
python -m tests.check_regressions
python -m muse_gateway --help
```

Use `python3` if that is your Python executable. You can create a project-local
environment with `python -m venv .venv` and activate it before installing.
The installed `muse-gateway launch` command and source wrappers use the same
subscription launcher; see [README.md](README.md).

### What a green result means

The checks cover different boundaries. Report which ones you ran, including
any platform skips. HTTP status and log messages alone do not establish that
Muse accepted a stream or completed a tool turn.

| Check | Evidence it provides | What it does not prove |
| --- | --- | --- |
| `python -m unittest -v` | Local schema/auth/transform/security tests, launcher startup checks, and public HTTP contracts below | Live account access, a working TUI session, or every Muse release |
| `python -m tests.check_regressions` | Selected historical defects actually make the behavioral tests fail | An exhaustive mutation score or absence of other bugs |
| `python -m tests.check_muse --muse-bin /path/to/muse-binary` | Installed Muse accepts translated text, executes a file tool and continues, and explicitly rejects a missing-sequence stream | TUI model picker/resume, live provider entitlement, or Claude CLI installation |

The HTTP contracts exercise routing, framing and converters against loopback
protocol peers. They check stream/final agreement, tool identity and arguments,
large-history replay, context budgets, error outcomes, bounded recovery, and
incremental delivery. These are protocol checks, not live provider tests.

The regression challenge runner deliberately introduces selected historical
defects in temporary source copies outside the checkout. It requires a passing
baseline and the expected behavioral failures. A surviving fault, skip, setup
error, timeout, or mutation that no longer applies fails the command.

For consumer acceptance, provide the actual Muse executable rather than an
auto-updating shell wrapper. The command uses temporary configuration and
workspace directories, dummy credentials and local upstreams, with no paid
inference. It requires successful run completion, actual file-tool execution
and replay, and an explicit decode rejection for the negative control.
Missing executables and inconclusive results fail rather than skip. Its known
Muse model profile is pinned to Devin/Claude HTTP at the gateway, so it does
not exercise the TUI's model picker or Claude CLI installation.

For a release, record the Muse version and consumer result, then check the
TUI catalog, a new session and a resumed cross-provider session with the intended
accounts. Hosted CI runs the offline checks and installed-import smoke test;
it does not certify live account access. Ordinary tests must not require
personal credentials or paid inference. Socket failures fail the suite;
narrow platform skips, such as unavailable symlink privileges, are reported.

### Keeping tests useful

- Assert observable behavior and failure outcomes. Mock external credential,
  network or process boundaries when useful, not the converter/handler whose
  compatibility the test purports to establish.
- Use independent protocol fixtures or documented golden vectors. An
  encoder/decoder round trip alone can reproduce the same bug twice.
- When fixing a bug, demonstrate the regression test failing before the fix
  and passing after it. Add a selected fault challenge for high-risk contracts.
- Remove weaker/duplicate tests once a stronger check covers their behavior.
  Do not preserve method-call, source-string, log-only or substring assertions
  as supposed evidence of a working model/tool turn.
- Keep fixtures synthetic and small, except where size is itself the contract.
  Do not publish personal sessions, live tokens, or raw captures as fixtures.

Background: [layered testing](https://martinfowler.com/articles/practical-test-pyramid.html)
and [mutation testing](https://mutmut.readthedocs.io/en/latest/).
Protocol references: [Connect](https://connectrpc.com/docs/protocol/)
and [SSE](https://html.spec.whatwg.org/multipage/server-sent-events.html).

## Areas of the codebase

- `muse_gateway.py`: model routing/catalogs, request transforms, HTTP server,
  diagnostics, the unified `launch` command, and the inherited low-level
  `run` launcher.
- `muse_providers.py`: provider authentication and protocol adapters.
- `muse_devin.py`, `muse_devin_wire.py`: Devin credentials, catalog, history,
  stream conversion, and Connect transport. `devin.proto` and `devin_pb2.py`
  are the wire schema and generated bindings; do not hand-edit the bindings.
- `muse-gateway` and `muse-gateway.ps1`: thin wrappers forwarding to
  `python muse_gateway.py launch`.
- `test_muse_shim.py` and `test_native_providers.py`: provider regression tests.
- `test_devin.py`: Devin schema golden vectors, credentials, discovery,
  translation, and stream-state tests.
- `test_launchers.py`: launcher contract tests (wrapper and direct `launch`
  startup, argument splitting, env-file loading).
- `tests/`: public HTTP contracts, protocol peers, selected fault challenges,
  and the explicit installed-Muse acceptance command.
- `test_capture.py`: private diagnostic capture and related security checks.

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
- Keep documentation about the project, not a development session. Store
  transcripts, personal handoff notes, captures, and one-off audit scripts
  outside the checkout. Keep reusable regression tests in the test suite.
- Check README examples against the installed entry point and source wrappers.
  Distinguish installing the gateway from the project Muse will work in, and
  document which commands load `.env`. Keep `.env.example` opt-in so copying
  it does not silently enable an unwanted local endpoint.
- Avoid new dependencies unless their value warrants the maintenance cost.

## Pull requests

Describe the user-visible change, tests run, and any compatibility implications.
Keep the patch small and reviewable. Do not include `.env`, credential files,
`CARRYOVER.md`, private logs, personal paths, or generated caches.

Report vulnerabilities as described in [SECURITY.md](SECURITY.md), not in public
issues. Participation follows [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
