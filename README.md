# muse-gateway

Use Muse Spark, OpenAI Codex, Claude, and local models from Muse Code's `/model`
menu. muse-gateway is an unofficial loopback proxy: Muse keeps the coding-agent
harness and executes tools; the gateway routes model requests to your providers.

Forked from [luckeyfaraday/muse-shim](https://github.com/luckeyfaraday/muse-shim),
with native Meta forwarding, local endpoint discovery, and Windows/Bash launchers.
The upstream MIT license and attribution are retained. This project is not
affiliated with Meta, OpenAI, or Anthropic.

## Requirements

- Python 3.10 or newer; no third-party Python runtime dependencies.
- Muse Code installed as `muse` and authenticated with `muse login`.
- Optional: a local or LiteLLM endpoint supporting **OpenAI Responses** with
  streaming and function tools. A Chat Completions-only endpoint needs an adapter.
- Optional: an existing `codex login` or an installed, authenticated Claude Code
  CLI for those providers.

## Quick start

Download or clone this repository and open a terminal in its directory.
There is one launcher interface, available three ways: the source wrappers
(`.\muse-gateway.ps1` on Windows, `bash ./muse-gateway` on Linux/macOS/WSL)
forward to the cross-platform `launch` subcommand, which is also the installed
`muse-gateway launch`. All three behave identically: they preserve Muse's
subscription authentication (no Meta API key needed), start the gateway, run
Muse, and stop their gateway process when Muse exits.

```powershell
.\muse-gateway.ps1
```

```bash
bash ./muse-gateway
```

```bash
muse-gateway launch   # installed console command, same behavior
```

Inside Muse, open `/model` and select a `Meta`, `Codex`, `Claude`, or `Local`
entry. A catalog entry does not prove account access: Codex and Claude fallback
entries may appear before you authenticate with those providers.

### Add a local endpoint

Copy `.env.example` to `.env` beside the launchers (or `./.env` for the
installed command), then edit it:

```dotenv
MUSE_SHIM_LOCAL_UPSTREAM=http://127.0.0.1:4000/v1
MUSE_SHIM_LOCAL_API_KEY=replace-with-your-upstream-key
# Optional: restrict the menu to these upstream model IDs.
# MUSE_SHIM_LOCAL_MODELS=my-coder,my-reasoner
```

The `MUSE_SHIM_*` prefix is retained for configuration compatibility with the
upstream project. Do not commit `.env` or credentials.

The launcher reads configuration in this order: command-line flags, nonempty
process environment variables, then `.env`. It does not create a config file.
It warns and continues without requiring a local key; setting the key enables
local-backend preflight. Clear unused `MUSE_SHIM_LOCAL_*` variables if you want no
local entries, since the Python server still honors explicit upstream/model
configuration independently of that launcher preflight. The wrappers pass
`--env-file <checkout>/.env`; the installed command falls back to `./.env`
— pass `--env-file` to be explicit.

With an upstream configured, the gateway discovers IDs from `GET /models`.
`MUSE_SHIM_LOCAL_MODELS` restricts that list when set. Discovery reads common
limit keys (`max_input_tokens`, `context_window`, `max_output_tokens`,
`max_completion_tokens`, and similar aliases) and image-input evidence
(`capabilities`, `modalities`, `supports_vision`, and similar fields) when the
endpoint provides them. Missing limits fall back to 128,000 context tokens and
32,768 output tokens, labeled **limits unverified** in the menu; these are not
measurements of your model's capabilities. Without image evidence, local entries
advertise text input only. A bare Ollama server (upstream ending in `/v1`) is
listed via its native `/api/tags` endpoint as a fallback.

### Launcher options

```powershell
.\muse-gateway.ps1 resume <session-id>
.\muse-gateway.ps1 exec "Summarize this repository" --model local/<model-id>
.\muse-gateway.ps1 -Port 9000
.\muse-gateway.ps1 -CheckOnly
```

```bash
bash ./muse-gateway resume <session-id>
bash ./muse-gateway exec "Summarize this repository" --model local/<model-id>
bash ./muse-gateway --port 9000
bash ./muse-gateway --check-only
muse-gateway launch --port 9000 --check-only
muse-gateway launch -h   # launcher help (--help passes through to Muse)
```

Local settings also accept `-LocalUpstream`, `-LocalApiKey`, and `-LocalModels`
in PowerShell, or `--local-upstream`, `--local-api-key`, and `--local-models`
after `launch` (the wrappers translate their named parameters to the same
flags). Prefer environment/config-file keys over command-line secrets.
Launcher options come first; other arguments pass to Muse. `-h` shows launcher
help while `--help` passes through to Muse. When using `powershell -File`,
omit the `--` separator.

The default address is `http://127.0.0.1:8787`. An occupied port is an error;
choose a different port rather than stopping an unrelated process.
`-CheckOnly` / `--check-only` checks startup and prints the catalog, then exits.
It does **not** test inference or launch an interactive Muse session.

## Routes and authentication

- **`meta/<id>`:** forwards Responses requests to `https://api.meta.ai/v1`.
  Namespace tools, reasoning, and cache fields remain intact. The routing prefix
  is removed; an explicitly configured output-token cap still applies.
  The source launchers clear Meta API-key variables so Muse uses its existing
  subscription login, whose authorization header is forwarded per request.
- **`local/<id>`:** forwards to your configured Responses endpoint. Namespace
  tools are flattened, and qualified historical tool-call names are rewritten
  to match. Uses `MUSE_SHIM_LOCAL_API_KEY`, then `MUSE_SHIM_API_KEY`, then the
  incoming authorization header.
- **`codex/<id>`:** uses Codex OAuth from its credential file or environment,
  including refresh support, and the native Codex Responses backend. It does
  not run the Codex CLI for inference.
- **`claude/<id>`:** invokes the official `claude -p` transport by default.
  Claude's built-in tools and session persistence are disabled; structured tool
  calls are returned to Muse to execute. Direct Anthropic Messages mode is
  available with `--claude-transport http` on the Python server.
- **`generic` / `http` mode:** forwards to a Responses-compatible endpoint such
  as OpenRouter. It is a separate server mode, not an additional auto-mode route.

Only send requests to upstreams you trust. Without a route-specific key, bearer
forwarding can send Muse's credentials to that upstream; the gateway logs a
one-time warning when forwarding to a non-default upstream (a custom Meta URL,
or a non-loopback local URL). A custom `MUSE_SHIM_META_UPSTREAM` also receives
native Meta payloads and credentials; putting LiteLLM there does not guarantee
it supports Meta's wire format.

## Advanced server use

Two interfaces share one launcher implementation (`launch_main` in
`muse_shim.py`):

- `muse-gateway.ps1` / `bash ./muse-gateway`: wrappers that forward to
  `python muse_shim.py launch --env-file <checkout>/.env`.
- Installed `muse-gateway launch`: the same flow without the wrappers.

`python -m muse_shim` without `launch` is the low-level server CLI; it does not
load `.env`. Server options may also go before the `launch` token
(`muse-gateway --verbose launch ...`) for host, timeout, and logging settings;
`launch` always serves the unified auto catalog.

For an explicit two-terminal setup, start the server with configuration already
exported in its environment:

```bash
python -m muse_shim auto
```

Then run Muse in the project you want to work on, with `META_API_KEY` unset for
subscription authentication:

```bash
muse --provider meta --base-url http://127.0.0.1:8787
```

For Meta API-key billing instead, configure `MUSE_SHIM_META_API_KEY` on the server
and the appropriate `META_API_KEY` for Muse; use this manual setup rather than
the subscription launcher.

The inherited `python -m muse_shim run` command sets a placeholder Muse key.
It is suitable for routes that supply their own credentials, but **not** for
forwarding Muse subscription authentication. Use `launch` for that.

### Server configuration

Run `python -m muse_shim --help` for all flags. Common environment settings:

- `MUSE_SHIM_PROVIDER`: `auto` (default), `meta`, `local`, `codex`, `claude`,
  `generic`, or `http` (alias of `generic`).
- `MUSE_SHIM_HOST`, `MUSE_SHIM_PORT`: default `127.0.0.1:8787`.
- `MUSE_SHIM_META_UPSTREAM`, `MUSE_SHIM_META_API_KEY`, `MUSE_SHIM_META_MODELS`:
  Meta route settings. With a static API key, the catalog is discovered live
  from `{upstream}/models`; otherwise (including subscription forwarding) the
  bundled list is used. An explicit `MUSE_SHIM_META_MODELS` curates names while
  discovery enriches their limits.
- `MUSE_SHIM_LOCAL_UPSTREAM`, `MUSE_SHIM_LOCAL_API_KEY`, `MUSE_SHIM_LOCAL_MODELS`:
  local route settings.
- `MUSE_SHIM_UPSTREAM`, `MUSE_SHIM_API_KEY`: generic route settings; the generic
  upstream defaults to `https://openrouter.ai/api/v1`.
- `MUSE_SHIM_CODEX_MODELS`, `MUSE_SHIM_CLAUDE_MODELS`: fallback/curated model IDs.
- `MUSE_SHIM_NO_MODEL_DISCOVERY=1`: skip catalog requests and use configured IDs.
- `MUSE_SHIM_PORTABLE=1`: remove reasoning and cache fields on non-Meta routes.
- `MUSE_SHIM_MAX_OUTPUT_TOKENS`: cap requested output tokens.
- `MUSE_SHIM_CLAUDE_TRANSPORT=cli|http`: select the Claude transport.
- `MUSE_SHIM_CLAUDE_CLI`: Claude executable; defaults to `claude`.
- `MUSE_SHIM_CODEX_AUTH`, `CODEX_HOME`: override Codex credential discovery.
- `MUSE_SHIM_TIMEOUT`: upstream timeout in seconds.
- `MUSE_SHIM_LOG_FILE`: diagnostic log path; defaults to `muse-gateway.log` in
  the working directory. Also available as `--log-file` or PowerShell `-LogFile`.
- `MUSE_SHIM_VERBOSE=1`: include HTTP access events in the diagnostic file.

`--model` on the Python server pins a model only in provider-specific mode.
In auto mode, select a model in Muse instead. The server's `--upstream` option
selects the endpoint in single-provider modes; use `--meta-upstream` and
`--local-upstream` for their routes in auto mode.

### Diagnostics

Every server writes a rotating diagnostic file (5 MiB per file, two backups).
The log directory must exist and be writable. Use separate `--log-file` paths
when running multiple gateway processes. Startup fails if the log cannot be
opened; the gateway does not silently claim that logging is enabled.

Errors and request metadata go to this file rather than interrupting the Muse
terminal. Records contain a request ID, model, recognized reasoning effort,
history/tool item counts, HTTP status, and known error categories. They do not
contain prompts, tool arguments/results, response text, authorization headers,
raw upstream error bodies, or response IDs. Error responses include the request
ID for correlation. `request_finished` means forwarding ended, not that the
model completed the task. Startup and credential-forwarding warnings still
appear in the terminal. Review even redacted logs before sharing them.

For PowerShell, follow the default log with `Get-Content .\muse-gateway.log -Wait`.
The log and rotated backups are ignored by Git. Log files are created
owner-only (0600) where the platform supports it. Credentialed upstreams
never follow a cross-origin redirect; such a redirect fails the request
instead of forwarding authorization elsewhere.

```bash
python -m muse_shim doctor --doctor-provider local
python -m muse_shim doctor --doctor-provider codex
python -m muse_shim doctor --doctor-provider claude
```

These commands read the current environment, not `.env`. Doctor checks
configuration/credential availability, not a completed inference request.
Explicit local model IDs are accepted without probing inference. When local
discovery is required but returns no usable models, doctor reports not ready
and exits nonzero. Bare `doctor` checks the default routes and configured local
route; Meta without a static key is reported not ready even when subscription
forwarding will work once Muse sends a request.

HTTP endpoints:

- `GET /health`: server liveness, not upstream readiness; identifies the service
  as `muse-gateway`. The launcher also recognizes older gateways when reusing a listener.
- `GET /muse-code/models` (also `/v1/muse-code/models`): Muse model catalog.
- `POST /responses` (also `/v1/responses`): inference.

## Compatibility and limitations

- Muse Code updates can change the expected protocol. This is community tooling,
  not a supported extension contract or a guarantee of provider feature parity.
- Meta entries are discovered live only when a static API key is configured;
  subscription setups use a bundled model list, which may not include every
  account-visible Muse model. Bundled Meta output limits are fallback values,
  not confirmed provider limits, and are labeled accordingly in the menu.
- The menu only lists providers this system can serve: Codex rows require
  stored credentials and Claude rows require the CLI (or HTTP credentials),
  matching what doctor reports. Meta is always listed (subscription forwarding
  cannot be checked statically) and local stays opt-in.
- Catalog discovery is cached for one hour. Failed or empty discovery retries
  after a minute instead, so a fresh login or recovered upstream appears
  without a restart. Restart the gateway to refresh immediately, or fetch the
  catalog with `Cache-Control: no-cache` to recall discovery on demand (for
  example after switching models). Meta subscription
  setups reuse the catalog request's own bearer for the default Meta upstream,
  mirroring inference forwarding; any failure keeps the bundled list.
  Discovered Codex models with missing limits and local/Meta
  fallbacks are labeled unverified. Native bundled fallback labels are legacy
  defaults, not live-verified limits. Codex reasoning choices
  come from model-specific discovery; client-only orchestration levels are not
  advertised as wire efforts. Meta rows advertise the levels Muse itself
  documents (`none` through `ultra`, default `high`); Muse's /effort options
  come from the same per-model variant lists. A menu entry does not verify
  tool quality or end-to-end inference compatibility.
- Local tool calls must arrive as structured Responses function-call events.
  The gateway does not convert XML-like tool syntax embedded in ordinary text.
- Codex runs stateless (`store=False`), so `previous_response_id` can never
  resolve upstream. When a turn carries complete input, the gateway drops the
  unresolvable reference, proceeds with the full turn, and logs
  `previous_response_id_dropped`. Only a turn with no input at all is
  rejected, since there is nothing to forward.
  String input is wrapped in the canonical user-message list form, which the
  backend requires.
  Text system messages are folded into `instructions`; nontext system content
  is rejected rather than dropped. Reasoning efforts are reconciled against
  model-specific discovery: known mislabels (`minimal`→`low`, `ultra`→`max`)
  are corrected with a log event, anything else unsupported is rejected with
  the supported list, and models without discovered effort evidence pass
  through untouched.
- Claude CLI output is returned after each subprocess completes, not token by
  token. This transport advertises text-only input. Claude mode does not support
  `previous_response_id` state restoration.
- Responses compatibility alone does not establish reliable multi-turn tool
  use. Unit tests use local fakes; test live providers with your own accounts.

## Security and project information

Keep the listener on loopback. The gateway has no client authentication or
multi-tenant isolation. `--allow-non-loopback` requires your own access controls.
OAuth credentials may be refreshed on disk. Do not upload auth files, `.env`,
private work notes, or unreviewed logs. See [SECURITY.md](SECURITY.md).

Contributions and local test instructions: [CONTRIBUTING.md](CONTRIBUTING.md).
Participation guidelines: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
License: [MIT](LICENSE), retaining the original muse-shim copyright notice.

Python module names, the legacy CLI alias, environment variables, and selected
protocol identifiers retain `muse_shim` / `muse-shim` for compatibility. The
project and user-facing product name are **muse-gateway**.
