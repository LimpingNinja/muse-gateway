# muse-gateway

Bring the models you already use into Muse Code. muse-gateway puts Muse Spark,
Codex, Claude, Devin, Kiro via Kiro Gateway, and local LLMs in the same workflow,
with model selection through Muse's `/model` menu.

You keep Muse's conversation, workspace, permissions, and tools. The gateway
runs alongside it on your machine, routes requests to the provider you choose,
and translates between APIs where needed. Muse still reads files, runs commands,
and executes tool calls; the provider supplies the model responses.

Use your existing provider logins or configure your own endpoint. There is no
gateway account to create, and your provider's access rules, usage limits, and
billing still apply.

This is an unofficial community project, forked from
[luckeyfaraday/muse-shim](https://github.com/luckeyfaraday/muse-shim). It is not
affiliated with Meta, OpenAI, Anthropic, Cognition, or AWS.

[Quick start](#quick-start) · [Providers](#provider-support) ·
[Configuration](#configuration-and-launch-options) · [Troubleshooting](#troubleshooting)

## Requirements

- **Python 3.10 or newer**, with pip. Install and run the gateway with the same
  Python environment. The package installs its one runtime dependency,
  `protobuf>=7.35.1`, for you; you do not need a protobuf compiler.
- **Muse Code**, available as `muse` in your terminal and signed in with
  `muse login` for the usual subscription-based launch.
- **Access to the providers you want to use.** You do not need to install every
  provider CLI. See [provider support](#provider-support) for each one's setup.
  Kiro models use a separately installed Kiro Gateway service; it is not a
  Python dependency of this project.

The Python entry point is shared across Windows, macOS, and Linux/WSL.
PowerShell and Bash wrappers are also included. A virtual environment is
optional; if you use one, activate it before both installation and launch.

## Quick start

### 1. Install the gateway

With Git:

```bash
git clone https://github.com/LimpingNinja/muse-gateway.git
cd muse-gateway
python -m pip install .
```

If you downloaded a ZIP instead, extract it, open a terminal in the extracted
directory, and run `python -m pip install .`. After pulling or downloading an
update, run the install command again to update the installed copy.

Use `python3` instead of `python` if that is your interpreter's name. On
Windows, `py -3` is another option.

### 2. Open Muse through the gateway

Sign in to Muse if you have not already:

```bash
muse login
```

Open a terminal in **the project you want to work on**, then run:

```bash
python -m muse_gateway launch
```

The installed `muse-gateway launch` command does the same thing. The module
form above also works when your Python scripts directory is not on `PATH`.

The launcher starts a gateway at `http://127.0.0.1:8787`, opens Muse, and stops
the gateway it started when Muse exits. If a compatible gateway is already
running there, it reuses it and leaves it running afterward. An unrelated or
incompatible listener is an error; use `--port 9000` rather than stopping a
process you do not recognise.

With Kiro configured, launch requires a free port rather than reusing an
existing gateway. This keeps it from attaching to a server that has different
proxy settings. Pick another port if one is already running.

### 3. Choose a model

Open `/model` in Muse and choose a provider/model entry. Start with a short
prompt, then try a file read or another tool call.

Meta is listed for subscription forwarding. Codex and Devin entries need local
credentials; Claude needs its CLI or HTTP credentials. Local entries appear
when you configure a local endpoint or model list. Kiro entries need a Kiro
Gateway proxy key and discovery or an explicit model list. These checks make
the menu useful, but an entry is not a guarantee of account access or a
successful turn.

Warnings about an unconfigured provider do not mean you must install it. In
particular, the launcher warns when no local API key is set and skips that
route's preflight; you can still use your other providers.

## Provider support

### Muse Spark

The normal launcher uses your existing `muse login` session. No Meta API key is
needed: Muse sends its authorization header to the gateway, which forwards it
to `https://api.meta.ai/v1` for `meta/<model-id>` requests.

Meta's namespace tools, reasoning, and cache fields are preserved. The routing
prefix is removed, and an explicitly configured output-token cap still applies.
The launcher deliberately clears static Meta API-key settings so that this path
uses subscription authentication.

The catalog can use a static key or the bearer on a Muse catalog request to
discover Meta models. Subscription discovery reuses that bearer only with the
default Meta upstream. If discovery is unavailable, the bundled list remains.
For API-key billing instead, use the [manual setup](#manual-server-setup).
`MUSE_GATEWAY_META_UPSTREAM` overrides the endpoint, but any replacement must
accept Meta's native payloads and be trusted with its authorization.

### Claude

Install [Claude Code](https://code.claude.com/docs/en/overview), sign in through
its normal login flow, and make sure `claude` is available in the same terminal
as the gateway. Then launch the gateway and select a `claude/<model-id>` entry.
You can set `MUSE_GATEWAY_CLAUDE_CLI` if the executable has a different path.

By default the gateway invokes the official `claude -p` command. Claude's own
tools and session persistence are disabled for these calls; requested tool
calls come back to Muse to execute. Output arrives after each subprocess
finishes, not token by token, and this transport advertises text-only input.

For direct Anthropic Messages requests instead, set
`MUSE_GATEWAY_CLAUDE_TRANSPORT=http`. That mode reads HTTP credentials from
supported Claude credential storage or the environment, including
`ANTHROPIC_API_KEY` for API-key billing. Set
`MUSE_GATEWAY_CLAUDE_CREDENTIALS` to use an explicit credential file. HTTP mode
does not require the CLI for inference, and supports credential refresh where
the stored login provides it.

Claude does not support restoring turns with `previous_response_id`; send the
conversation history inline. Native signed-thinking replay is not supported.

### Codex

Sign in with the Codex CLI:

```bash
codex login
```

The gateway reads the existing credentials and lists `codex/<model-id>` entries.
It uses the native Codex Responses backend directly, with OAuth refresh
support; it does not start the Codex CLI for inference. `CODEX_HOME` or
`MUSE_GATEWAY_CODEX_AUTH` can point it at another credential location.

Codex requests are stateless (`store=False`). When complete history is present,
the gateway drops an unresolvable `previous_response_id` and sends that history
instead. It also normalises Muse's replay item IDs without changing the IDs
that pair tool calls with their results, and folds text system messages into
`instructions`.

If Codex explicitly rejects encrypted reasoning retained from another provider,
the gateway retries once with its visible summary in place of that opaque
state. Conversation and tool history stay intact; the recovery is logged as
`codex_reasoning_demoted`. Known reasoning-effort mismatches are reconciled
against model-specific discovery, rather than assuming every model accepts
the same effort names.

### Devin

Sign in with the [Devin CLI](https://cli.devin.ai/):

```bash
devin auth login
```

The gateway speaks Devin's Connect/protobuf API directly. The CLI is a source
of credentials, not the process used for inference. Credential lookup uses:

1. `MUSE_GATEWAY_DEVIN_API_KEY` (or `DEVIN_API_KEY`) for an explicit session token.
2. Your existing Devin CLI login.
3. Your OMP login, normally in `~/.omp/agent/agent.db`.

Use `MUSE_GATEWAY_DEVIN_AUTH_DB` for another OMP database and
`MUSE_GATEWAY_DEVIN_CREDENTIAL_ID` to select a particular stored credential.
An explicit credential ID targets the store instead of the CLI login.
There is no refresh grant in this integration; an expired or rejected login
means signing in again.

Select a `devin/<model-id>` entry. Discovery uses the active credential and
includes directly servable models, routers, and supported effort lanes. Fusion
pairings are excluded because their sidekick half cannot run in this gateway.
Without discovery, the small bundled SWE model list is a fallback, not an
account-verified roster.

Text streams incrementally. Function calls are released once their names and
arguments are final, so the streamed and completed arguments agree. Developer
messages retain their position in history. Native signed-reasoning replay is
not implemented; incoming signatures are not forwarded as provider-owned state.

### Kiro Gateway

Follow the [Kiro Gateway README](https://github.com/jwadow/kiro-gateway) for
installation, Kiro sign-in, and service configuration. It runs separately;
muse-gateway connects to its Messages API, not your Kiro login files. From
its Python checkout, the basic start command after setup is:

```bash
python main.py
```

#### Connect it to Muse

Add these settings to muse-gateway's private env file. Use the same value as
Kiro Gateway's `PROXY_API_KEY`, not a Kiro refresh token, Muse key, or Claude
credential:

```dotenv
MUSE_GATEWAY_KIRO_UPSTREAM=http://127.0.0.1:8000/v1
MUSE_GATEWAY_KIRO_API_KEY=replace-with-your-kiro-gateway-proxy-key
```

Then launch from the project you want Muse to work in:

```bash
python -m muse_gateway launch --env-file /path/to/private.env --port 8791
```

Keep Kiro Gateway running; our launcher does not start or stop it. The
muse-gateway port must be free: Kiro-configured launches do not attach to an
existing gateway, whose proxy settings might differ. For local use, bind the
separate Kiro Gateway service to loopback too (`SERVER_HOST=127.0.0.1` in its
configuration). Remote Kiro endpoints require verified HTTPS.

#### Models

Choose a `kiro/<model-id>` entry in `/model`. The prefix is removed before
forwarding; `kiro:` also works. In auto mode, an unprefixed Claude name still
uses the native Claude route.

Models come from the service's authenticated `/v1/models` endpoint. Optional
`MUSE_GATEWAY_KIRO_MODELS` curates that list; `--kiro-models` takes precedence.
An explicit list stays restricted after an account upgrade. You can supply
current IDs if the service's catalog is stale, or when discovery is disabled.
Listed IDs are not proof of account access or tested capabilities. Without a
proxy key, there are no automatic Kiro entries or discovery calls.

#### Behavior and caveats

- **Text and tools:** text streams, and Muse executes returned function calls.
  Full conversation and tool-result history must travel inline. Redundant
  `previous_response_id` values can be dropped; ID-only chaining and unresolved
  references are rejected. Media, foreign encrypted reasoning, and signed-thinking
  replay are unsupported; visible reasoning summaries can be retained.
- **Tools disabled after earlier calls:** stock Kiro Gateway converts that
  structured history to plain text when no current tools are offered, including
  with `tool_choice: "none"`. We reject this combination before inference,
  without changing history or re-enabling tools. `none` still works in
  conversations without tool-call history.
- **Unsupported controls:** sampling/stop controls, required or named tool
  choices, and `parallel_tool_calls: false` are rejected before inference. Use
  another route if single-call turns are required; allowing multiple calls is
  not a substitute for that policy.
- **Reasoning and recovery:** the route advertises no reasoning variants and
  sends `thinking: {"type": "disabled"}`. In
  [Kiro Gateway's own configuration](https://github.com/jwadow/kiro-gateway/blob/main/.env.example),
  we recommend `FAKE_REASONING=false`, `WEB_SEARCH_ENABLED=false`,
  `TRUNCATION_RECOVERY=false`, and `AUTO_TRIM_PAYLOAD=false` to avoid injected
  thinking/search tools, recovery turns, and history trimming. These are not
  muse-gateway settings; per-request thinking does not override every global
  service setting. Our adapter adds no history trimming or inference retries.
- **Budgets:** `MUSE_GATEWAY_KIRO_CONTEXT_LIMIT` (65,536) and
  `MUSE_GATEWAY_KIRO_OUTPUT_LIMIT` (8,192) are configurable, **unverified**
  planning hints, not measured capacity or backend-enforced caps. The separate
  service also has a payload byte-size limit.

For errors, retry hints, and cancellation limits, see [Logs and doctor](#logs-and-doctor).
Muse may label a Kiro 401 as a Meta-key error; check the gateway's route and
upstream evidence before changing credentials. Offline checks cover the
translated protocol, not live entitlement, TUI model selection, or resume;
check those with your intended account, service, model, and Muse version.

### Local LLMs and other Responses endpoints

Use the `local/` route for a local model server, a LiteLLM proxy, or another
endpoint you operate. It needs **OpenAI Responses** support: `POST /responses`,
streaming SSE, and structured function-call events for tools. A server that
only implements Chat Completions needs an adapter. Discovering an Ollama model
does not by itself establish Responses or tool compatibility.

Copy [.env.example](.env.example) to a private `.env`, uncomment the local
settings, and edit them for your endpoint:

```dotenv
MUSE_GATEWAY_LOCAL_UPSTREAM=http://127.0.0.1:4000/v1
MUSE_GATEWAY_LOCAL_API_KEY=replace-with-your-upstream-key
# Optional: restrict the menu to these upstream IDs.
# MUSE_GATEWAY_LOCAL_MODELS=my-coder,my-reasoner
```

Run from your working project, giving the config file's location explicitly:

```bash
python -m muse_gateway launch --env-file /path/to/muse-gateway/.env
```

The gateway discovers IDs from the upstream's `GET /models`. An explicit model
list curates those names. Discovery reads advertised context/output limits and
image-input capabilities; missing local limits fall back to 128,000 context
and 32,768 output tokens, labelled **limits unverified**. Without image
evidence, entries advertise text input only. For an Ollama URL ending in `/v1`,
`/api/tags` is a discovery fallback.

Local requests flatten namespace tools and rewrite qualified historical call
names to match. Tools must arrive as structured function calls; XML-like tool
syntax in ordinary response text is not converted.

A local API key enables launcher preflight. A keyless server can still be
configured, but preflight is skipped and the gateway may forward Muse's bearer
when no dedicated route key is set. Use a dedicated upstream key where possible;
see [security](#security).

For a single Responses endpoint without the combined menu, the separate
`generic` (also `http`) server mode is available. Its default upstream is
OpenRouter; it is not another route in auto mode.

## Configuration and launch options

Settings use `MUSE_GATEWAY_*`. The older `MUSE_SHIM_*` names remain fallbacks;
the newer spelling wins when both are set. Flags override environment values,
and the launcher loads nonempty `.env` values only where the same environment
variable is not already set. Use the newer names consistently to avoid mixing
aliases across configuration sources.

Use `--env-file PATH` after `launch` to choose a file. Otherwise it looks beside
`muse_gateway.py`, then in the current directory. The source wrappers explicitly
use `.env` beside the scripts. Installing the package does not copy your source
checkout's `.env`, and no launcher creates one automatically. Environment
changes do not reconfigure an already running gateway that a launcher reuses.

Common settings (the provider sections above explain authentication):

| Setting | Purpose |
| --- | --- |
| `MUSE_GATEWAY_HOST`, `MUSE_GATEWAY_PORT` | Listener; defaults to `127.0.0.1:8787` |
| `MUSE_GATEWAY_PROVIDER` | Low-level server mode; `launch` always uses `auto` |
| `MUSE_GATEWAY_META_MODELS` | Curate Meta model IDs; discovery enriches their limits |
| `MUSE_GATEWAY_CODEX_MODELS`, `MUSE_GATEWAY_CLAUDE_MODELS` | Set fallback model IDs; discovery can add models |
| `MUSE_GATEWAY_DEVIN_MODELS` | Restrict Devin discovery to selected IDs |
| `MUSE_GATEWAY_KIRO_UPSTREAM`, `MUSE_GATEWAY_KIRO_API_KEY`, `MUSE_GATEWAY_KIRO_MODELS` | Separate Kiro Gateway endpoint, proxy key, and optional curated IDs |
| `MUSE_GATEWAY_KIRO_CONTEXT_LIMIT`, `MUSE_GATEWAY_KIRO_OUTPUT_LIMIT` | Unverified positive Muse planning budgets; not backend-enforced capacities |
| `MUSE_GATEWAY_LOCAL_UPSTREAM`, `MUSE_GATEWAY_LOCAL_API_KEY`, `MUSE_GATEWAY_LOCAL_MODELS` | Local endpoint, key, and model list |
| `MUSE_GATEWAY_UPSTREAM`, `MUSE_GATEWAY_API_KEY` | Generic endpoint/key; the generic key is also a fallback for local requests |
| `MUSE_GATEWAY_NO_MODEL_DISCOVERY=1` | Disable discovery and use configured/bundled models |
| `MUSE_GATEWAY_MAX_OUTPUT_TOKENS` | Cap requested output tokens; backend enforcement depends on the route (not enforced by Kiro Gateway) |
| `MUSE_GATEWAY_PORTABLE=1` | Remove reasoning/cache fields on non-Meta routes |
| `MUSE_GATEWAY_TIMEOUT` | Upstream timeout in seconds; default 600 |
| `MUSE_GATEWAY_LOG_FILE`, `MUSE_GATEWAY_VERBOSE=1` | Log path and extra HTTP access events |
| `MUSE_GATEWAY_CAPTURE_DIR` | Opt-in private wire captures; contains conversation content |

Use the installed command or the matching Python module form:

```bash
muse-gateway launch --port 9000
muse-gateway launch --check-only
muse-gateway launch resume <session-id>
muse-gateway launch -h
```

`--check-only` checks startup and the catalog, then exits without launching Muse
or testing inference. Launcher options go first; remaining arguments go to Muse.
`-h` shows launcher help, while `--help` passes through to Muse.

The source wrappers use the same launcher and preserve your working directory:

```powershell
# Windows PowerShell, from the checkout:
.\muse-gateway.ps1 -Port 9000
.\muse-gateway.ps1 -CheckOnly
.\muse-gateway.ps1 resume <session-id>
```

```bash
# macOS / Linux / WSL, from the checkout:
bash ./muse-gateway --port 9000
bash ./muse-gateway --check-only
bash ./muse-gateway resume <session-id>
```

You can invoke those scripts by absolute path from another project. The
PowerShell wrapper also accepts `-EnvFile`, `-MuseBin`, `-LogFile`,
`-LocalUpstream`, `-LocalApiKey`, and `-LocalModels`. Prefer config/environment
settings to command-line secrets. Kiro options are `-KiroUpstream`, `-KiroApiKey`,
`-KiroModels`, `-KiroContextLimit`, and `-KiroOutputLimit`, corresponding to the
same `--kiro-*` launch flags. With `powershell -File`, omit the `--`
separator and pass Muse arguments directly.

### Manual server setup

For separate server and Muse terminals, first export your configuration in the
server's environment, then start:

```bash
python -m muse_gateway auto
```

The low-level server does **not** load `.env`. In your working project, run
Muse with `META_API_KEY` unset for subscription authentication:

```bash
muse --provider meta --base-url http://127.0.0.1:8787
```

For Meta API-key billing, set `MUSE_GATEWAY_META_API_KEY` on the server and the
appropriate `META_API_KEY` for Muse instead of using the subscription launcher.

`python -m muse_gateway --help` lists server flags and modes, including `devin`
and `kiro`.
Server options go **before** the `launch` token when combining the interfaces,
for example `python -m muse_gateway --timeout 120 launch`. `launch` always
serves the combined auto catalog; `--model` pins a model only in a
provider-specific server mode, not auto mode. Use per-route upstream settings
in auto mode, not the generic `--upstream` setting.

The inherited `python -m muse_gateway run` command sets a placeholder Muse key.
It is suitable only for routes supplying their own credentials, not forwarding
a Muse subscription login. Use `launch` for the usual workflow.

## Troubleshooting

### Checking a failed turn

An HTTP 200 does not guarantee a complete, readable model response. Check the
stream's terminal event and Muse's outcome, including any tool continuation.
Health and catalog checks confirm the listener and model listing, not inference.

- **Missing provider:** check its login or executable, then refresh the catalog.
  Successful discovery is cached for an hour; failed/empty discovery retries
  after a minute. Restart to refresh immediately, or fetch the catalog with
  `Cache-Control: no-cache`.
- **Local startup fails:** verify the endpoint/key and its model listing, or
  configure an explicit local model list. Explicit IDs bypass discovery checks,
  not inference requirements.
- **Devin authentication fails:** sign in again; the gateway does not refresh
  that login. A token rejected during this gateway process is quarantined in
  memory, so restart after replacing a bad token if needed.
- **Kiro is missing or fails authentication:** check the separate service and
  its proxy key. Muse and native Claude credentials are never a fallback for
  this route. Successful catalog discovery does not establish model access.
  Muse can label a Kiro 401 as a rejected Meta key because it uses the Meta
  client interface; check the gateway's route and upstream error evidence
  before changing credentials.
- **Kiro launch refuses an occupied port:** use a free `--port`; the launcher
  will not ignore your Kiro configuration by attaching to another gateway.
- **Muse reports malformed data or retries:** match the Muse session error to
  the gateway request ID and capture the failing stream if needed. Use the
  terminal event and decoder error to distinguish completion from a partial
  or unreadable stream.
- **Headless `muse exec` fails before making a request:** Muse can reject
  unfamiliar model IDs while resolving compaction budgets. Use interactive
  `/model` selection to check the route; do not assume the gateway received it.

### Logs and doctor

The default diagnostic file is `muse-gateway.log` in the gateway's working
directory. It rotates at 5 MiB with two backups. Its directory must exist and
be writable; startup fails if logging cannot be opened. Use distinct paths for
separate gateway processes. If a launcher reuses a server, inspect that
server's log, not a newly chosen path in the attaching terminal.

Records contain request IDs, routes/models, recognised effort, item counts,
HTTP status, and error categories. Ordinary logs exclude prompts, tool
arguments/results, response text, authorization headers, and raw error bodies.
Review even redacted logs before sharing. On PowerShell:

```powershell
Get-Content .\muse-gateway.log -Wait
```

Kiro errors also preserve bounded, validated error codes and correlation IDs
in `error.metadata`, separately from the gateway request ID. HTTP rejections
keep their upstream status and valid `Retry-After` headers; errors after
streaming begins retain the partial answer and end with `response.failed`.
Unknown or conflicting evidence stays explicit. These fields do not add
retries, judge answer quality, or recover details already discarded by Kiro
Gateway. Muse may not display every field; use the matching gateway record.

Client disconnects are detected when forwarding encounters an error. If the
upstream stream is idle, its connection can remain open until more data
arrives or the configured read timeout expires (`--timeout`, 600 seconds by
default). Do not assume immediate upstream cancellation or stopped billing
when closing the client. A forced-disconnect check does not establish how
Muse's Escape-key cancellation behaves.

Doctor reads the environment, not `.env`, and checks configuration/credential
availability rather than running an inference request:

```bash
python -m muse_gateway doctor --doctor-provider codex
python -m muse_gateway doctor --doctor-provider claude
python -m muse_gateway doctor --doctor-provider devin
python -m muse_gateway doctor --doctor-provider kiro
python -m muse_gateway doctor --doctor-provider local
```

Bare `doctor` checks the default routes and configured local/Kiro routes. It can
report Meta as not ready without a static key even when subscription forwarding
will work once Muse sends its bearer. For local models, keyless setups or
unusable discovery can also make doctor report not ready; that is a preflight
result, not an attempted model turn.

Kiro preflight distinguishes a configured proxy key from an authenticated
catalog response. Neither means inference has been validated. It does not
sign in to Kiro or refresh the external service's credentials.

HTTP endpoints:

- `GET /health`: listener health, not upstream readiness. Includes process and
  source identity so you can check which gateway is running.
- `GET /muse-code/models`: Muse's model catalog.
- `POST /responses`: inference.

The catalog and inference endpoints also accept a `/v1` prefix.

### Private failure captures

Capture is off by default. For a controlled diagnosis, use a separate port and
a private destination outside the repository:

```powershell
python -m muse_gateway --capture-dir "$env:TEMP\muse-gateway-captures" launch --port 8791
```

`MUSE_GATEWAY_CAPTURE_DIR` enables the same mode. `/health` identifies the
capture session as well as the process/source. Each gateway request gets its
own directory containing request transforms, outgoing bytes, upstream headers
and error bodies, downstream HTTP/SSE bytes, and a monotonic read/frame/write
timeline. Devin protobufs are also saved in readable form. Known credentials
and identifying truncated prefixes are scrubbed before writing; this can also
remove matching fragments from conversation data. Saved Devin frames are
scrubbed, potentially re-encoded copies. Frames that cannot safely be scrubbed
are marked omitted, and hashes of original bytes are identified as such.

**Captures contain conversation content.** Owner-only permissions are applied,
including a protected Windows ACL, and the gateway does not upload these files.
Each request has a 64 MiB budget. Check `manifest.json` for truncation, omitted
data, capture errors, byte counts, and terminal events. Even a complete capture
is transport evidence: correlate it with Muse's completion or decoder error.
See [SECURITY.md](SECURITY.md) before sharing any files.

## Compatibility

Muse's protocol can change between releases. Re-run consumer compatibility
checks after updating Muse; they are specific to the executable being tested.
[CONTRIBUTING.md](CONTRIBUTING.md#what-a-green-result-means) explains the checks
and which account/TUI workflows still need verification.

Advertised model limits are discovery data or fallbacks, not measurements.
Local/Meta, Kiro planning budgets, and undiscovered Devin limits are marked
unverified; legacy native fallback labels are not proof of live-verified limits
either. Model quality,
tool reliability, and features such as signed reasoning remain provider-specific.
Switching providers does not make their opaque reasoning state interchangeable.

## Security

Keep the listener on loopback. There is no client authentication or multi-tenant
isolation; `--allow-non-loopback` requires your own access controls.

Only configure upstreams you trust. Local requests use the local key, then the
generic gateway key, then incoming authorization. Meta uses its static route
key or incoming authorization. Without a dedicated key, your configured server
may receive Muse's subscription bearer. A custom Meta URL also receives native
Meta payloads; an OpenAI-compatible label alone does not establish support.

The Kiro route requires its own proxy key and never forwards Muse's bearer or
native Claude credentials. Plain HTTP is allowed only for loopback Kiro
endpoints; remote Kiro endpoints require verified HTTPS. Credentialed Kiro
requests reject redirects rather than sending the key to a different URL.

Native Codex/Claude OAuth destinations are allowlisted, and credentialed
requests do not follow cross-origin redirects. Do not use
`--allow-oauth-host-override` casually. Prefer TLS for non-loopback upstreams.
OAuth refresh may update credential files on disk. Keep `.env`, credentials,
logs, and captures private; the gateway cannot remove every secret a user or
model might include in conversation content. See [SECURITY.md](SECURITY.md).

## Contributing and attribution

Development and test instructions: [CONTRIBUTING.md](CONTRIBUTING.md).
Community guidelines: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Licensed under [MIT](LICENSE), retaining muse-shim's original copyright and
attribution. Devin protocol adaptations also retain the upstream notices in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

Kiro Gateway remains a separate service under its own license. Its source is
not bundled or copied into this project.

The `muse_shim` module, `muse-shim` console command and wrapper, `MUSE_SHIM_*`
settings, older health-service identifiers, and `shim_error` wire type remain
for compatibility. The project name is **muse-gateway**.
