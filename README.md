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
  `protobuf>=7.35.1,<8`, for you; you do not need a protobuf compiler.
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
when you configure a local endpoint with complete per-model capacities. Kiro
entries need a Kiro Gateway proxy key, discovery, and exact model metadata. These checks make
the menu useful, but an entry is not a guarantee of account access or a
successful turn.

Warnings about an unconfigured provider do not mean you must install it. In
particular, an unset local upstream disables local routing. Local keys are
optional; an unset local key sends no Authorization header.

## Provider support

### Muse Spark

The normal launcher uses your existing `muse login` session. No Meta API key is
needed: Muse sends its authorization header to the gateway, which forwards it
to `https://api.meta.ai/v1` for `meta/<model-id>` requests.

Meta's namespace tools, reasoning, and cache fields are preserved. The routing
prefix is removed. Per-request `max_output_tokens` is bounded by known model
output metadata on supported routes; there is no gateway-wide output cap.
The subscription launcher rejects `META_API_KEY` or `--meta-api-key` rather
than silently changing billing or clearing your configuration.

The catalog can use a static key or the bearer on a Muse catalog request to
discover Meta models. Subscription discovery reuses that bearer only with the
default Meta upstream. If discovery is unavailable, the bundled list remains.
For API-key billing instead, use the [manual setup](#manual-server-setup).
`--meta-upstream` overrides the endpoint, but any replacement must
accept Meta's native payloads and be trusted with its authorization.

### Claude

Install [Claude Code](https://code.claude.com/docs/en/overview), sign in through
its normal login flow, and make sure `claude` is available in the same terminal
as the gateway. Then launch the gateway and select a `claude/<model-id>` entry.
Use `--claude-cli` if the executable has a different path.

By default the gateway invokes the official `claude -p` command. Claude's own
tools and session persistence are disabled for these calls; requested tool
calls come back to Muse to execute. Output arrives after each subprocess
finishes, not token by token, and this transport advertises text-only input
(image input is rejected with `claude_cli_images_unsupported`). A per-request
output cap below the model limit is rejected with
`claude_cli_output_limit_unsupported`, because `claude -p` cannot enforce it.
Time to first token therefore equals the whole generation time, and
cancelling waits for the subprocess (bounded by `--timeout`). The gateway does
not fake token pacing. Use `--claude-transport http` when you need streaming.

For direct Anthropic Messages requests instead, set
`MUSE_GATEWAY_CLAUDE_TRANSPORT=http`. That mode reads HTTP credentials from
supported Claude credential storage or the environment, including
`CLAUDE_CODE_OAUTH_TOKEN`, `CLAUDE_CODE_REFRESH_TOKEN`, and `ANTHROPIC_API_KEY`
for API-key billing. Use `--claude-credentials` for an explicit credential file. HTTP mode
does not require the CLI for inference, and supports credential refresh where
the stored login provides it.

Claude does not support restoring turns with `previous_response_id`; send the
conversation history inline. A `previous_response_id` that is null or empty is
ignored; one sent beside complete inline history is dropped and logged as
`previous_response_id_dropped`; an ID with only a new prompt is rejected with
`claude_previous_response_unsupported`. Native signed-thinking replay is not
supported.

### Codex

Sign in with the Codex CLI:

```bash
codex login
```

The gateway reads the existing credentials and lists `codex/<model-id>` entries.
It uses the native Codex Responses backend directly, with OAuth refresh
support; it does not start the Codex CLI for inference. `CODEX_HOME` or
`--codex-auth-path` can point it at another credential location; for Claude,
`CLAUDE_CONFIG_DIR` or `CLAUDE_CODE_CREDENTIALS_PATH` (or
`--claude-credentials`) does the same. Native token
injection via environment is retired; use CLI auth storage or `--access-token`.
Codex drops the unsupported request `max_output_tokens` field; no output-cap
guarantee is claimed for that route. Reserve/review slugs stay hidden from the
menu; reserve quota retry and explicit routing remain available.

Codex requests are stateless (`store=False`). When complete history is present,
the gateway drops an unresolvable `previous_response_id` and sends that history
instead. An ID sent with only a new prompt (delta-only chaining) is rejected
with `codex_previous_response_unsupported` rather than answered without
context; Claude, Devin, and Kiro apply the same rule. It also normalises Muse's replay item IDs without changing the IDs
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

1. `--devin-api-key` or standard `DEVIN_API_KEY` for an explicit session token.
2. Your existing Devin CLI login.
3. No authentication when neither is available.

`doctor --doctor-provider devin` reports readiness without showing tokens.
Credentials are read-only. The gateway does not read credential databases,
select stored rows, or read/write a saved credential preference.
Retired database, credential-ID, and default-file options now fail with a
value-free diagnostic; remove them from flags and environment/config files.
Existing saved preferences are unused and are not modified.
There is no refresh grant in this integration; a rejected login requires
signing in again. A rejected explicit token is not silently replaced by the
CLI login.

If Devin's auth response directs chat traffic to another server
(`custom_api_server_url`), that server must use HTTPS unless it is loopback;
the gateway logs the redirect once. A `--devin-upstream http://…` on a
non-loopback host is allowed for on-premises setups but warns at startup.

Select a `devin/<model-id>` entry. Discovery uses the active credential and
includes directly servable models, routers, and supported effort lanes. Fusion
pairings are excluded because their sidekick half cannot run in this gateway.
Without discovery, the small bundled SWE model list is a fallback, not an
account-verified roster.

Text streams incrementally. Function calls are released once their names and
arguments are final, so the streamed and completed arguments agree. Because a
call is held until final, text produced after it in the same upstream turn can
stream first; the final output order is the release order. Developer messages
retain their position in history. Native signed-reasoning replay is
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

Keep Kiro Gateway running; our launcher does not start or stop it unless
you opt into autostart below. The
muse-gateway port must be free: Kiro-configured launches do not attach to an
existing gateway, whose proxy settings might differ. For local use, bind the
separate Kiro Gateway service to loopback too (`SERVER_HOST=127.0.0.1` in its
configuration). Remote Kiro endpoints require verified HTTPS.

#### Auto-start (optional)

`launch` can start the checkout for you when its loopback service is down:

```bash
python -m muse_gateway launch --kiro-autostart --kiro-server-dir ~/kiro-gateway
```

(Autostart and its checkout are CLI flags only — `--kiro-autostart` and
`--kiro-server-dir`; no environment or file setting enables them.) Only
loopback upstreams are autostarted; the checkout's `.venv` python runs
`main.py --port <upstream-port>`, its output goes to `autostart.log` in the
checkout (appended, never rotated; delete it by hand if it grows), and the
service stops when `launch` exits. The startup line reports
`service already running`, `service autostarted (pid N)`, or the failure
reason. Bare `run`/serve mode never autostarts.

#### Models

Choose a `kiro/<model-id>` entry in `/model`. The prefix is removed before
forwarding; `kiro:` also works. In auto mode, an unprefixed Claude name still
uses the native Claude route.

Models come from the service's authenticated `/v1/models` endpoint.
`MUSE_GATEWAY_MODELS` optionally filters exact IDs such as
`kiro/claude-haiku-4.5`; it cannot supply new rows or establish entitlement.
Discovery names lacking exact capacity metadata stay internally known and
directly routable, but are omitted from the numeric menu with a diagnostic.
Without a proxy key, there are no automatic Kiro entries or discovery calls.

Some upstream Kiro Gateway development revisions skip discovery for runtime
endpoints and serve a bundled static list. Two Kiro-side options are provided:

- **Native autodiscovery:** apply [kiro-model-discovery.patch](errata/kiro-model-discovery.patch)
  to the pinned clean upstream base. It restores discovery through the native
  management API without changing inference or web-search endpoints.
- **Static configuration:** copy the `FALLBACK_MODELS` declaration from
  [models.py.example](errata/models.py.example) into Kiro Gateway's existing
  `kiro/config.py`. This one-time working-catalog snapshot needs no loader or
  discovery patch. Do not replace the whole configuration file.

See [Kiro discovery errata](errata/kiro-model-discovery.md) for the base revision,
application instructions, upstream PR attribution, and limitations. Restart
Kiro after applying either option. Muse still autodiscovers from Kiro; no
model-list override is needed. The patch is separately licensed
under [Kiro's AGPL license](errata/KIRO-LICENSE), not Muse Gateway's MIT license.

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
- **Capacities:** exact registry fields preserve the approved values, with
  documented and user-approved provenance distinguished. Unknown IDs receive
  no provider-wide capacity guesses. Budget overrides and the global output
  cap were removed. Per-request output is translated to `max_tokens`, bounded
  by known output metadata; Kiro Gateway does not guarantee a hard cap.
  The separate service also has a payload byte-size limit.

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
# Optional: filter the menu by exact IDs, without inventing models.
# MUSE_GATEWAY_MODELS=local/my-coder,meta/muse-spark-1.3
```

Run from your working project, giving the config file's location explicitly:

```bash
python -m muse_gateway launch --env-file /path/to/muse-gateway/.env
```

The gateway discovers IDs, context/output capacities, and image-input
capabilities from the upstream's `GET /models`. `MUSE_GATEWAY_MODELS` filters
those entries without creating models or changing capacities. Models missing
required capacity metadata are reported and omitted from the numeric menu;
there is no provider-wide capacity guess. Without image evidence, entries
advertise text input only. For an Ollama URL ending in `/v1`, `/api/tags` can
discover names, but those names still need capacity metadata to appear.

Local requests flatten namespace tools and rewrite qualified historical call
names to match. Tools must arrive as structured function calls; XML-like tool
syntax in ordinary response text is not converted.

A configured local upstream enables preflight whether or not it needs a key.
The gateway sends only `MUSE_GATEWAY_LOCAL_API_KEY` when set. A keyless local
server receives no Authorization header; Muse's subscription credential and
other providers' keys are never borrowed. See [security](#security).

For a single Responses endpoint without the combined menu, the separate
`generic` (also `http`) server mode is available. Its default upstream is
OpenRouter; it is not another route in auto mode.

### Image input

Image parts in user, developer, or tool-result content are loaded and checked
once in the gateway, before the request is forwarded to the provider (the
model catalog may be consulted first to check image support), and forwarded
as inline base64.
Accepted forms: `input_image`/`image_url` parts whose `image_url` is a string
or `{"url": ...}`, and Anthropic-style `image` parts with a `base64` or `url`
source. The URL may be a `data:...;base64,` URI, an `https://` or `http://`
URL, or a `file:///absolute/path` reference. Malformed parts and images in
system, assistant, or instruction content are rejected (`image_invalid`); no
image is silently dropped.

- Formats: PNG, JPEG, GIF, and WebP, identified by their leading bytes. The
  declared media type, `Content-Type`, and file extension are never trusted;
  SVG, BMP, TIFF, HEIC, AVIF, and PDF are rejected.
- Size: 20 MiB per image (5 MiB on Claude HTTP), 24 MiB of decoded image data
  per request, and at most 100 images per request (`image_too_large`).
- Remote URLs: only hosts whose every DNS address is public are fetched;
  loopback, private, link-local, and other non-global targets are rejected
  (`image_url_rejected`), and each of up to 3 redirects is re-checked. No
  credentials, cookies, or proxy are used: `HTTP(S)_PROXY` settings are
  ignored, so networks that require a proxy cannot fetch image URLs. A fetch
  that fails, times out, or does not return HTTP 200 is `image_url_unreachable`
  (502). DNS resolution uses the system resolver's own timeout and is not
  bounded by the fetch budget. There is no caching: replayed history fetches
  each URL again.
  Claude HTTP no longer forwards image URLs as URL sources; the gateway sends
  the fetched bytes instead. A host with any non-public DNS record is rejected
  even if other records are public.
- Local files: off unless the server starts with `--allow-image-files`
  (CLI flag only); otherwise `file://` parts fail with `image_file_rejected`.
  Paths must be absolute, regular files with no symbolic link anywhere in the
  path. On macOS `/tmp` and `/var` are symlinks, so pass `/private/tmp/...` or
  `/private/var/...`. The file's bytes are sent to the selected provider, and
  inbound client authentication is opt-in (`--require-inbound-token`, off by
  default), so enable this only on a loopback listener you control.
- Routes: Claude CLI transport and Kiro accept text only and reject images
  (`claude_cli_images_unsupported`, `kiro_unsupported_input`). Local models
  need image evidence from their catalog; otherwise `image_input_unsupported`.
  Devin rejects images inside tool results (`image_input_unsupported`) rather
  than dropping them. A Devin request whose history is 512 KiB or more (easily
  reached with images) may treat an upstream `invalid_argument` "internal
  error" as a context overflow.
- Captures (`--capture-dir`) now contain fetched and file-loaded image bytes
  as base64.

## Configuration and launch options

Connection and diagnostic settings use `MUSE_GATEWAY_*`. Provider credentials
use their standard names (`DEVIN_API_KEY`, `META_API_KEY`, and the supported
Claude variables). Flags override the current environment, which overrides
file values; an explicitly blank environment value is preserved.

Removed settings and `MUSE_SHIM_*` names are not read as aliases. They produce
migration guidance without exposing values; conflicting old/new settings fail.
Removed gateway flags fail before they can be forwarded to Muse.

Commonly needed flags that otherwise appear only in `--help`: `--models`
(print the catalog and exit; put it first with the Bash wrapper),
`--log-file PATH` (diagnostic log location), and the per-provider
`--meta-upstream`, `--codex-upstream`, `--claude-upstream`, `--devin-upstream`,
`--kiro-upstream`, and `--local-upstream` spellings (there are no
`--*-base-url` flags).

Use `--env-file PATH` after `launch` to choose a file. Otherwise it looks beside
`muse_gateway.py` only — the current directory's `.env` is never loaded
implicitly, because the working directory is the (untrusted) project under edit.
A named `--env-file` that does not exist is an error. Symlinked env files are
followed; for the beside-module default, startup prints the resolved target.
Startup prints a `config:` line naming the file loaded (or `config: no env file`). The source
wrappers pass `.env` beside the scripts only when that file exists. Installing the package does not copy your source
checkout's `.env`, and no launcher creates one automatically. Environment
changes do not reconfigure an already running gateway that a launcher reuses.

Common settings (the provider sections above explain authentication):

| Setting | Purpose |
| --- | --- |
| `MUSE_GATEWAY_PORT` | Listener port; default 8787 on loopback |
| `MUSE_GATEWAY_TIMEOUT` | Upstream connect/headers budget, also the Claude CLI subprocess deadline; default 600 seconds |
| `MUSE_GATEWAY_STREAM_IDLE_TIMEOUT` | Upstream stream idle budget after headers; default 120 seconds |
| `MUSE_GATEWAY_TURN_TIMEOUT` | Absolute streaming turn deadline; default 3600 seconds |
| `MUSE_GATEWAY_MAX_CONCURRENT_REQUESTS` | Concurrent inference POST limit; default 8, excess requests receive 503 |
| `MUSE_GATEWAY_LOG_FILE` | Rotating diagnostics; default working-directory log |
| `--capture-dir` (CLI flag only) | Opt-in private captures; off by default |
| `--allow-image-files` (CLI flag only) | Accept `file://` image references and send the file bytes upstream; off by default. See [image input](#image-input) |
| `MUSE_GATEWAY_MODELS` | Optional exact provider/model menu allowlist; unset/blank shows all available |
| `MUSE_GATEWAY_LOCAL_UPSTREAM` | Local Responses service; unset disables local routing |
| `MUSE_GATEWAY_LOCAL_API_KEY` | Optional local-only credential; unset sends no Authorization header |
| `MUSE_GATEWAY_CLAUDE_TRANSPORT` | `cli` (default) or direct Messages `http` |
| `MUSE_GATEWAY_CLAUDE_CLI_PASSTHROUGH` / `--claude-cli-passthrough` | Which Claude credential vars the `claude -p` child may inherit: `oauth` (default), `api-key`, `all`, `none`. Every tier also keeps `CLAUDE_CONFIG_DIR` (a location, not a credential). Server-side policy of the serving gateway |
| `MUSE_GATEWAY_KIRO_UPSTREAM` | Separate Kiro Gateway; default loopback port 8000, `/v1` |
| `MUSE_GATEWAY_KIRO_API_KEY` | Kiro Gateway proxy key only |
| `--kiro-autostart` (CLI flag only) | Start the configured local checkout if needed; off by default |
| `--kiro-server-dir` (CLI flag only) | Checkout for autostart; default `~/kiro-gateway` |

`MUSE_GATEWAY_MODELS` only filters the menu. It preserves backend ordering,
never fabricates models, and does not block direct requests or remove internal
capability metadata. Unresolved IDs produce a diagnostic.

Model capacities resolve each field independently: positive backend metadata,
then the exact routed-model fallback in the packaged [provider catalogs](providers/),
otherwise missing. Context/output sources are recorded separately. There are
no global or provider-wide capacity overrides or global response-length setting.

Each provider has a JSON catalog: [Meta](providers/meta.json),
[Codex](providers/codex.json), [Claude](providers/claude.json),
[Kiro](providers/kiro.json), [Devin](providers/devin.json), and
[local](providers/local.json). These contain exact model IDs, capacity fallbacks,
per-field provenance, and ordered fallback rosters. Meta also supplies its
fallback effort metadata; Devin supplies its fallback labels/capabilities.
The local catalog is deliberately empty: no capacities are invented.

Backend values win. Catalog membership never grants account access, and
successful Devin discovery replaces the fallback roster rather than adding
absent supplemental models. Fallback rosters apply only when discovery is
unavailable or disabled. Catalogs are packaged with the application, validated
once at startup, and loaded independently of the working directory. In a
source checkout or editable install, update fallback data by editing the relevant
JSON catalog; no Python edits or new environment settings are needed. A regular
`pip install` copies the catalogs into site-packages, so edits there are
overwritten on upgrade. Invalid data fails clearly instead of silently disappearing.

Advanced operations stay CLI-only: `--host`, `--provider`, `--model`,
`--no-model-discovery`, `--portable` (local/generic only), `--max-body-bytes`,
`--insecure-upstream`, `--verbose`, credential/executable paths, and guarded
native endpoint overrides. See `python -m muse_gateway --help`.

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
`-LocalUpstream` and `-LocalApiKey`. Prefer environment settings to command-line
secrets. Kiro options are `-KiroUpstream`, `-KiroApiKey`, `-KiroServerDir`, and
`-KiroAutostart`, corresponding to the same `--kiro-*` launch flags. Devin
options are `-DevinUpstream` and `-DevinApiKey` (`--devin-upstream`,
`--devin-api-key`). `-Models` prints the model catalog instead of launching
(`--models`), and `-ModelsProvider` limits it to one provider
(`--models-provider`). With `powershell -File`, omit the `--`
separator and pass Muse arguments directly.

### Manual server setup

For separate server and Muse terminals, first export your configuration in the
server's environment, then start:

```bash
python -m muse_gateway auto
```

Server, doctor, model listing, and launch use the same `--env-file` loading
rules. In your working project, run Muse with `META_API_KEY` unset for
subscription authentication:

```bash
muse --provider meta --base-url http://127.0.0.1:8787
```

For Meta API-key billing, set standard `META_API_KEY` on the server (or use
`--meta-api-key`) and the appropriate `META_API_KEY` for Muse instead of using
the subscription launcher.

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
- **Local startup fails:** verify the endpoint, optional key, and its model
  metadata. The menu requires context and output capacities; a menu allowlist
  cannot substitute for missing metadata or create an available model.
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

Under WSL or with more than one home directory (for example a Windows and a
Linux login), the gateway reads credentials from the home of the process that
runs it. If `doctor` reports a credential as not found, check that the CLI
login lives in that same home; a symlinked credential file is deliberately
not followed, and `doctor` names it as a refused symlink.

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
arrives or the stream idle budget expires (`--stream-idle-timeout`, 120
seconds by default). `--timeout` (600 seconds by default) separately bounds
connect/headers and the Claude CLI subprocess. Do not assume immediate
upstream cancellation or stopped billing when closing the client. A
forced-disconnect check does not establish how Muse's Escape-key
cancellation behaves.

Doctor loads the same configuration file and environment as the other entry
points. It checks configuration/credential availability, not inference:

```bash
python -m muse_gateway doctor --doctor-provider codex
python -m muse_gateway doctor --doctor-provider claude
python -m muse_gateway doctor --doctor-provider devin
python -m muse_gateway doctor --doctor-provider kiro
python -m muse_gateway doctor --doctor-provider local
```

Bare `doctor` checks the default routes and configured local/Kiro routes. It can
report Meta as not ready without a static key even when subscription forwarding
will work once Muse sends its bearer. For local models, unusable discovery or
missing capacities make doctor report not ready; an optional missing key alone
does not. These are preflight results, not attempted model turns.

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

Capture is enabled only by the `--capture-dir` CLI flag. `/health` identifies the
capture session as well as the process/source. Each gateway request gets its
own directory containing request transforms, outgoing bytes, upstream headers
and error bodies, downstream HTTP/SSE bytes, and a monotonic read/frame/write
timeline. Devin protobufs are also saved in readable form. Known credentials
and identifying truncated prefixes are scrubbed before writing; this can also
remove matching fragments from conversation data. Saved Devin frames are
scrubbed, potentially re-encoded copies. Frames that cannot safely be scrubbed
are marked omitted, and hashes of original bytes are identified as such.
Client-facing error text is capped at 512 UTF-8 bytes; a Devin trailer kept as
capture evidence is scrubbed and truncated to 2,000 characters.
Capture does synchronous disk and redaction work on the request path, so it can
add or reshape latency; treat capture timelines as evidence from an instrumented
run, not as an uninstrumented performance measurement.

**Captures contain conversation content.** Owner-only permissions are applied,
including a protected Windows ACL, and the gateway does not upload these files.
Each request has a 64 MiB budget. Check `manifest.json` for truncation, omitted
data, capture errors, byte counts, and terminal events. Even a complete capture
is transport evidence: correlate it with Muse's completion or decoder error.
See [SECURITY.md](SECURITY.md) before sharing any files.

Captures are operator-owned forensic data. The gateway never deletes or
rotates them; remove capture directories by hand when the diagnosis is done.

## Compatibility

Muse's protocol can change between releases. Re-run consumer compatibility
checks after updating Muse; they are specific to the executable being tested.
[CONTRIBUTING.md](CONTRIBUTING.md#what-a-green-result-means) explains the checks
and which account/TUI workflows still need verification.

Advertised model limits are discovery data or fallbacks, not measurements.
Each capacity field is labeled by source: live backend evidence, documented
fallback, user-approved fallback, or bundled seed. Verification means the
backend supplied both limits; nothing else is proof of live-verified limits. Model quality,
tool reliability, and features such as signed reasoning remain provider-specific.
Switching providers does not make their opaque reasoning state interchangeable.

## Security

Keep the listener on loopback. There is no multi-tenant isolation. Loopback
requests require a loopback `Host`, and inference POSTs with a browser `Origin`
are rejected. `--require-inbound-token` optionally requires
`MUSE_GATEWAY_INBOUND_TOKEN` (sent as `x-muse-gateway-token` or Bearer) on
inference POSTs; `--allow-non-loopback` refuses to start without it.

Only configure upstreams you trust. Local requests use only the optional local
key; without one they send no Authorization header. Generic keys and Muse's
subscription credential are never forwarded to the local service. Meta uses
its explicit API key or incoming Muse authorization. A custom Meta URL receives
that authorization and native Meta payloads; an OpenAI-compatible label alone
does not establish support.

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

The `muse_shim` module, `muse-shim` console command and wrapper, older
health-service identifiers, and `shim_error` wire type remain for compatibility.
`MUSE_SHIM_*` settings are retired and produce migration guidance, not aliases.
The project name is **muse-gateway**.
