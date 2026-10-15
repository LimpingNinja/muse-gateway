<#
.SYNOPSIS
  One-shot launcher: starts the muse-gateway, then Muse on top of it.

.DESCRIPTION
  Thin wrapper around `python muse_gateway.py launch` (same as the installed
  `muse-gateway launch` command). Named parameters cover the launcher options;
  all remaining arguments pass straight through to `muse`, so this script
  behaves like the bare `muse` command line: subcommands (resume, exec),
  flags (--model, --yolo), and --help all work. Your working directory is
  preserved for the Muse process. If the port already serves a compatible
  gateway (e.g. from another terminal), Muse attaches to it instead of
  starting a new one. A Kiro-configured launch instead requires a free port,
  so this terminal's proxy settings cannot be silently ignored.

  Meta models ride your `muse login` subscription (OAuth) via header
  forwarding, so no Meta API key is configured. Codex uses your existing
  `codex login`. Install once with `python -m pip install .` from the checkout;
  protobuf is installed with the package. A virtual environment is optional.

  Local-backend config comes from flags (-LocalUpstream, -LocalApiKey,
  -LocalModels), environment (MUSE_GATEWAY_LOCAL_UPSTREAM,
  MUSE_GATEWAY_LOCAL_API_KEY, MUSE_GATEWAY_LOCAL_MODELS; the MUSE_SHIM_*
  spellings still work), or a `.env` file next to this script (copy
  `.env.example` to start; override with -EnvFile); flags win over env, env
  wins over the file. Without a key the launcher warns and continues.
  -LogFile and -MuseBin map to --log-file and --muse-bin; -Port is only
  forwarded when explicitly supplied.

  Kiro Gateway is a separate service. -KiroUpstream, -KiroApiKey, -KiroModels,
  -KiroContextLimit and -KiroOutputLimit map to the corresponding --kiro-*
  launcher options. Prefer a private env file for the proxy key.

.EXAMPLE
  .\muse-gateway.ps1
  Starts the gateway on :8787 and opens interactive Muse.

.EXAMPLE
  .\muse-gateway.ps1 resume <session-id>
  Resume a previous session through the gateway.

.EXAMPLE
  .\muse-gateway.ps1 -CheckOnly
  Preflight + gateway + catalog check, then exits (no Muse or inference).

.EXAMPLE
  .\muse-gateway.ps1 -Port 9000
  Starts the gateway and Muse on a different port.

.NOTES
  For Meta API-key billing, use the manual server setup in README.md instead
  of this subscription-authentication launcher.
  Headless `exec` arguments pass to Muse, whose model-profile/compaction rules
  still apply. Invoke this script by absolute path to work in another project.
  When launching via `powershell -File`, omit the `--` separator and pass
  Muse arguments directly (bare subcommands and flags all work); `-File`
  cannot forward a literal `--` token.
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Port = 8787,
    [switch]$CheckOnly,
    [string]$LocalUpstream = "",
    [string]$LocalApiKey = "",
    [string]$LocalModels = "",
    [string]$KiroUpstream = "",
    [string]$KiroApiKey = "",
    [string]$KiroModels = "",
    [int]$KiroContextLimit,
    [int]$KiroOutputLimit,
    [string]$LogFile = "",
    [string]$MuseBin = "",
    [string]$EnvFile = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MuseArgs
)

$ErrorActionPreference = "Stop"
$GatewayScript = Join-Path $PSScriptRoot "muse_gateway.py"
if (-not $EnvFile) { $EnvFile = Join-Path $PSScriptRoot ".env" }

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $PyLauncher) { throw "python not found on PATH (tried 'python' and 'py')" }
    $PythonCommand = @($PyLauncher.Source, "-3")
} else {
    $PythonCommand = @($Python.Source)
}
$launchArgs = @("launch", "--env-file", $EnvFile)
if ($PSBoundParameters.ContainsKey("Port")) { $launchArgs += "--port", "$Port" }
if ($LogFile) { $launchArgs += "--log-file", $LogFile }
if ($CheckOnly) { $launchArgs += "--check-only" }
if ($LocalUpstream) { $launchArgs += "--local-upstream", $LocalUpstream }
if ($LocalApiKey) { $launchArgs += "--local-api-key", $LocalApiKey }
if ($LocalModels) { $launchArgs += "--local-models", $LocalModels }
if ($KiroUpstream) { $launchArgs += "--kiro-upstream", $KiroUpstream }
if ($KiroApiKey) { $launchArgs += "--kiro-api-key", $KiroApiKey }
if ($KiroModels) { $launchArgs += "--kiro-models", $KiroModels }
if ($PSBoundParameters.ContainsKey("KiroContextLimit")) { $launchArgs += "--kiro-context-limit", "$KiroContextLimit" }
if ($PSBoundParameters.ContainsKey("KiroOutputLimit")) { $launchArgs += "--kiro-output-limit", "$KiroOutputLimit" }
if ($MuseBin) { $launchArgs += "--muse-bin", $MuseBin }
if ($PythonCommand.Count -gt 1) {
    & $PythonCommand[0] $PythonCommand[1] $GatewayScript @launchArgs @MuseArgs
} else {
    & $PythonCommand[0] $GatewayScript @launchArgs @MuseArgs
}
exit $LASTEXITCODE
