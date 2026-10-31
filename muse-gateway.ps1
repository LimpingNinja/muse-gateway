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

  Local config uses -LocalUpstream / -LocalApiKey or current
  MUSE_GATEWAY_LOCAL_UPSTREAM / MUSE_GATEWAY_LOCAL_API_KEY settings.
  Unset upstream disables local; its key is optional, never borrowed.
  MUSE_GATEWAY_MODELS filters exact provider/model IDs, not entitlement.
  Flags win over current environment, then the existing file (-EnvFile).
  Removed settings warn; conflicting aliases and removed flags fail.
  -LogFile and -MuseBin map to --log-file and --muse-bin; -Port is only
  forwarded when explicitly supplied.

  Kiro Gateway is a separate service. -KiroUpstream, -KiroApiKey,
  -KiroServerDir and -KiroAutostart map to the corresponding --kiro-*
  launcher options. Prefer a private env file for the proxy key.

  -DevinUpstream and -DevinApiKey map to --devin-upstream and
  --devin-api-key. -Models prints the model catalog (--models) instead of
  launching; -ModelsProvider (--models-provider) limits it to one provider.

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
    [switch]$Models,
    [string]$LocalUpstream = "",
    [string]$LocalApiKey = "",
    [string]$KiroUpstream = "",
    [string]$KiroApiKey = "",
    [string]$KiroServerDir = "",
    [switch]$KiroAutostart,
    [string]$DevinUpstream = "",
    [string]$DevinApiKey = "",
    [string]$ModelsProvider = "",
    [string]$LogFile = "",
    [string]$MuseBin = "",
    [string]$EnvFile = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MuseArgs
)

$ErrorActionPreference = "Stop"
foreach ($arg in $MuseArgs) {
    $name = ($arg -split "[=:]", 2)[0]
    if ($name -match '^-(MetaModels|CodexModels|ClaudeModels|DevinModels|KiroModels|LocalModels)$') {
        throw "Removed gateway option $name; use MUSE_GATEWAY_MODELS with exact provider/model IDs (filter only)."
    }
    if ($name -match '^-(MetaBaseUrl|CodexBaseUrl|ClaudeBaseUrl|DevinBaseUrl|KiroBaseUrl|LocalBaseUrl)$') {
        throw "Removed gateway option $name; use the corresponding -*Upstream option."
    }
    if ($name -match '^-(DevinAuthDb|DevinCredentialId|DevinDefaultFile)$') {
        throw "Removed Devin credential selection option $name; use devin auth login or DEVIN_API_KEY."
    }
    if ($name -match '^-(KiroContextLimit|KiroOutputLimit|KiroModelBudgets|MaxOutputTokens|RefreshSkew)$') {
        throw "Removed gateway option $name; remove the budget/timing override. Per-request max_output_tokens remains supported."
    }
}
if ($Models -and $CheckOnly) { throw "Use -Models or -CheckOnly, not both." }
$GatewayScript = Join-Path $PSScriptRoot "muse_gateway.py"
$EnvFileBound = $PSBoundParameters.ContainsKey("EnvFile")
if (-not $EnvFileBound) { $EnvFile = Join-Path $PSScriptRoot ".env" }

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonCommand = @($VenvPython)
} else {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        $PythonCommand = @($PyLauncher.Source, "-3")
    } else {
        $Python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $Python) { throw "python not found on PATH (tried 'py' and 'python')" }
        $PythonCommand = @($Python.Source)
    }
}
$launchArgs = @("launch")
if ($Models) { $launchArgs = @("--models") }
if ($EnvFileBound) { $launchArgs += "--env-file=$EnvFile" }
elseif (Test-Path $EnvFile) { $launchArgs += "--env-file=$EnvFile" }
if ($PSBoundParameters.ContainsKey("Port")) { $launchArgs += "--port", "$Port" }
if ($PSBoundParameters.ContainsKey("LogFile")) { $launchArgs += "--log-file=$LogFile" }
if ($CheckOnly) { $launchArgs += "--check-only" }
if ($PSBoundParameters.ContainsKey("LocalUpstream")) { $launchArgs += "--local-upstream=$LocalUpstream" }
if ($PSBoundParameters.ContainsKey("LocalApiKey")) { $launchArgs += "--local-api-key=$LocalApiKey" }
if ($PSBoundParameters.ContainsKey("KiroUpstream")) { $launchArgs += "--kiro-upstream=$KiroUpstream" }
if ($PSBoundParameters.ContainsKey("KiroApiKey")) { $launchArgs += "--kiro-api-key=$KiroApiKey" }
if ($PSBoundParameters.ContainsKey("KiroServerDir")) { $launchArgs += "--kiro-server-dir=$KiroServerDir" }
if ($PSBoundParameters.ContainsKey("DevinUpstream")) { $launchArgs += "--devin-upstream=$DevinUpstream" }
if ($PSBoundParameters.ContainsKey("DevinApiKey")) { $launchArgs += "--devin-api-key=$DevinApiKey" }
if ($PSBoundParameters.ContainsKey("ModelsProvider")) { $launchArgs += "--models-provider=$ModelsProvider" }
if ($KiroAutostart) { $launchArgs += "--kiro-autostart" }
if ($PSBoundParameters.ContainsKey("MuseBin")) { $launchArgs += "--muse-bin=$MuseBin" }
if ($PythonCommand.Count -gt 1) {
    & $PythonCommand[0] $PythonCommand[1] $GatewayScript @launchArgs @MuseArgs
} else {
    & $PythonCommand[0] $GatewayScript @launchArgs @MuseArgs
}
exit $LASTEXITCODE
