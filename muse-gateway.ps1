<#
.SYNOPSIS
  One-shot launcher: starts the muse-gateway, then Muse on top of it.

.DESCRIPTION
  Thin wrapper around `python muse_shim.py launch` (same as the installed
  `muse-gateway launch` command). Named parameters cover the launcher options;
  all remaining arguments pass straight through to `muse`, so this script
  behaves like the bare `muse` command line: subcommands (resume, exec),
  flags (--model, --yolo), and --help all work. Your working directory is
  preserved for the Muse process.

  Meta models ride your `muse login` subscription (OAuth) via header
  forwarding, so no Meta API key is configured. Codex uses your existing
  `codex login`. No virtualenv needed: the gateway is stdlib-only,
  zero dependencies.

  Local-backend config comes from flags (-LocalUpstream, -LocalApiKey,
  -LocalModels), environment (MUSE_SHIM_LOCAL_UPSTREAM, MUSE_SHIM_LOCAL_API_KEY,
  MUSE_SHIM_LOCAL_MODELS), or a `.env` file next to this script (copy
  `.env.example` to start; override with -EnvFile); flags win over env, env
  wins over the file. Without a key the launcher warns and continues.
  -LogFile and -MuseBin map to --log-file and --muse-bin; -Port is only
  forwarded when explicitly supplied.

.EXAMPLE
  .\muse-gateway.ps1
  Starts the gateway on :8787 and opens interactive Muse.

.EXAMPLE
  .\muse-gateway.ps1 resume <session-id>
  Resume a previous session through the gateway.

.EXAMPLE
  .\muse-gateway.ps1 exec "summarize the repo root" --model local/<alias>
  One-shot agent run through the gateway on a local model.

.EXAMPLE
  .\muse-gateway.ps1 -CheckOnly
  Preflight + gateway + catalog check, then exits (no Muse launch).

.EXAMPLE
  .\muse-gateway.ps1 -Port 9000 -- --yolo
  Custom port; extra args after -- go straight to `muse`.

.NOTES
  For Meta API-key billing, use the manual server setup in README.md instead
  of this subscription-authentication launcher.
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
    [string]$LogFile = "",
    [string]$MuseBin = "",
    [string]$EnvFile = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MuseArgs
)

$ErrorActionPreference = "Stop"
$ShimScript = Join-Path $PSScriptRoot "muse_shim.py"
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
if ($MuseBin) { $launchArgs += "--muse-bin", $MuseBin }
if ($PythonCommand.Count -gt 1) {
    & $PythonCommand[0] $PythonCommand[1] $ShimScript @launchArgs @MuseArgs
} else {
    & $PythonCommand[0] $ShimScript @launchArgs @MuseArgs
}
exit $LASTEXITCODE
