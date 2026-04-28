param(
  [string]$ApiBaseUrl = "http://127.0.0.1:8000",
  [string]$FrontendDir = "frontend",
  [int]$StartupWaitSeconds = 25,
  [switch]$AutoStartBackend,
  [string]$BackendCommand = "",
  [int]$BackendStartupTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
$backend = $null
$tauri = $null
$backendOutLog = $null
$backendErrLog = $null

function Test-Backend([string]$baseUrl) {
  try {
    Invoke-RestMethod -Method GET -Uri "$baseUrl/openapi.json" -TimeoutSec 3 | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Resolve-BackendCommand() {
  if ($BackendCommand.Trim().Length -gt 0) {
    return $BackendCommand
  }

  $venvPythonCandidate = Join-Path $PSScriptRoot "..\\MultiAgentPatent\\Scripts\\python.exe"
  $venvResolved = Resolve-Path $venvPythonCandidate -ErrorAction SilentlyContinue
  if ($venvResolved) {
    $venvPython = $venvResolved.Path
    return "`"$venvPython`" -m uvicorn main:app --host 127.0.0.1 --port 8000"
  }

  return "python -m uvicorn main:app --host 127.0.0.1 --port 8000"
}

function Get-LogTail([string]$path, [int]$lines = 40) {
  if (-not $path -or -not (Test-Path $path)) {
    return "<no log output>"
  }
  $tail = Get-Content -Path $path -ErrorAction SilentlyContinue | Select-Object -Last $lines
  if (-not $tail) {
    return "<empty>"
  }
  return ($tail -join [Environment]::NewLine)
}

function Start-Backend([string]$command) {
  $tempDir = Join-Path $PSScriptRoot "..\\scripts\\_tmp"
  New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $backendOutLog = (Join-Path $tempDir "backend_$stamp.out.log")
  $backendErrLog = (Join-Path $tempDir "backend_$stamp.err.log")

  $proc = Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList "/c", $command `
    -WorkingDirectory (Resolve-Path (Join-Path $PSScriptRoot "..")).Path `
    -RedirectStandardOutput $backendOutLog `
    -RedirectStandardError $backendErrLog `
    -PassThru

  return @{
    Process = $proc
    OutLog = $backendOutLog
    ErrLog = $backendErrLog
  }
}

function Invoke-JsonPost([string]$uri, [object]$payload, [int]$depth = 12) {
  $json = $payload | ConvertTo-Json -Depth $depth -Compress
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
  return Invoke-RestMethod -Method POST -Uri $uri -ContentType "application/json; charset=utf-8" -Body $bytes
}

try {
  Write-Host "[0/4] Checking backend API..." -ForegroundColor Cyan
  if (-not (Test-Backend $ApiBaseUrl)) {
    if ($AutoStartBackend) {
      $resolvedBackendCommand = Resolve-BackendCommand
      Write-Host "Backend not reachable. Starting backend with: $resolvedBackendCommand" -ForegroundColor Yellow

      $started = Start-Backend $resolvedBackendCommand
      $backend = $started.Process
      $backendOutLog = $started.OutLog
      $backendErrLog = $started.ErrLog

      $ready = $false
      for ($i = 0; $i -lt $BackendStartupTimeoutSeconds; $i++) {
        if ($backend.HasExited) {
          $outTail = Get-LogTail $backendOutLog
          $errTail = Get-LogTail $backendErrLog
          throw "Backend exited early with code $($backend.ExitCode).`n--- stdout ---`n$outTail`n--- stderr ---`n$errTail"
        }
        if (Test-Backend $ApiBaseUrl) {
          $ready = $true
          break
        }
        Start-Sleep -Seconds 1
      }

      if (-not $ready) {
        $outTail = Get-LogTail $backendOutLog
        $errTail = Get-LogTail $backendErrLog
        throw "Backend API still unreachable at $ApiBaseUrl after $BackendStartupTimeoutSeconds seconds.`n--- stdout ---`n$outTail`n--- stderr ---`n$errTail"
      }
    } else {
      throw "Backend API unreachable at $ApiBaseUrl. Please start backend first (e.g. .\\MultiAgentPatent\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000), or rerun script with -AutoStartBackend."
    }
  }

  Write-Host "[1/4] Starting Tauri desktop app..." -ForegroundColor Cyan
  $tauri = Start-Process -FilePath "pnpm.cmd" -ArgumentList "--dir", $FrontendDir, "tauri:dev" -PassThru
  Start-Sleep -Seconds $StartupWaitSeconds

  if ($tauri.HasExited) {
    throw "tauri:dev exited early with code $($tauri.ExitCode)"
  }

  Write-Host "[2/4] Running Draft flow (start -> continue)..." -ForegroundColor Cyan
  $draftStartPayload = @{
    idempotency_key = [guid]::NewGuid().ToString()
    disclosure_text = ("Frontend smoke disclosure text. " * 12)
    metadata = @{ source = "desktop-smoke" }
  }

  $draftStart = Invoke-JsonPost -Uri "$ApiBaseUrl/api/v1/draft/start" -Payload $draftStartPayload -Depth 8
  if (-not $draftStart.session_id) { throw "draft/start did not return session_id" }

  $draftContinuePayload = @{
    session_id = $draftStart.session_id
    approved_claims = $draftStart.data.claims
  }

  # Validate serialization early to avoid opaque backend parse errors.
  $null = $draftContinuePayload | ConvertTo-Json -Depth 20 -Compress

  $draftContinue = Invoke-JsonPost -Uri "$ApiBaseUrl/api/v1/draft/continue" -Payload $draftContinuePayload -Depth 20
  if ($draftContinue.status -notin @("completed", "running")) {
    throw "draft/continue unexpected status: $($draftContinue.status)"
  }

  Write-Host "[3/4] Running OA flow (start -> completed)..." -ForegroundColor Cyan
  $oaPayload = @{
    idempotency_key = [guid]::NewGuid().ToString()
    oa_text = "Examiner argues claim 1 lacks inventiveness over D1 and D2."
    original_claims = @{ independent_claims = @(@{ id = 1; text = "A system ..."; depends_on = @() }); dependent_claims = @() }
    prior_arts_paths = @()
  }

  $oaStart = Invoke-JsonPost -Uri "$ApiBaseUrl/api/v1/oa/start" -Payload $oaPayload -Depth 10
  if ($oaStart.status -notin @("completed", "running")) {
    throw "oa/start unexpected status: $($oaStart.status)"
  }

  Write-Host "[4/4] Smoke passed." -ForegroundColor Green
}
finally {
  Write-Host "Stopping started processes..." -ForegroundColor DarkGray
  if ($tauri -and -not $tauri.HasExited) {
    Stop-Process -Id $tauri.Id -Force
  }
  if ($backend -and -not $backend.HasExited) {
    Stop-Process -Id $backend.Id -Force
  }
}
