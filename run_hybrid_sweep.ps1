param(
  [int]$Wall = 60,
  [int]$Batch = 64,
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Run-Case {
  param(
    [string]$Name,
    [hashtable]$EnvOverrides
  )

  Write-Host "=== Running $Name ==="
  $env:TP6_DEVICE = $Device
  $env:TP6_PRECISION = "fp32"
  $env:TP6_SYNTH = "1"
  $env:TP6_SYNTH_MODE = "markov0"
  $env:TP6_SYNTH_LEN = "256"
  $env:TP6_MAX_SAMPLES = "2048"
  $env:TP6_EVAL_SAMPLES = "256"
  $env:TP6_BATCH_SIZE = "$Batch"
  $env:TP6_LR = "1e-4"
  $env:TP6_WALL = "$Wall"
  $env:TP6_HEARTBEAT_SECS = "5"
  $env:TP6_HEARTBEAT = "10"
  $env:TP6_LIVE_TRACE_EVERY = "1"
  $env:TP6_LIVE_TRACE = "$root\\live_trace_$Name.json"
  $env:TP6_DEBUG_NAN = "1"
  $env:TP6_DEBUG_STATS = "1"
  $env:TP6_DEBUG_EVERY = "1"
  $env:TP6_GRAD_CLIP = "0.5"
  $env:TP6_STATE_CLIP = "5.0"
  $env:TP6_STATE_DECAY = "0.995"
  $env:TP6_UPDATE_SCALE = "0.5"

  # Hybrid defaults (phantom + soft slice + continuous pointer)
  $env:TP6_PTR_PHANTOM = "1"
  $env:TP6_PTR_PHANTOM_OFF = "0.5"
  $env:TP6_PTR_NO_ROUND = "1"
  $env:TP6_SOFT_READOUT = "1"
  $env:TP6_SOFT_READOUT_K = "2"
  $env:TP6_SOFT_READOUT_TAU = "0.5"
  $env:TP6_PTR_WALK_PROB = "0.2"

  foreach ($k in $EnvOverrides.Keys) {
    Set-Item -Path "Env:$k" -Value $EnvOverrides[$k]
  }

  $logPath = "$root\\audit_$Name.log"
  $summaryPath = "$root\\summary_$Name.json"
  if (Test-Path $logPath) { Remove-Item $logPath -Force }
  if (Test-Path $summaryPath) { Remove-Item $summaryPath -Force }

  & "$root\\.venv\\Scripts\\python.exe" "$root\\tournament_phase6.py" 2>&1 | Tee-Object -FilePath $logPath

  $logSrc = Join-Path $root "logs\\current\\tournament_phase6.log"
  $summarySrc = Join-Path $root "summaries\\current\\tournament_phase6_summary.json"
  if (Test-Path $logSrc) {
    Copy-Item $logSrc "$root\\tournament_phase6_$Name.log" -Force
  }
  if (Test-Path $summarySrc) {
    Copy-Item $summarySrc $summaryPath -Force
  }
}

$inertias = @("0.1", "0.2", "0.3")
$deadzones = @("0.00", "0.02")

foreach ($dz in $deadzones) {
  foreach ($in in $inertias) {
    $name = "hyb_i${in}_dz${dz}"
    Run-Case $name @{
      TP6_PTR_INERTIA = $in
      TP6_PTR_DEADZONE = $dz
      TP6_PTR_DEADZONE_TAU = "0.001"
    }
  }
}

Write-Host "=== Hybrid sweep complete ==="
