param(
  [int]$Wall = 120,
  [int]$Interval = 5,
  [switch]$StopOnDecision = $true
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-LastTraceLine {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return $null }
  $line = Get-Content -Path $Path -Tail 1 -ErrorAction SilentlyContinue
  if (-not $line) { return $null }
  try {
    return $line | ConvertFrom-Json
  } catch {
    return $null
  }
}

function Get-SeqMnistSummary {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return $null }
  try {
    $data = Get-Content -Path $Path -Raw | ConvertFrom-Json
  } catch {
    return $null
  }
  foreach ($row in $data) {
    if ($row.dataset -eq "seq_mnist") {
      if ($null -ne $row.absolute_hallway) {
        $train = $row.absolute_hallway.train
        $eval = $row.absolute_hallway.eval
        return [pscustomobject]@{
          ptr_flip_rate = $train.ptr_flip_rate
          ptr_mean_dwell = $train.ptr_mean_dwell
          ptr_max_dwell = $train.ptr_max_dwell
          eval_acc = $eval.eval_acc
          eval_loss = $eval.eval_loss
        }
      }
    }
  }
  return $null
}

function Get-LogError {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return $null }
  $tail = Get-Content -Path $Path -Tail 5 -ErrorAction SilentlyContinue
  foreach ($line in $tail) {
    if ($line -match "Traceback" -or $line -match "nan_guard" -or $line -match "NaN" -or $line -match "live_trace write failed") {
      return $line
    }
  }
  return $null
}

function Run-Phase {
  param(
    [string]$Name,
    [hashtable]$EnvVars,
    [double]$BaselineFlip = 0.0,
    [double]$BaselineDwell = 0.0
  )

  $tracePath = Join-Path $root "live_trace_$Name.json"
  $logPath = Join-Path $root "logs\\current\\tournament_phase6.log"
  $summaryPath = Join-Path $root "summaries\\current\\tournament_phase6_summary.json"
  $summaryOut = Join-Path $root "summary_$Name.json"

  foreach ($k in $EnvVars.Keys) {
    Set-Item -Path ("Env:{0}" -f $k) -Value $EnvVars[$k]
  }
  Set-Item -Path "Env:VAR_LIVE_TRACE_PATH" -Value $tracePath
  Set-Item -Path "Env:TP6_WALL" -Value $Wall
  Set-Item -Path "Env:VAR_LOG_EVERY_N_SECS" -Value $Interval
  Set-Item -Path "Env:VAR_LIVE_TRACE_EVERY_N_STEPS" -Value 1
  Set-Item -Path "Env:TP6_DEBUG_NAN" -Value 1
  Set-Item -Path "Env:TP6_DEBUG_STATS" -Value 1
  Set-Item -Path "Env:TP6_DEBUG_EVERY" -Value 1

  Write-Host ""
  Write-Host "==> Running $Name for $Wall s (heartbeat=$Interval s)"

  if (Test-Path $tracePath) { Remove-Item $tracePath -Force }

  $proc = Start-Process -FilePath "python" -ArgumentList "tournament_phase6.py" -WorkingDirectory $root -PassThru -NoNewWindow

  $hitCount = 0
  while (-not $proc.HasExited) {
    Start-Sleep -Seconds $Interval
    $t = Get-LastTraceLine -Path $tracePath
    if ($null -ne $t) {
      $delta = if ($null -ne $t.ptr_delta_abs_mean) { "{0:n4}" -f $t.ptr_delta_abs_mean } else { "n/a" }
      $jump = if ($null -ne $t.jump_p_mean) { "{0:n3}" -f $t.jump_p_mean } else { "n/a" }
      $mem = if ($null -ne $t.cuda_mem_alloc_mb) { "{0:n0}MB/{1:n0}MB" -f $t.cuda_mem_alloc_mb, $t.cuda_mem_reserved_mb } else { "n/a" }
      $msg = "[{0}] {1} step {2} loss {3:n4} flip {4:n3} dwell {5:n2} d| {6} p {7} mem {8}" -f (Get-Date -Format HH:mm:ss), $Name, $t.step, $t.loss, $t.ptr_flip_rate, $t.ptr_mean_dwell, $delta, $jump, $mem
      Write-Host $msg
      if ($StopOnDecision -and $Name -eq "stabilized" -and $BaselineFlip -gt 0.0) {
        $flipTarget = $BaselineFlip - 0.5
        $dwellTarget = [Math]::Max($BaselineDwell * 3.0, 5.0)
        if (($t.ptr_flip_rate -le $flipTarget) -and ($t.ptr_mean_dwell -ge $dwellTarget)) {
          $hitCount += 1
        } else {
          $hitCount = 0
        }
        if ($hitCount -ge 3) {
          Write-Host "Decision rule met (flip<=${flipTarget}, dwell>=${dwellTarget}). Stopping $Name early."
          Stop-Process -Id $proc.Id -Force
          break
        }
      }
    }

    $err = Get-LogError -Path $logPath
    if ($null -ne $err) {
      Write-Host "Error detected: $err"
      Stop-Process -Id $proc.Id -Force
      break
    }
  }

  try { Wait-Process -Id $proc.Id -ErrorAction SilentlyContinue } catch {}
  if (Test-Path $summaryPath) { Copy-Item $summaryPath $summaryOut -Force }

  return Get-SeqMnistSummary -Path $summaryOut
}

$common = @{
  VAR_COMPUTE_DEVICE = "cuda"
  TP6_PRECISION = "fp32"
  TP6_BATCH_SIZE = "128"
  TP6_LR = "1e-3"
  TP6_GRAD_CLIP = "1.0"
  TP6_STATE_CLIP = "5.0"
  TP6_STATE_DECAY = "0.99"
  PARAM_POINTER_FORWARD_STEP_PROB = "0.2"
}

$baseline = $common.Clone()
$baseline.TP6_PTR_INERTIA = "0.0"
$baseline.TP6_PTR_DEADZONE = "0.0"
$baseline.TP6_PTR_DEADZONE_TAU = "0.001"
$baseline.TP6_PTR_PHANTOM = "0"
$baseline.TP6_PTR_PHANTOM_OFF = "0.5"

$stabilized = $common.Clone()
$stabilized.TP6_PTR_INERTIA = "0.9"
$stabilized.TP6_PTR_DEADZONE = "0.2"
$stabilized.TP6_PTR_DEADZONE_TAU = "0.01"
$stabilized.TP6_PTR_PHANTOM = "1"
$stabilized.TP6_PTR_PHANTOM_OFF = "0.5"
$stabilized.PARAM_POINTER_FORWARD_STEP_PROB = "0.0"

$baseSummary = Run-Phase -Name "baseline" -EnvVars $baseline
if ($null -eq $baseSummary) {
  Write-Host "Baseline summary missing. Aborting."
  exit 1
}

$stabSummary = Run-Phase -Name "stabilized" -EnvVars $stabilized -BaselineFlip $baseSummary.ptr_flip_rate -BaselineDwell $baseSummary.ptr_mean_dwell
if ($null -eq $stabSummary) {
  Write-Host "Stabilized summary missing."
  exit 1
}

Write-Host ""
Write-Host "==> A/B Summary"
Write-Host ("baseline  flip {0:n3} dwell {1:n2} acc {2:n3} loss {3:n4}" -f $baseSummary.ptr_flip_rate, $baseSummary.ptr_mean_dwell, $baseSummary.eval_acc, $baseSummary.eval_loss)
Write-Host ("stabilzd  flip {0:n3} dwell {1:n2} acc {2:n3} loss {3:n4}" -f $stabSummary.ptr_flip_rate, $stabSummary.ptr_mean_dwell, $stabSummary.eval_acc, $stabSummary.eval_loss)




