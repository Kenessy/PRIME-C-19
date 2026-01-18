param(
  [int]$Steps = 200,
  [int]$Wall = 1200,
  [int]$Interval = 5,
  [string]$Seeds = "123,456,789"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$summaryPath = Join-Path $root "summaries\current\tournament_phase6_summary.json"
$csvOut = Join-Path $root "proof_ab.csv"

function Get-SeqMnistSummary {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return $null }
  try {
    $data = Get-Content -Path $Path -Raw | ConvertFrom-Json
  } catch {
    return $null
  }
  foreach ($row in $data) {
    if ($row.dataset -eq "seq_mnist" -and $null -ne $row.absolute_hallway) {
      $train = $row.absolute_hallway.train
      $eval = $row.absolute_hallway.eval
      return [pscustomobject]@{
        loss_slope = $train.loss_slope
        steps = $train.steps
        ptr_flip_rate = $train.ptr_flip_rate
        ptr_mean_dwell = $train.ptr_mean_dwell
        ptr_max_dwell = $train.ptr_max_dwell
        eval_acc = $eval.eval_acc
        eval_loss = $eval.eval_loss
      }
    }
  }
  return $null
}

function Run-Once {
  param(
    [string]$Name,
    [hashtable]$EnvVars,
    [int]$Seed
  )
  foreach ($k in $EnvVars.Keys) {
    Set-Item -Path ("Env:{0}" -f $k) -Value $EnvVars[$k]
  }
  Set-Item -Path "Env:VAR_RUN_SEED" -Value $Seed
  Set-Item -Path "Env:TP6_MAX_STEPS" -Value $Steps
  Set-Item -Path "Env:TP6_WALL" -Value $Wall
  Set-Item -Path "Env:VAR_LOG_EVERY_N_SECS" -Value $Interval
  Set-Item -Path "Env:VAR_LIVE_TRACE_PATH" -Value ""
  Set-Item -Path "Env:VAR_LIVE_TRACE_EVERY_N_STEPS" -Value 1

  if (Test-Path $summaryPath) { Remove-Item $summaryPath -Force }

  Write-Host ""
  Write-Host ("==> {0} | seed {1} | steps {2}" -f $Name, $Seed, $Steps)
  $proc = Start-Process -FilePath "python" -ArgumentList "tournament_phase6.py" -WorkingDirectory $root -PassThru -NoNewWindow
  Wait-Process -Id $proc.Id

  $outPath = Join-Path $root ("summary_{0}_seed{1}.json" -f $Name, $Seed)
  if (Test-Path $summaryPath) { Copy-Item $summaryPath $outPath -Force }
  return Get-SeqMnistSummary -Path $outPath
}

$common = @{
  VAR_COMPUTE_DEVICE = "cuda"
  TP6_PRECISION = "fp32"
  TP6_BATCH_SIZE = "128"
  TP6_LR = "1e-3"
  TP6_GRAD_CLIP = "1.0"
  TP6_STATE_CLIP = "5.0"
  TP6_STATE_DECAY = "0.99"
  TP6_PTR_WALK_PROB = "0.2"
  TP6_MAX_SAMPLES = "5000"
  TP6_EVAL_SAMPLES = "1024"
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
$stabilized.TP6_PTR_WALK_PROB = "0.0"

$seedList = $Seeds -split "," | ForEach-Object { [int]$_.Trim() } | Where-Object { $_ -gt 0 }
$rows = @()

foreach ($seed in $seedList) {
  $base = Run-Once -Name "baseline" -EnvVars $baseline -Seed $seed
  if ($null -eq $base) { throw "Baseline summary missing for seed $seed" }
  $rows += [pscustomobject]@{
    seed = $seed
    variant = "baseline"
    loss_slope = $base.loss_slope
    steps = $base.steps
    ptr_flip_rate = $base.ptr_flip_rate
    ptr_mean_dwell = $base.ptr_mean_dwell
    ptr_max_dwell = $base.ptr_max_dwell
    eval_acc = $base.eval_acc
    eval_loss = $base.eval_loss
  }

  $stab = Run-Once -Name "stabilized" -EnvVars $stabilized -Seed $seed
  if ($null -eq $stab) { throw "Stabilized summary missing for seed $seed" }
  $rows += [pscustomobject]@{
    seed = $seed
    variant = "stabilized"
    loss_slope = $stab.loss_slope
    steps = $stab.steps
    ptr_flip_rate = $stab.ptr_flip_rate
    ptr_mean_dwell = $stab.ptr_mean_dwell
    ptr_max_dwell = $stab.ptr_max_dwell
    eval_acc = $stab.eval_acc
    eval_loss = $stab.eval_loss
  }
}

$rows | Export-Csv -Path $csvOut -NoTypeInformation
Write-Host ""
Write-Host "==> Proof run complete: $csvOut"

foreach ($variant in @("baseline", "stabilized")) {
  $subset = $rows | Where-Object { $_.variant -eq $variant }
  $avgFlip = ($subset | Measure-Object -Property ptr_flip_rate -Average).Average
  $avgDwell = ($subset | Measure-Object -Property ptr_mean_dwell -Average).Average
  $avgAcc = ($subset | Measure-Object -Property eval_acc -Average).Average
  $avgLoss = ($subset | Measure-Object -Property eval_loss -Average).Average
  Write-Host ("{0,-10} flip {1:n3} | dwell {2:n2} | acc {3:n3} | loss {4:n4}" -f $variant, $avgFlip, $avgDwell, $avgAcc, $avgLoss)
}





