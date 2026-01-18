# Runtime Config (TP6_*)

This table mirrors the current environment variables used by `tournament_phase6.py`
and `prime_c19/settings.py`. Defaults are from code; scripts may override them.

## System & paths

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| VAR_PROJECT_ROOT | repo root | str | Project root override (data/logs relative paths). |
| VAR_COMPUTE_DEVICE | auto | str | `cuda` or `cpu` device selection. |
| PILOT_OFFLINE | 1 | flag | Disable network downloads (offline-only mode). |
| VAR_TORCHAUDIO_BACKEND | "" | str | Torchaudio backend name (if available). |
| VAR_RUN_SEED | 123 | int | RNG seed for numpy/torch/python. |
| VAR_LOGGING_PATH | (derived) | str | Override log file path (default: `logs/current/tournament_phase6.log`). |

## Data & batching

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_MAX_SAMPLES | 5000 | int | Max training samples (dataset cap). |
| TP6_EVAL_SAMPLES | 1024 | int | Eval sample count. |
| TP6_EVAL_SPLIT | test | str | Eval split (`test` or `train` subset). |
| TP6_BATCH_SIZE | 128 | int | Batch size. |
| TP6_SYNTH | 0 | flag | Use synthetic dataset instead of MNIST. |
| TP6_SYNTH_MODE | random | str | Synthetic mode (markov0, assoc_clean, etc). |
| TP6_SYNTH_LEN | 256 | int | Synthetic sequence length. |
| TP6_SYNTH_SHUFFLE | 0 | flag | Shuffle synthetic samples. |
| TP6_HAND_MIN | 256 | int | Minimum length for hand_kv JSONL. |
| TP6_ASSOC_KEYS | 4 | int | Assoc-clean keys per sample. |
| TP6_ASSOC_PAIRS | 3 | int | Assoc-clean key/value pairs. |

## Training schedule & logging

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_LR | 1e-3 | float | Optimizer LR. |
| TP6_WALL | 900 | int | Wall-clock seconds for `train_wallclock`. |
| TP6_MAX_STEPS | 0 | int | Hard cap on steps (0 = no cap). |
| VAR_LOG_EVERY_N_STEPS | 10 | int | Log every N steps. |
| VAR_LOG_EVERY_N_SECS | 0.0 | float | Log every N seconds (0 = off). |
| VAR_LIVE_TRACE_EVERY_N_STEPS | heartbeat | int | Live trace interval (steps). |
| VAR_LIVE_TRACE_PATH | traces/current/live_trace.json | str | Live trace output path. |
| VAR_TRAINING_TRACE_ENABLED | 0 | flag | Write train_steps trace JSONL. |
| VAR_TRAINING_TRACE_PATH | traces/current/train_steps_trace.jsonl | str | Train trace output path. |
| VAR_LOSS_HISTORY_LEN | 2000 | int | Loss history length for slope. |
| TP6_SATIETY | 0.98 | float | Early-exit confidence threshold. |

## Ring geometry & kernels

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_RING_LEN | 4096 | int | Ring length. |
| TP6_PTR_STRIDE | 1 | int | Param stride for ptr parameters. |
| TP6_GAUSS_K | 2 | int | Kernel radius (window = 2K+1). |
| TP6_GAUSS_TAU | 0.5 | float | Gaussian tau (width). |
| TP6_PTR_KERNEL | gauss | str | Kernel type (`gauss` or `vonmises`). |
| TP6_PTR_KAPPA | 4.0 | float | Von Mises kappa. |
| TP6_PTR_EDGE_EPS | 0.0 | float | Edge band for debug stats. |
| TP6_LMOVE | 1e-3 | float | Movement penalty coefficient. |

## Pointer motion controls

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_PTR_INERTIA | 0.0 | float | Stay bias (0 = none). |
| TP6_PTR_DEADZONE | 0.0 | float | Distance below which movement is resisted. |
| TP6_PTR_DEADZONE_TAU | 1e-3 | float | Soft mask temperature for deadzone. |
| TP6_PTR_WARMUP_STEPS | 0 | int | Warmup steps with pointer locked. |
| TP6_PTR_WALK_PROB | 0.2 | float | Walk vs stay when not jumping. |
| TP6_PTR_NO_ROUND | 0 | flag | Use continuous target (no STE rounding). |
| TP6_PTR_PHANTOM | 0 | flag | Phantom hysteresis quantizer. |
| TP6_PTR_PHANTOM_OFF | 0.5 | float | Phantom offset. |
| TP6_PTR_PHANTOM_READ | 0 | flag | Quantize read pointer to phantom bin. |
| TP6_SOFT_READOUT | 0 | flag | Soft readout (kernel around pointer). |
| TP6_SOFT_READOUT_K | 2 | int | Soft readout window radius. |
| TP6_SOFT_READOUT_TAU | gauss_tau | float | Soft readout tau. |
| TP6_PTR_LOCK | 0 | flag | Lock pointer to fixed value. |
| TP6_PTR_LOCK_VALUE | 0.5 | float | Lock value in [0,1] of ring. |

## Pointer velocity governor

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_PTR_VEL | 0 | flag | Enable velocity governor. |
| TP6_PTR_VEL_DECAY | 0.9 | float | Velocity EMA decay. |
| TP6_PTR_VEL_CAP | 0.5 | float | Max velocity. |
| TP6_PTR_VEL_SCALE | 1.0 | float | Torque scale. |

## Pointer update cadence

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_PTR_UPDATE_EVERY | 1 | int | Update pointer every N steps. |
| TP6_PTR_UPDATE_AUTO | 0 | flag | Auto-adjust cadence by flip rate. |
| TP6_PTR_UPDATE_MIN | 1 | int | Minimum cadence. |
| TP6_PTR_UPDATE_MAX | 16 | int | Maximum cadence. |
| TP6_PTR_UPDATE_EVERY_STEP | 20 | int | Auto cadence update interval. |
| TP6_PTR_UPDATE_TARGET_FLIP | 0.2 | float | Target flip rate for auto cadence. |
| TP6_PTR_UPDATE_EMA | 0.9 | float | EMA for flip rate. |
| TP6_PTR_UPDATE_GOV | 0 | flag | Cadence governor (grad+flip+loss). |
| TP6_PTR_UPDATE_GOV_WARMUP | ptr_warmup_steps | int | Governor warmup steps. |
| TP6_PTR_UPDATE_GOV_GRAD_HIGH | 45.0 | float | Grad high threshold. |
| TP6_PTR_UPDATE_GOV_GRAD_LOW | 2.0 | float | Grad low threshold. |
| TP6_PTR_UPDATE_GOV_LOSS_FLAT | 0.001 | float | Loss flat band. |
| TP6_PTR_UPDATE_GOV_LOSS_SPIKE | 0.1 | float | Loss spike threshold. |
| TP6_PTR_UPDATE_GOV_STEP_UP | 0.5 | float | Governor step up (slower). |
| TP6_PTR_UPDATE_GOV_STEP_DOWN | 0.2 | float | Governor step down (faster). |
| TP6_PTR_GATE_MODE | none | str | Pointer gating mode (`none` or `steps`). |
| TP6_PTR_GATE_STEPS | "" | str | Comma list of allowed update steps. |
| TP6_PTR_SOFT_GATE | 0 | flag | Learn gate to scale pointer delta. |
| TP6_PTR_JUMP_DISABLED | 0 | flag | Disable jumps entirely. |
| TP6_PTR_JUMP_CAP | 1.0 | float | Clamp jump probability. |

## Thermostat / panic

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_THERMO | 0 | flag | Enable thermostat (flip-rate control). |
| TP6_THERMO_EVERY | 20 | int | Thermostat interval. |
| TP6_THERMO_TARGET_FLIP | 0.2 | float | Target flip rate. |
| TP6_THERMO_EMA | 0.9 | float | EMA for flip rate. |
| TP6_THERMO_INERTIA_STEP | 0.05 | float | Inertia step size. |
| TP6_THERMO_DEADZONE_STEP | 0.02 | float | Deadzone step size. |
| TP6_THERMO_WALK_STEP | 0.02 | float | Walk step size. |
| TP6_THERMO_INERTIA_MIN | 0.0 | float | Inertia lower bound. |
| TP6_THERMO_INERTIA_MAX | 0.95 | float | Inertia upper bound. |
| TP6_THERMO_DEADZONE_MIN | 0.0 | float | Deadzone lower bound. |
| TP6_THERMO_DEADZONE_MAX | 0.5 | float | Deadzone upper bound. |
| TP6_THERMO_WALK_MIN | 0.0 | float | Walk lower bound. |
| TP6_THERMO_WALK_MAX | 0.3 | float | Walk upper bound. |
| TP6_PANIC | 0 | flag | Enable panic reflex. |
| TP6_PANIC_THRESHOLD | 1.5 | float | Loss threshold for panic. |
| TP6_PANIC_BETA | 0.9 | float | Panic EMA beta. |
| TP6_PANIC_RECOVERY | 0.01 | float | Panic recovery rate. |
| TP6_PANIC_INERTIA_LOW | 0.1 | float | Inertia when panicking. |
| TP6_PANIC_INERTIA_HIGH | 0.95 | float | Inertia ceiling. |
| TP6_PANIC_WALK_MAX | 0.2 | float | Walk max when panicking. |

## Activation & Möbius

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_ACT | c19 | str | Activation function name. |
| TP6_C13_P | 2.0 | float | C13 parameter (if used). |
| TP6_MOBIUS | 0 | flag | Enable Möbius phase embedding. |
| TP6_MOBIUS_EMB | 0.1 | float | Möbius embedding scale. |

## Precision & debug

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_PRECISION | fp32 | str | Model precision (`fp32`, `fp16`, `bf16`, `amp`). |
| TP6_PTR_DTYPE | fp32 | str | Pointer dtype (`fp32` recommended). |
| TP6_DEBUG_NAN | 0 | flag | NaN guard (raises if enabled). |
| TP6_DEBUG_STATS | 0 | flag | Log debug stats each step. |
| TP6_DEBUG_EVERY | 0 | int | Debug stat interval. |
| TP6_MI_SHUFFLE | 0 | flag | Shuffle MI labels during eval. |

## Loop metrics

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_STATE_LOOP_METRICS | 0 | flag | Track state loop modes. |
| TP6_STATE_LOOP_EVERY | 1 | int | Loop metric interval. |
| TP6_STATE_LOOP_SAMPLES | 0 | int | Samples to track (0 = all). |
| TP6_STATE_LOOP_DIM | 16 | int | Mode dimension for loop metrics. |

## Optimizer safety

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_GRAD_CLIP | 0.0 | float | Gradient clip norm. |
| TP6_STATE_CLIP | 0.0 | float | Clamp state values after writes. |
| TP6_STATE_DECAY | 1.0 | float | State decay per step. |
| TP6_UPDATE_SCALE | 1.0 | float | Write amplitude scale (verticality). |

## Checkpoint / resume

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_CKPT | checkpoint.pt | str | Checkpoint file path. |
| TP6_SAVE_EVERY | 100 | int | Save checkpoint every N steps. |
| TP6_RESUME | 0 | flag | Resume from checkpoint. |

## Run modes

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_MODE | train | str | Run mode (`train`, `evolution`, `lockout`). |
| TP6_PHASE_A_STEPS | 50 | int | Lockout test phase A steps. |
| TP6_PHASE_B_STEPS | 50 | int | Lockout test phase B steps. |

## Evolution mode

| Env var | Default | Type | Purpose |
| --- | --- | --- | --- |
| TP6_EVO_POP | 6 | int | Evolution population size. |
| TP6_EVO_GENS | 3 | int | Number of generations (0 = infinite). |
| TP6_EVO_STEPS | 100 | int | Steps per individual. |
| TP6_EVO_MUT_STD | 0.02 | float | Mutation std. |
| TP6_EVO_POINTER_ONLY | 0 | flag | Mutate pointer params only. |
| TP6_EVO_CKPT_EVERY | 1 | int | Checkpoint every N gens. |
| TP6_EVO_RESUME | 0 | flag | Resume evolution from latest. |
| TP6_EVO_CKPT_INDIV | 1 | flag | Save per-individual checkpoints. |
| TP6_EVO_PROGRESS | 1 | flag | Log per-individual progress. |




