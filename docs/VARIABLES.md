# Variables Cheat Sheet

Prefix rules:
- `VAR_`: runtime paths/logging. Does not change model behavior.
- `PARAM_`: learning-sensitive controls. Changing these can affect training dynamics.

## VAR_PROJECT_ROOT
Type: `str`  
Default: repo root (`Path(__file__).resolve().parents[1]`)  
Purpose: Overrides the base folder used for data and log paths.  
Behavior impact: None (paths/logging only).

## VAR_LOGGING_PATH
Type: `str`  
Default: `logs/current/tournament_phase6.log`  
Purpose: Overrides the main log file path.  
Behavior impact: None (logging only). Use a unique path per run to avoid mixed logs.

## VAR_TORCHAUDIO_BACKEND
Type: `str`  
Default: empty string  
Purpose: Torchaudio backend name (if torchaudio is available).  
Behavior impact: None unless audio loading is used.

## VAR_LOG_EVERY_N_STEPS
Type: `int`  
Default: `10`  
Purpose: Log every N steps.  
Behavior impact: None (logging only).

## VAR_LOG_EVERY_N_SECS
Type: `float`  
Default: `0.0`  
Purpose: Log every N seconds (0 = off).  
Behavior impact: None (logging only).

## VAR_LIVE_TRACE_PATH
Type: `str`  
Default: `traces/current/live_trace.json`  
Purpose: Live trace output path.  
Behavior impact: None (logging only).

## VAR_LIVE_TRACE_EVERY_N_STEPS
Type: `int`  
Default: `heartbeat steps`  
Purpose: Live trace interval in steps (defaults to `VAR_LOG_EVERY_N_STEPS`).  
Behavior impact: None (logging only).

## VAR_TRAINING_TRACE_ENABLED
Type: `flag`  
Default: `0`  
Purpose: Write train_steps trace JSONL.  
Behavior impact: None (logging only).

## VAR_TRAINING_TRACE_PATH
Type: `str`  
Default: `traces/current/train_steps_trace.jsonl`  
Purpose: Train trace output path.  
Behavior impact: None (logging only).

## VAR_LOSS_HISTORY_LEN
Type: `int`  
Default: `2000`  
Purpose: Loss history length used for slope and summaries.  
Behavior impact: None (logging/summary only).

## VAR_RUN_SEED
Type: `int`  
Default: `123`  
Purpose: RNG seed for Python/NumPy/Torch (run reproducibility).  
Behavior impact: None (repro only).

## VAR_COMPUTE_DEVICE
Type: `str`  
Default: `auto`  
Purpose: Device selection (`cuda` or `cpu`), falls back to available hardware.  
Behavior impact: None (hardware choice).

## PARAM_POINTER_FORWARD_STEP_PROB
Type: `float`  
Default: `0.2`  
Purpose: Probability of advancing the pointer by 1 bin when not jumping (`0.0` = stay, `1.0` = always step).  
Behavior impact: High (controls pointer drift/exploration).



