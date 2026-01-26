# Control Board (TP6 / Phase 6.5)

This file is a **catalog of all runtime knobs** (static config + dynamic params)
found in the minimal TP6 codebase. It is meant as a quick reference when
refactoring, debugging, or running sweeps.

**Scope:**
- `prime_c19/settings.py` (`load_settings()` → `Settings` dataclass)
- `tournament_phase6.py` module-level overrides + kernel runtime params
- extracted control-loop helpers in `prime_c19/tp6/*`

> Note: Some symbols are defined but currently **unused** by the code paths in
> this minimal zip. Those are marked **(unused)**.

---

## Legend

- **Env var**: environment variable name (string) read by `load_settings()` or `tournament_phase6.py`.
- **Setting**: `Settings` field name (Python).
- **Type**: expected type after parsing.
- **Static**: read once at startup.
- **Dynamic param**: mutated during training or recomputed per frame.

---

## 1) Settings: `prime_c19/settings.py` (startup config)

These are read in `load_settings()` and surfaced as the `Settings` dataclass.

### Paths / environment

| Setting | Env var | Type | Default | Notes |
|---|---|---:|---:|---|
| `root` | `VAR_PROJECT_ROOT` | str | `os.getcwd()` | project root |
| `data_dir` | `TP6_DATA_DIR` | str | `<root>/data` | dataset cache dir |
| `log_path` | `TP6_LOG_PATH` | str | `<root>/tp6_run.log` | log file |
| `seed` | `TP6_SEED` | int | 1337 | RNG seed |
| `device` | `TP6_DEVICE` | str | `"cuda" if torch.cuda.is_available() else "cpu"` | device selector |
| `offline_only` | `TP6_OFFLINE_ONLY` | bool (0/1) | 1 | forbid downloads/network |
| `audio_backend` | `TP6_AUDIO_BACKEND` | str | "" | passed to `torchaudio.set_audio_backend()` |

### Budget / eval

| Setting | Env var | Type | Default | Notes |
|---|---|---:|---:|---|
| `max_samples` | `TP6_MAX_SAMPLES` | int | 6000 | training sample cap (data subset) |
| `eval_samples` | `TP6_EVAL_SAMPLES` | int | 1024 | eval subset size |
| `eval_split` | `TP6_EVAL_SPLIT` | float | 0.2 | eval split fraction |
| `eval_ptr_deterministic` | `TP6_EVAL_PTR_DETERMINISTIC` | bool | 0 | eval pointer determinism toggle |
| `batch_size` | `TP6_BATCH` | int | 32 | training batch size |
| `lr` | `TP6_LR` | float | 1e-3 | optimizer learning rate |
| `wall_clock_seconds` | `TP6_WALL` | float | 600 | wall clock budget |
| `max_steps` | `TP6_MAX_STEPS` | int | 0 | 0 means unlimited unless overridden elsewhere |
| `ignore_max_steps` | `TP6_IGNORE_MAX_STEPS` | bool | 0 | ignore max_steps stop condition |
| `ignore_wall_clock` | `TP6_IGNORE_WALL_CLOCK` | bool | 0 | ignore wall clock stop condition |
| `heartbeat_steps` | `TP6_HEARTBEAT` | int | 50 | logging cadence in steps |
| `heartbeat_secs` | `TP6_HEARTBEAT_SECS` | float | 0 | logging cadence in seconds (0 disables) |
| `live_trace_every` | `TP6_LIVE_TRACE_EVERY` | int | 0 | optional live trace logger cadence |
| `satiety_thresh` | `TP6_SATIETY_THRESH` | float | 0.05 | exit criterion for satiety (see model) |

### Core model shape

| Setting | Env var | Type | Default | Notes |
|---|---|---:|---:|---|
| `ring_len` | `TP6_RING_LEN` | int | 256 | ring positions |
| `slot_dim` | `TP6_SLOT_DIM` | int | 576 | slot feature dim |
| `ptr_dtype` | `TP6_PTR_DTYPE` | str | "float" | pointer tensor dtype hint |
| `ptr_param_stride` | `TP6_PTR_PARAM_STRIDE` | int | 4 | stride for pointer params |
| `gauss_k` | `TP6_GAUSS_K` | int | 3 | gaussian stencil radius |
| `lambda_move` | `TP6_LAMBDA_MOVE` | float | 0.0 | movement penalty weight |
| `state_loop_dim` | `TP6_STATE_LOOP_DIM` | int | 256 | MLP loop dim |
| `strat_dim` | `TP6_STRAT_DIM` | int | 0 | extra strategy vector dim |
| `pointer_hist_bins` | `TP6_POINTER_HIST_BINS` | int | 64 | pointer histogram bins |

### Pointer / AGC / cadence / thermostat

| Setting | Env var | Type | Default | Notes |
|---|---|---:|---:|---|
| `update_scale` | `TP6_UPDATE_SCALE` | float | 1.0 | initial update scale (AGC controls this) |
| `agc_scale_min` | `TP6_AGC_SCALE_MIN` | float | 1e-4 | minimum AGC scale |
| `agc_scale_max` | `TP6_AGC_SCALE_MAX` | float | 1.0 | maximum AGC scale |
| `agc_scale_max_min` | `TP6_AGC_SCALE_MAX_MIN` | float | 0.001 | plateau-decay floor |
| `agc_scale_max_decay` | `TP6_AGC_SCALE_MAX_DECAY` | float | 0.95 | plateau decay factor |
| `agc_plateau_window` | `TP6_PLATEAU_WINDOW` | int | 0 | plateau window length |
| `agc_plateau_min_steps` | `TP6_PLATEAU_MIN_STEPS` | int | 200 | plateau detection warmup |
| `agc_plateau_std` | `TP6_PLATEAU_STD` | float | 0.001 | plateau std threshold |
| `theta_ptr_dim` | `TP6_THETA_PTR_DIM` | int | 16 | pointer param embed dim |
| `ptr_update_every` | `TP6_PTR_UPDATE_EVERY` | int | 1 | pointer update cadence (steps per update) |
| `ptr_update_auto` | `TP6_PTR_UPDATE_AUTO` | bool | 0 | model-internal cadence governor |
| `ptr_update_every_step` | `TP6_PTR_UPDATE_EVERY_STEP` | int | 10 | model-internal cadence adjust step |
| `ptr_update_min` | `TP6_PTR_UPDATE_MIN` | int | 1 | cadence lower bound |
| `ptr_update_max` | `TP6_PTR_UPDATE_MAX` | int | 16 | cadence upper bound |
| `ptr_update_ema` | `TP6_PTR_UPDATE_EMA` | float | 0.98 | cadence governor EMA |
| `ptr_update_target_flip` | `TP6_PTR_UPDATE_TARGET_FLIP` | float | 0.1 | cadence target flip rate |
| `ptr_update_gov` | `TP6_PTR_UPDATE_GOV` | bool | 0 | external cadence governor (training loop) |
| `ptr_update_gov_warmup` | `TP6_PTR_UPDATE_GOV_WARMUP` | int | 200 | governor warmup steps |
| `ptr_update_gov_step_up` | `TP6_PTR_UPDATE_GOV_STEP_UP` | float | 1.0 | governor slow-down step |
| `ptr_update_gov_step_down` | `TP6_PTR_UPDATE_GOV_STEP_DOWN` | float | 1.0 | governor speed-up step |
| `ptr_update_gov_grad_high` | `TP6_PTR_UPDATE_GOV_GRAD_HIGH` | float | 1.0 | high-grad slow-down threshold |
| `ptr_update_gov_grad_low` | `TP6_PTR_UPDATE_GOV_GRAD_LOW` | float | 0.05 | low-grad laminar threshold |
| `ptr_update_gov_loss_flat` | `TP6_PTR_UPDATE_GOV_LOSS_FLAT` | float | 0.0005 | laminar loss delta threshold |
| `ptr_update_gov_loss_spike` | `TP6_PTR_UPDATE_GOV_LOSS_SPIKE` | float | 0.02 | spike slow-down threshold |
| `ptr_inertia` | `TP6_PTR_INERTIA` | float | 0.5 | base pointer inertia |
| `ptr_deadzone` | `TP6_PTR_DEADZONE` | float | 0.25 | base pointer deadzone |
| `ptr_walk_prob` | `PARAM_POINTER_FORWARD_STEP_PROB` | float | 0.1 | base pointer random walk probability |
| `ptr_vel` | `TP6_PTR_VEL` | bool | 0 | velocity governor toggle |
| `ptr_vel_full` | `TP6_PTR_VEL_FULL` | float | 0.5 | velocity “full scale” for inertia auto |
| `inertia_ema` | `TP6_INERTIA_EMA` | float | 0.98 | inertia-auto EMA |
| `inertia_auto` | `TP6_INERTIA_AUTO` | bool | 0 | enable inertia auto-control |
| `inertia_min` | `TP6_INERTIA_MIN` | float | 0.0 | inertia auto min |
| `inertia_max` | `TP6_INERTIA_MAX` | float | 1.0 | inertia auto max |
| `panic_enabled` | `TP6_PANIC` | bool | 0 | enable panic reflex |
| `panic_beta` | `TP6_PANIC_BETA` | float | 0.98 | panic EMA beta |
| `panic_threshold` | `TP6_PANIC_THRESH` | float | 1.5 | panic threshold |
| `panic_recovery` | `TP6_PANIC_RECOVERY` | float | 0.01 | recovery rate |
| `panic_inertia_low` | `TP6_PANIC_INERTIA_LOW` | float | 0.0 | panic inertia lower |
| `panic_inertia_high` | `TP6_PANIC_INERTIA_HIGH` | float | 1.0 | panic inertia upper |
| `panic_walk_max` | `TP6_PANIC_WALK_MAX` | float | 0.4 | panic walk prob upper |
| `thermo_enabled` | `TP6_THERMO` | bool | 0 | enable thermostat |
| `thermo_every` | `TP6_THERMO_EVERY` | int | 50 | thermostat update interval |
| `thermo_ema` | `TP6_THERMO_EMA` | float | 0.98 | flip-rate EMA |
| `thermo_target_flip` | `TP6_THERMO_TARGET` | float | 0.1 | target flip rate |
| `thermo_inertia_step` | `TP6_THERMO_INERTIA_STEP` | float | 0.02 | step size |
| `thermo_deadzone_step` | `TP6_THERMO_DEADZONE_STEP` | float | 0.01 | step size |
| `thermo_walk_step` | `TP6_THERMO_WALK_STEP` | float | 0.02 | step size |
| `thermo_inertia_min` | `TP6_THERMO_INERTIA_MIN` | float | 0.0 | clamp |
| `thermo_inertia_max` | `TP6_THERMO_INERTIA_MAX` | float | 1.0 | clamp |
| `thermo_deadzone_min` | `TP6_THERMO_DEADZONE_MIN` | float | 0.0 | clamp |
| `thermo_deadzone_max` | `TP6_THERMO_DEADZONE_MAX` | float | 1.0 | clamp |
| `thermo_walk_min` | `TP6_THERMO_WALK_MIN` | float | 0.0 | clamp |
| `thermo_walk_max` | `TP6_THERMO_WALK_MAX` | float | 0.4 | clamp |

### Runtime / IO / safety

| Setting | Env var | Type | Default | Notes |
|---|---|---:|---:|---|
| `disable_sync` | `TP6_DISABLE_SYNC` | bool | 0 | disables `torch.cuda.synchronize()` |
| `use_amp` | `TP6_AMP` | bool | 0 | mixed precision toggle |
| `grad_clip` | `TP6_GRAD_CLIP` | float | 0.0 | gradient clipping norm |
| `state_decay` | `TP6_STATE_DECAY` | float | 0.0 | state decay coefficient |
| `checkpoint_path` | `TP6_CKPT` | str | `<root>/tp6_ckpt.pt` | checkpoint file |
| `save_every_steps` | `TP6_SAVE_EVERY` | int | 0 | checkpoint cadence |
| `save_history` | `TP6_SAVE_HISTORY` | int | 3 | saved checkpoint history |
| `save_last_good` | `TP6_SAVE_LAST_GOOD` | bool | 1 | save last good model |
| `save_bad` | `TP6_SAVE_BAD` | bool | 0 | save bad checkpoints |
| `resume` | `TP6_RESUME` | bool | 0 | resume from checkpoint |
| `loss_keep` | `TP6_LOSS_KEEP` | int | 500 | how many losses to retain for plateau |
| `train_trace` | `TP6_TRAIN_TRACE` | bool | 0 | enable trace dumping |
| `train_trace_path` | `TP6_TRAIN_TRACE_PATH` | str | `<root>/tp6_train_trace.jsonl` | trace output |

### Synthetic “lockout” data + evolution harness

| Setting | Env var | Type | Default | Notes |
|---|---|---:|---:|---|
| `phase_a_steps` | `TP6_PHASE_A` | int | 2000 | lockout A steps |
| `phase_b_steps` | `TP6_PHASE_B` | int | 2000 | lockout B steps |
| `synth_len` | `TP6_SYNTH_LEN` | int | 8 | synthetic sequence length |
| `synth_shuffle` | `TP6_SYNTH_SHUFFLE` | bool | 0 | shuffle synthetic rows |
| `synth_mode` | `TP6_SYNTH` | str | "" | enable synth dataset + mode selector |
| `hand_min` | `TP6_HAND_MIN` | int | 1 | synthesis control |
| `assoc_keys` | `TP6_ASSOC_KEYS` | int | 8 | synthesis control |
| `assoc_pairs` | `TP6_ASSOC_PAIRS` | int | 16 | synthesis control |
| `evo_pop` | `TP6_EVO_POP` | int | 8 | evolution population |
| `evo_gens` | `TP6_EVO_GENS` | int | 2 | evolution generations |
| `evo_steps` | `TP6_EVO_STEPS` | int | 200 | evolution steps per individual |
| `evo_mut_std` | `TP6_EVO_MUT_STD` | float | 0.05 | mutation std |
| `evo_pointer_only` | `TP6_EVO_POINTER_ONLY` | bool | 1 | mutate pointer-only |
| `evo_checkpoint_every` | `TP6_EVO_CKPT_EVERY` | int | 0 | evo checkpoint cadence |
| `evo_resume` | `TP6_EVO_RESUME` | bool | 0 | resume evo |
| `evo_checkpoint_individual` | `TP6_EVO_CKPT_INDIV` | bool | 0 | per-individual checkpoints |
| `evo_progress` | `TP6_EVO_PROGRESS` | bool | 1 | progress print |

### Dataset head config

| Setting | Env var | Type | Default | Notes |
|---|---|---:|---:|---|
| `eval_dataset` | `TP6_EVAL_DATASET` | str | "mnist" | dataset selection |
| `num_classes` | `TP6_NUM_CLASSES` | int | 10 | classification output size |
| `input_dim` | `TP6_INPUT_DIM` | int | 784 | input feature dim |

---

## 2) Extra env vars & knobs read directly in `tournament_phase6.py`

These are **not** part of `Settings`, but they affect behavior.

| Env var | Type | Default | Effect |
|---|---:|---:|---|
| `TP6_EXPERT_HEADS` | int | 1 | number of expert output heads (router uses `ptr_int % N`) |
| `TP6_PRECISION` | str | unset | selects dtype (`bf16`/`fp16`/`fp32`) |
| `TP6_AGC_PLATEAU_WINDOW` | int | `Settings.agc_plateau_window` | plateau window override name alias |
| `TP6_SAVE_EVERY_STEPS` | int | `Settings.save_every_steps` | save cadence override name alias |
| `TP6_PTR_INERTIA_OVERRIDE` | float | unset | **manual override**; disables neural heads + thermostat/inertia auto writes |
| `TP6_FORCE_CADENCE_1` | bool | 0 | forces `ptr_update_every=1` in training loop |
| `TP6_SHARD` | bool | 0 | enable sharding |
| `TP6_SHARD_SIZE` | int | 0 | fixed shard size (0 disables) |
| `TP6_SHARD_ADAPT` | bool | 0 | adaptive shard sizing (VASC) |
| `TP6_SHARD_ADAPT_EVERY` | int | 10 | adaptation interval |
| `TP6_VASC_MIN_GROUP_RATIO` | float | 0.02 | shard size floor as ratio of batch |
| `TP6_TRACTION` | bool | 0 | enable traction metric (debug) |
| `TP6_INERTIA_DWELL` | bool | 0 | inertia auto uses dwell instead of velocity |
| `TP6_DWELL_THRESH` | float | 0.5 | dwell threshold for inertia auto |
| `TP6_WALK_PULSE` | bool | 0 | enable walk-pulse mode |
| `TP6_WALK_PULSE_EVERY` | int | 200 | pulse interval |
| `TP6_WALK_PULSE_STEPS` | int | 5 | pulse length |
| `TP6_WALK_PULSE_MAX` | float | 0.8 | pulse walk probability |
| `TP6_STATE_LOOP_EXTRA` | int | 0 | adds extra dims to state loop |
| `TP6_STATE_EXTRA` | int | 0 | adds extra state dims |
| `TP6_COLD_START_STEPS` | int | 0 | (unused) |
| `TP6_COLD_PTR_UPDATE_MIN` | int | 1 | (unused) |
| `TP6_COLD_UPDATE_SCALE` | float | 0.0005 | (unused) |
| `TP6_SPEED_GOV` | bool | 0 | (unused) |
| `TP6_SPEED_GOV_PEAK` | float | 0.5 | (unused) |
| `TP6_SPEED_GOV_TAU` | float | 0.5 | (unused) |
| `TP6_SCALE_UP` | float | 1.02 | AGC scale-up factor |
| `TP6_SCALE_DOWN` | float | 0.98 | AGC scale-down factor |
| `TP6_SCALE_WARMUP_STEPS` | int | 200 | warmup for AGC scale floor |
| `TP6_SCALE_WARMUP_INIT` | float | 1e-6 | warmup init floor |
| `TP6_PTR_UPDATE_GOV_VEL_HIGH` | float | 0.5 | cadence governor velocity short-circuit |

---

## 3) Dynamic params (mutated at runtime)

These are the “interactive” parameters—the ones that behave like the **neural
system of the pylot**, updated continuously during training and/or per frame.

### Pointer dynamics (per frame in `AbsoluteHallway.forward()`)

- `model.ptr_inertia` (dynamic): can be overridden by neural head outputs unless
  `TP6_PTR_INERTIA_OVERRIDE` is set.
- `model.ptr_deadzone` (dynamic): same.
- `model.ptr_walk_prob` (dynamic): same.
- `model.ptr_update_every` (dynamic): cadence value used by `update_allowed = (t % ptr_update_every)==0`.

### Control loops (per step in training loop)

- **AGC** updates:
  - `model.update_scale`
  - `model.agc_scale_max` / `model.agc_scale_cap`
- **Thermostat** updates:
  - `model.ptr_inertia`, `model.ptr_deadzone`, `model.ptr_walk_prob`
- **Inertia auto** updates:
  - `model.ptr_inertia`, `model.ptr_inertia_ema`
- **Panic reflex** updates:
  - `model.ptr_inertia`, `model.ptr_walk_prob` (when panic active)
- **Cadence governor** updates:
  - `model.ptr_update_every`

### Sharding / VASC (per step when enabled)

- `local_shard_size` (dynamic): shard size picked by `calculate_adaptive_vasc()`.
- `vasc_grad_ema` (dynamic): EMA of pointer grad norm.
- `vasc_max_dwell` (dynamic): max dwell observed.

---

## 4) Notes on (currently) unused settings/knobs

The following Settings fields exist but are not referenced by the TP6 kernel in
this minimal zip:

- `speed_gov_enabled`, `speed_gov_peak`, `speed_gov_tau` (**unused**)

The following env vars are parsed in `tournament_phase6.py` but not used:

- `TP6_COLD_START_STEPS`, `TP6_COLD_PTR_UPDATE_MIN`, `TP6_COLD_UPDATE_SCALE` (**unused**)
- `TP6_SPEED_GOV`, `TP6_SPEED_GOV_PEAK`, `TP6_SPEED_GOV_TAU` (**unused**)
- `TP6_SHARD_ADAPT_GRAD`, `TP6_SHARD_ADAPT_DWELL`, `TP6_SHARD_MIN_PER_SHARD` (**unused**) 

