# GPT_CHECK_GUIDE.md

This file is a deep, code-accurate guide to the current VRAXION/PRIME-C19 kernel.
It is written for another engineer/LLM to audit and refine the code without missing
critical behaviors or hidden overrides.

Scope of this zip:
- VRAXION_INFINITE.py (runner/bootstrap)
- tournament_phase6.py (core model + training loop)
- prime_c19/settings.py (env parsing + defaults)
- prime_c19/smoke_checks.py (helpers)
- prime_c19/__init__.py

------------------------------------------------------------------------------
1) Execution overview (where the run starts)
------------------------------------------------------------------------------

Entry points:
- VRAXION_INFINITE.py: wrapper that sets environment variables (bootstrap),
  optionally runs auto-tune probes, then calls tournament_phase6.main() in a
  restart loop.
- tournament_phase6.py: contains model, synthetic data generator, and training
  loop. If TP6_SYNTH=1 it bypasses MNIST entirely and runs synth mode forever.

Default behavior in VRAXION_INFINITE.py:
1) bootstrap_env() sets env vars for a long-running synthetic run.
2) It clears the main log file if TP6_RESUME=0.
3) auto-tune is currently OFF (TP6_AUTO_TUNE=0).
4) It imports tournament_phase6 and calls main() repeatedly (restart on crash).

Key logs:
- VAR_LOGGING_PATH controls where logs go (prime_c19/settings.py).
- VRAXION_INFINITE currently sets: logs/expert_infinite.log

------------------------------------------------------------------------------
2) Model architecture (AbsoluteHallway)
------------------------------------------------------------------------------

The core model class is AbsoluteHallway in tournament_phase6.py.

High-level flow (AbsoluteHallway.forward):
1) Input projection: input -> input_proj -> GRUCell.
2) Pointer update:
   - Theta params per ring position (theta_ptr_reduced, theta_gate_reduced).
   - Jump score, stay/walk blend, and shortest-arc interpolation on the ring.
   - Inertia/deadzone/walk can be dynamic (thermostat/panic) or forced by env.
3) Ring write:
   - Compute kernel weights around pointer (Gaussian or von Mises).
   - scatter_add into ring state.
   - BF16 scatter_add fallback uses float32 to avoid CUDA bf16 crash on Windows.
4) Ring read:
   - Read from pointer position (soft readout optional).
   - Optional phantom pointer read.
5) Output head:
   - LocationExpertRouter (1 or N experts).
   - Deterministic routing: expert_id = pointer_address % num_experts.

Important architecture components:
- RING_LEN: number of ring slots.
- SLOT_DIM: size of each slot (hidden dim).
- LocationExpertRouter: deterministic expert routing based on ptr_int.
- Pointer motion: uses shortest-arc interpolation (wrap_delta, circ_lerp) on ring.
- Kernel read/write: Gaussian/vonmises window around pointer with fractional offsets.

------------------------------------------------------------------------------
3) LocationExpertRouter (Mixture-of-Experts)
------------------------------------------------------------------------------

File: tournament_phase6.py
Class: LocationExpertRouter

Key logic:
- If num_experts == 1: single Linear layer (backward-compatible).
- Else: ModuleList of Linear layers (one per expert).
- Routing: expert_indices = pointer_addresses % num_experts
- For each expert, mask the batch and apply its Linear layer.

Why it matters:
- Different pointers can specialize to different experts.
- Prevents interference from unrelated samples.

Env var:
- TP6_EXPERT_HEADS (default 1; VRAXION_INFINITE sets 16).

------------------------------------------------------------------------------
4) Synthetic data modes (assoc_clean / assoc_byte)
------------------------------------------------------------------------------

File: tournament_phase6.py, function get_seq_mnist_loader()

When TP6_SYNTH=1:
- synth_mode is SYNTH_MODE (from settings or env TP6_SYNTH_MODE).

assoc_clean:
- Key tokens are 2+key_id.
- Value tokens are -1 or -2.
- Query token is key; target is binary (0/1).
- num_classes = 2.

assoc_byte:
- Key tokens are 2+key_id.
- Value tokens are negative numbers: -1..-val_range
- Query token is key; target is value in [0, val_range-1].
- num_classes = val_range.

Synth env vars:
- TP6_SYNTH=1
- TP6_SYNTH_MODE=assoc_clean | assoc_byte | hand_kv
- TP6_SYNTH_LEN (sequence length)
- TP6_ASSOC_KEYS
- TP6_ASSOC_PAIRS
- TP6_ASSOC_VAL_RANGE (assoc_byte only)

------------------------------------------------------------------------------
5) Control loop and dynamic hyperparameters
------------------------------------------------------------------------------

Core control functions (tournament_phase6.py):

apply_update_agc(model, grad_norm, raw_delta=None, step=None)
- Warmup floor: SCALE_WARMUP_INIT -> AGC_SCALE_MIN over SCALE_WARMUP_STEPS.
- Scale updates:
  - if grad_norm < AGC_GRAD_LOW: scale *= AGC_SCALE_UP
  - if grad_norm > AGC_GRAD_HIGH: scale *= AGC_SCALE_DOWN
- Clamp: scale in [floor, cap], where cap is agc_scale_max/agc_scale_cap.
- Writes to model.update_scale and logs debug at step 0.

apply_thermostat(model, flip_rate, ema, focus=None, tension=None, raw_delta=None)
- Computes EMA of flip_rate.
- If TP6_PTR_INERTIA_OVERRIDE is set, returns EMA and does NOT mutate inertia/deadzone/walk.
- If focus/tension provided:
  - focus = clamp(dwell / max_dwell_limit)
  - tension = clamp(grad_norm / ema_grad_norm)
  - stuck = 1 / (1 + raw_delta)
  - drive = max(tension, 1 - focus, stuck)
  - target_inertia = lerp(THERMO_INERTIA_MIN, THERMO_INERTIA_MAX, focus*(1-tension))
  - target_deadzone = lerp(THERMO_DEADZONE_MIN, THERMO_DEADZONE_MAX, tension)
  - target_walk = lerp(THERMO_WALK_MIN, THERMO_WALK_MAX, drive)
  - blend = max(1e-3, 1-THERMO_EMA)
  - ptr_inertia/deadzone/walk are blended toward targets.
- If focus/tension not provided, fallback stepper:
  - adjust inertia/deadzone/walk by fixed step sizes based on flip EMA.

PanicReflex (loss-based):
- Keeps loss EMA.
- If loss spikes above threshold, reduces inertia and increases walk_prob.
- Gated in training loop: inertia/walk updates are skipped if TP6_PTR_INERTIA_OVERRIDE is set.

CadenceGovernor:
- Adjusts ptr_update_every based on grad_norm, flip_rate, and pointer velocity.
- If ptr velocity is too high, it forces lower cadence (more frequent updates).

Pointer dynamics in AbsoluteHallway.forward:
- ptr_float updated by jump/stay/walk, then inertia and deadzone applied.
- If PTR_VEL enabled, pointer velocity acts as a smoothing term.
- ptr_int derived from ptr_float for routing and expert selection.

------------------------------------------------------------------------------
6) VASC (Adaptive Sharding)
------------------------------------------------------------------------------

calculate_adaptive_vasc(batch_size, dwell, grad_norm, max_dwell_limit, ema_grad_norm, min_group_ratio)
- focus = dwell / max_dwell_limit
- tension = grad_norm / ema_grad_norm
- cohesion = clamp(focus - tension)
- ceiling = batch_size
- floor = max(1, ceiling * min_group_ratio)
- target_group_size = floor + (ceiling - floor) * cohesion
- raw_shards = ceiling / target_group_size
- choose shard_count nearest divisor of batch_size
- returns shard_count, group_size, focus, tension, cohesion

Shard env vars:
- TP6_SHARD_BATCH (enable sharding)
- TP6_SHARD_SIZE (fixed shard size when adaptive is off)
- TP6_SHARD_ADAPT (enable VASC)
- TP6_SHARD_ADAPT_EVERY (how often to update)
- TP6_VASC_MIN_GROUP_RATIO (default 0.02)

Logging:
Logs include shard=X/Y, traction, focus, tension, cohesion, experts count.

------------------------------------------------------------------------------
7) Precision and bf16
------------------------------------------------------------------------------

Precision defaults:
- settings.py chooses precision; if TP6_PRECISION not set and CUDA supports bf16,
  code forces DTYPE to bf16 and USE_AMP true.
- A known CUDA bf16 scatter_add issue is handled by doing scatter_add in float32,
  then casting back.

Env vars:
- TP6_PRECISION=bf16 or fp32
- TP6_DISABLE_SYNC=1 disables cuda.synchronize and empty_cache calls.

------------------------------------------------------------------------------
8) Logging and checkpoints
------------------------------------------------------------------------------

Logging:
- VAR_LOGGING_PATH sets log file path (settings.py).
- default is logs/current/tournament_phase6.log if VAR_LOGGING_PATH empty.

Checkpoints:
- TP6_CKPT sets checkpoint path.
- TP6_SAVE_EVERY_STEPS controls interval.
- Files include update_scale, ptr_inertia, agc_scale_max, and ground_speed_ema.

Infinite run:
- TP6_IGNORE_MAX_STEPS=1 and TP6_IGNORE_WALL_CLOCK=1 skip early exits.
- For probes: TP6_SYNTH_ONCE=1 + TP6_MAX_STEPS=5 (or N) to run once.

------------------------------------------------------------------------------
9) VRAXION_INFINITE bootstrap (current values)
------------------------------------------------------------------------------

The infinite runner hardcodes env vars for a stable run:
- TP6_EXPERT_HEADS = 16
- TP6_SYNTH_MODE = assoc_byte
- TP6_SYNTH_LEN = 512
- TP6_ASSOC_KEYS = 64
- TP6_ASSOC_PAIRS = 4
- TP6_BATCH_SIZE = 448 (fixed)
- TP6_UPDATE_SCALE = 0.05
- TP6_SCALE_MIN = 1e-4
- TP6_SCALE_WARMUP_STEPS = 100
- TP6_SCALE_WARMUP_INIT = 1e-6
- TP6_SCALE_MAX = 1.0
- TP6_THERMO = 1
- TP6_THERMO_EVERY = 5
- PARAM_POINTER_FORWARD_STEP_PROB = 0.1
- TP6_PTR_UPDATE_GOV = 1
- TP6_PTR_UPDATE_AUTO = 1
- TP6_PTR_UPDATE_EVERY = 1
- TP6_PTR_VEL = 0
- TP6_SHARD_ADAPT = 1
- TP6_SHARD_ADAPT_EVERY = 1
- TP6_TRACTION_LOG = 1
- TP6_STATE_DECAY = 1.0
- TP6_GRAD_CLIP = 0
- TP6_DISABLE_SYNC = 1
- TP6_SAVE_EVERY_STEPS = 100
- TP6_IGNORE_MAX_STEPS = 1
- TP6_IGNORE_WALL_CLOCK = 1

The main log path is set to:
- VAR_LOGGING_PATH = logs/expert_infinite.log

------------------------------------------------------------------------------
10) Where to change behavior (key knobs)
------------------------------------------------------------------------------

Capacity:
- RING_LEN, SLOT_DIM in settings.py (TP6_RING_LEN, TP6_SLOT_DIM envs).
- TP6_EXPERT_HEADS for expert count.

Pointer physics:
- TP6_PTR_INERTIA_OVERRIDE (locks inertia, disables thermostat mutations).
- TP6_PTR_WALK_PROB uses PARAM_POINTER_FORWARD_STEP_PROB.
- THERMO_* in settings.py to control dynamic inertia/deadzone/walk.

Learning speed:
- TP6_UPDATE_SCALE / TP6_SCALE_MIN / TP6_SCALE_MAX / TP6_SCALE_WARMUP_*.
- AGC_* thresholds and multipliers.

Sharding:
- TP6_SHARD_ADAPT (on/off).
- TP6_SHARD_ADAPT_EVERY (steps between updates).
- TP6_VASC_MIN_GROUP_RATIO (floor).

Cadence:
- TP6_PTR_UPDATE_GOV / TP6_PTR_UPDATE_AUTO / TP6_PTR_UPDATE_EVERY.

Precision:
- TP6_PRECISION, TP6_DISABLE_SYNC.

------------------------------------------------------------------------------
11) Quick verification runs
------------------------------------------------------------------------------

Minimal sanity (synthetic):
- TP6_SYNTH=1
- TP6_SYNTH_MODE=assoc_byte
- TP6_EXPERT_HEADS=16
- TP6_RESUME=0
- VAR_LOGGING_PATH=logs/expert_demo.log

Expect:
- logs show "experts=16" and "shard=..."
- loss should move off the random baseline (approx ln(256)=5.545).

------------------------------------------------------------------------------
12) Known intentional behaviors
------------------------------------------------------------------------------

1) TP6_PTR_INERTIA_OVERRIDE:
   If set, thermostat, panic, and auto inertia updates are gated.
   This is intentional to allow manual hard-locking.

2) TP6_SYNTH_ONCE:
   Used by auto-tune probes to run a single phase and exit.

3) BF16 scatter_add:
   scatter_add is done in float32 to prevent BF16 runtime crash on Windows.

------------------------------------------------------------------------------
End of guide.
