# STATE LOOP PROBE (Offline)

This is a CPU-only, offline probe intended to estimate "loopiness" in pointer
motion and state dynamics from saved checkpoints. It does not train and does not
measure task accuracy.

## Method (Summary)

- Load checkpoint weights on CPU only (GPU disabled).
- Run a short, fixed forward pass on a small seq_mnist batch.
- Collect the existing debug metrics (pointer flips, dwell, entropy, and
  state-loop stats) over 64 steps.
- Compare current checkpoint vs older evolution checkpoints.

## Results Snapshot

### Current checkpoint (slot_dim=64)

Source: `G:\AI\pilot_pulse\checkpoint.pt`

- ptr_flip_rate: 0.982
- ptr_pingpong_rate: 0.0098
- ptr_mean_dwell: 1.002
- ptr_max_dwell: 2
- ptr_delta_abs_mean: 59.7
- pointer_hist_entropy: 6.07
- state_loop_abab_rate: 0.0137
- state_loop_mean_dwell: 4.0
- state_loop_max_dwell: 58
- top pointer bins: (123:20), (126:19), (127:18), (124:18), (125:17)

### Older checkpoints (slot_dim=8)

Sources (sampled):
- `G:\AI\pilot_pulse\checkpoint_fresh.pt`
- `G:\AI\pilot_pulse\artifacts\evolution\step_ckpts\evo_0_1_step_0500.pt`
- `G:\AI\pilot_pulse\artifacts\evolution\step_ckpts\evo_0_2_step_0400.pt`
- `G:\AI\pilot_pulse\artifacts\evolution\step_ckpts\evo_0_2_step_0500.pt`

Observed ranges:
- ptr_flip_rate: ~0.984
- ptr_mean_dwell: ~1.0
- ptr_max_dwell: 1
- ptr_delta_abs_mean: ~486 to 497
- pointer_hist_entropy: ~6.74 to 6.83
- state_loop_abab_rate: ~0.002 to 0.010

## Interpretation (Tentative)

- Pointer jump magnitude is much lower in the slot_dim=64 checkpoint
  (ptr_delta_abs_mean ~60 vs ~490), suggesting more localized motion.
- Pointer histogram entropy is lower in slot_dim=64, consistent with a more
  concentrated address pattern.
- State-loop ABAB rates remain low but nonzero; long dwell bursts appear in the
  state-loop metrics (max_dwell up to ~58) in the slot_dim=64 probe.

## Caveats

- Checkpoints do not store pointer trajectories. This probe infers loopiness
  from a short, synthetic forward run only.
- These metrics are not a count of explicit "A-B-C" loop sequences; for that,
  we would need to log ptr_int sequences directly during a trace run.
- Results depend on the sampled batch and step window (64 steps).

## Next Steps (If Needed)

- Add a CPU-only tracer that records ptr_int per step and computes explicit
  cycle counts (AB, ABA, ABC, etc.) for a given checkpoint.
- Increase window length and repeat across multiple seeds for stability.
