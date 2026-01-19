# Changelog

All dates are local time (YYYY-MM-DD).

## 2026-01-18
- Diagnostics: offline checkpoint loopiness probe (pointer flip/dwell/entropy + state-loop metrics) added in docs/STATE_LOOP_PROBE.md.
- Control: AGC now honors a scale cap (`TP6_SCALE_MAX`) and persists it in checkpoints.
- Control: AGC uses pre-clip pointer grad norm (brake visibility under clipping).
- Control: adaptive inertia added (`TP6_PTR_INERTIA_AUTO` + min/max/vel/EMA).
- Logging: pointer raw delta (`ptr_delta_raw_mean`) added to runtime logs and traces.
- Checkpoint: resume now restores update_scale and adaptive inertia state.
- Control: update_scale is now dynamic (AGC loop) to prevent runaway gradients.
- Control: cadence governor now uses pointer velocity (downshift on high motion).
- Control: AGC unscales grads when AMP is enabled to avoid false spikes.
- Logs: runtime now surfaces dynamic update_scale in training output.
- Tools: `tools/bench_small_prime.py` accepts pointer/governor overrides and dataset selection.

## 2026-01-17
- Fix: wrap-safe parameter interpolation in `_gather_params` (seam-safe control lookup).
- Fix: consolidate loss-based panic reflex into `tournament_phase6.py` (TP6_PANIC* env flags).
- Fix: panic overrides controls only when status is PANIC (LOCKED no longer clobbers thermostat/manual settings).
- Refactor: `interactive_teach.py` now uses shared `PanicReflex` from `tournament_phase6.py`.
- Refactor: centralize env parsing in `prime_c19/settings.py` and map into legacy globals.
- Refactor: move A/B artifacts into `artifacts/ab_runs/` and move legacy scripts/tools into `tools/`.
- Config: default activation switched to C-19 (`TP6_ACT=c19`).
- Config: default auto-checkpoint interval set to 100 steps (`TP6_SAVE_EVERY=100`).
- Evolution: per-generation checkpoints saved to `artifacts/evolution/` (evo_latest.pt + evo_gen_XXXXXX.pt).
- Evolution: optional resume from evo_latest.pt (`TP6_EVO_RESUME=1`).
- Evolution: pointer-only mutation option (`TP6_EVO_POINTER_ONLY=1`).
- Evolution: infinite generations when `TP6_EVO_GENS=0`.
- Logging: heartbeat status + grad_norm added to evolution training loop.
- Logging: optional per-step evolution checkpoints (`TP6_EVO_CKPT_INDIV=1`) and train trace JSONL (`VAR_TRAINING_TRACE_ENABLED=1`).
- Logging: expanded debug stats to include pointer entropy, satiety exits, and state-loop metrics in traces.
- Data: added `assoc_clean` synthetic mode (no-noise associative recall) with `TP6_ASSOC_KEYS/TP6_ASSOC_PAIRS`.
- Fix: readout now uses pre-update pointer (aligns read with write). Restores CE gradients on assoc_clean.
- Diagnostics: adversarial checks pass for ring-wrap lerp and kernel continuity.
- Docs: add update notes and patch references.
- Data: add `artifacts/ab_runs/proof_ab.csv` (short A/B smoke, 60 steps, seed 123) and updated summary JSONs.
- Fix: pointer math forced to FP32 (sub-bin stability under fp16/amp).
- Fix: satiety freeze masks state writes for inactive samples.
- Config: add `TP6_PTR_JUMP_CAP` (clamp jump probability) and `TP6_PTR_JUMP_DISABLED`.
- Config: add `TP6_PTR_SOFT_GATE` (pointer update gate for smoother control).
- Experiments: assoc_clean cadence sweep confirms update_every >= 8 yields stable learning on small task.
- Experiments: jump-cap alone does not fix update_every=1 collapse on small assoc_clean.
- Experiments: governor-enabled assoc_clean (len=8) reaches 1.00 acc at 800 steps (3 seeds).
- Config: add cadence governor controls (`TP6_PTR_UPDATE_GOV*`) combining grad-shock + flip-rate.
- Experiments: small synthetic bench (xor/two_moons/circles/spiral/sine) logged in `docs/bench_small_prime.md`.
- Docs: add bench charts (accuracy + sine MSE) embedded in README.
- Docs: add trigger-based publicity roadmap (`docs/roadmap_publicity.md`).

### A/B smoke (60 steps, seed 123)
- baseline: flip 0.9956 | dwell 1.0005 | acc 0.12695 | loss 2.3017
- stabilized: flip 0.4075 | dwell 5.2708 | acc 0.10938 | loss 2.8126



