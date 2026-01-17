# Changelog

All dates are local time (YYYY-MM-DD).

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
- Logging: optional per-step evolution checkpoints (`TP6_EVO_CKPT_INDIV=1`) and train trace JSONL (`TP6_TRAIN_TRACE=1`).
- Logging: expanded debug stats to include pointer entropy, satiety exits, and state-loop metrics in traces.
- Data: added `assoc_clean` synthetic mode (no-noise associative recall) with `TP6_ASSOC_KEYS/TP6_ASSOC_PAIRS`.
- Diagnostics: adversarial checks pass for ring-wrap lerp and kernel continuity; assoc_clean CE gradients are zero (learning signal blocked, under investigation).
- Docs: add update notes and patch references.
- Data: add `artifacts/ab_runs/proof_ab.csv` (short A/B smoke, 60 steps, seed 123) and updated summary JSONs.

### A/B smoke (60 steps, seed 123)
- baseline: flip 0.9956 | dwell 1.0005 | acc 0.12695 | loss 2.3017
- stabilized: flip 0.4075 | dwell 5.2708 | acc 0.10938 | loss 2.8126
