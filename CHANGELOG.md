# Changelog

All dates are local time (YYYY-MM-DD).

## 2026-01-17
- Fix: wrap-safe parameter interpolation in `_gather_params` (seam-safe control lookup).
- Fix: consolidate loss-based panic reflex into `tournament_phase6.py` (TP6_PANIC* env flags).
- Refactor: `interactive_teach.py` now uses shared `PanicReflex` from `tournament_phase6.py`.
- Refactor: centralize env parsing in `prime_c19/settings.py` and map into legacy globals.
- Refactor: move A/B artifacts into `artifacts/ab_runs/` and move legacy scripts/tools into `tools/`.
- Config: default activation switched to C-19 (`TP6_ACT=c19`).
- Docs: add update notes and patch references.
- Data: add `artifacts/ab_runs/proof_ab.csv` (short A/B smoke, 60 steps, seed 123) and updated summary JSONs.

### A/B smoke (60 steps, seed 123)
- baseline: flip 0.9956 | dwell 1.0005 | acc 0.12695 | loss 2.3017
- stabilized: flip 0.4075 | dwell 5.2708 | acc 0.10938 | loss 2.8126
