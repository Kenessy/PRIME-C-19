# Changelog

All dates are local time (YYYY-MM-DD).

## 2026-01-17
- Fix: wrap-safe parameter interpolation in `_gather_params` (seam-safe control lookup).
- Docs: add update notes and patch references.
- Data: add `proof_ab.csv` (short A/B smoke, 60 steps, seed 123) and updated summary JSONs.

### A/B smoke (60 steps, seed 123)
- baseline: flip 0.9956 | dwell 1.0005 | acc 0.12695 | loss 2.3017
- stabilized: flip 0.4075 | dwell 5.2708 | acc 0.10938 | loss 2.8126

