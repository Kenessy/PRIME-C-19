# REPRODUCIBILITY

This project is a research prototype. Results are unstable and hardware
sensitive. Use this file to record your own environment details.

## Environment checklist

Record these in your notes or in a run log:
- OS, Python version
- torch version, CUDA version (if applicable)
- GPU model and driver
- CPU model

Quick commands (optional):
- python --version
- python -c "import torch; print(torch.__version__)"
- python -c "import torch; print(torch.version.cuda)"

## Minimal reproduction (micro assoc_clean)

From README (Windows PowerShell):

```
set TP6_SYNTH=1
set TP6_SYNTH_MODE=assoc_clean
set TP6_SYNTH_LEN=8
set TP6_ASSOC_KEYS=2
set TP6_ASSOC_PAIRS=1
set TP6_MAX_SAMPLES=512
set TP6_BATCH_SIZE=32
set TP6_MAX_STEPS=200

set TP6_PTR_SOFT_GATE=1
set PARAM_POINTER_FORWARD_STEP_PROB=0.05
set TP6_PTR_INERTIA=0.1
set TP6_PTR_DEADZONE=0
set TP6_PTR_NO_ROUND=1
set TP6_SOFT_READOUT=1
set TP6_LMOVE=0

set TP6_PANIC_ENABLED=0
set TP6_THERMO_ENABLED=0

python tournament_phase6.py
```

Expected behavior:
- Training runs without NaNs.
- Loss should show small downward drift.
- For longer runs (800 steps), cadence >= 8 is required to reach high accuracy.

## Notes

- Always set a seed if you need exact reproducibility:
  `set VAR_RUN_SEED=123`
- Logs default to `logs/current/tournament_phase6.log`.

