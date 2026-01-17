# Assoc Clean Cadence Sweep (Micro)

Scope: no-noise associative recall (synthetic).

## Config (micro task)

```
TP6_SYNTH=1
TP6_SYNTH_MODE=assoc_clean
TP6_SYNTH_LEN=8
TP6_ASSOC_KEYS=2
TP6_ASSOC_PAIRS=1
TP6_MAX_SAMPLES=512
TP6_BATCH_SIZE=32
TP6_MAX_STEPS=200

TP6_PTR_SOFT_GATE=1
TP6_PTR_WALK_PROB=0.05
TP6_PTR_INERTIA=0.1
TP6_PTR_DEADZONE=0
TP6_PTR_NO_ROUND=1
TP6_SOFT_READOUT=1
TP6_LMOVE=0

TP6_PANIC_ENABLED=0
TP6_THERMO_ENABLED=0
```

## Cadence Sweep (update_every)

```
update_every  eval_acc  eval_loss
1             0.5430    0.6907
2             0.5430    0.6899
4             0.7070    0.6735
8             0.8047    0.4310
16            0.8047    0.4310
```

## Jump Cap Check (update_every=1)

```
cap=1.0  eval_acc 0.5430  eval_loss 0.6907
cap=0.2  eval_acc 0.5430  eval_loss 0.6915
```

## Takeaways

- There is a clear cadence knee around update_every >= 8 for this micro task.
- Jump-cap alone does not fix update_every=1 collapse.
- This is a small task; hard assoc_clean (len=32, keys=4, pairs=2) remains unstable and needs a dedicated sweep.
