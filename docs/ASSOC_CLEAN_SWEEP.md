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
PARAM_POINTER_FORWARD_STEP_PROB=0.05
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
1             0.5176    0.6922  (mean, 3 seeds)
2             0.5208    0.6925  (mean, 3 seeds)
4             0.5430    0.6829  (mean, 3 seeds)
8             0.7129    0.5466  (mean, 3 seeds)
16            0.7129    0.5466  (mean, 3 seeds)
```

## Jump Cap Check (update_every=1)

```
cap=1.0  eval_acc 0.5430  eval_loss 0.6907
cap=0.2  eval_acc 0.5430  eval_loss 0.6915
```

## Governor Run (Unified Manifold Governor)

```
steps  eval_acc  eval_loss
400    0.8867    0.3557  (mean, 3 seeds)
800    1.0000    0.0737  (mean, 3 seeds)
```

## Takeaways

- There is a clear cadence knee around update_every >= 8 for this micro task.
- Jump-cap alone does not fix update_every=1 collapse.
- Governor + longer runs remove the plateau on the micro task.
- This is a small task; hard assoc_clean (len=32, keys=4, pairs=2) remains unstable and needs a dedicated sweep.

---

## Hard Assoc Clean (len=32, keys=4, pairs=2)

Settings:
```
TP6_SYNTH_LEN=32
TP6_ASSOC_KEYS=4
TP6_ASSOC_PAIRS=2
TP6_MAX_SAMPLES=1024
TP6_MAX_STEPS=400
TP6_PTR_UPDATE_EVERY=8
```

Activation A/B:
```
c19  eval_acc 0.5039  eval_loss 0.7004
silu eval_acc 0.4912  eval_loss 0.7754
```

Takeaway: both are near chance on the hard task; activation is not the primary limiter here.
