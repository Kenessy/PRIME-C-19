# PRIME C-19: Phase-Recurring Infinite Manifold Engine
by Daniel Kenessy

[![Status: Research Preview](https://img.shields.io/badge/Status-Research%20Preview-blue.svg)]()
[![Architecture: Recurrent](https://img.shields.io/badge/Arch-Manifold%20RNN-purple.svg)]()

Status: PRE-ALPHA (research prototype). This is a proof of concept published early
for prior art. It is not production-ready and is not expected to work end-to-end.
Expect breaking changes, unstable results, and incomplete components.

Last updated: 2026-01-17 (local time)

PRIME C-19 is a recurrent neural memory architecture that navigates a continuous
1D circular manifold (ring buffer). It focuses on topological and numerical fixes
that stabilize gradient descent on closed loops and remove seam teleportation.

---

## One-line Pitch

Shortest-arc pointer control + fractional read/write kernels + cadence-aware
updates to keep memory stable on a ring.

---

## Key Innovations (Current)

1) Shortest-Arc Interpolation (Topology)
Delta = ((P_target - P_current + N/2) mod N) - N/2
This forces error signals to flow through the shortest bridge across the ring.

2) Fractional Gaussian Kernels (Gradients)
Discrete pointers have zero gradients between steps. PRIME C-19 uses fractional
read/write heads with truncated Gaussian kernels. Pointer math is forced to FP32
for stable sub-bin gradients even under fp16/amp.

3) Mobius Phase Embedding (Capacity)
Optional continuous phase embedding over a logical [0, 2N) coordinate space.
This is a smooth helix (cos/sin phase), not a hard sign flip at wrap.

4) Cadence as a Physical Limit
Update cadence (PTR_UPDATE_EVERY) is an empirical limiter. Micro assoc_clean
shows a clear knee at update_every >= 8 (see Evidence below).

---

## Evidence Snapshot (Assoc Clean, Micro)

Task: assoc_clean (len=8, keys=2, pairs=1), soft gate ON, no panic/thermo.

```
update_every  eval_acc
1             0.5430
2             0.5430
4             0.7070
8             0.8047
16            0.8047
```

Jump-cap alone does not fix "jump every step" failure:
- update_every=1: cap=0.2 and no-cap both ~0.543 acc.

Details: docs/ASSOC_CLEAN_SWEEP.md

Hard assoc_clean (len=32, keys=4, pairs=2) remains near chance:
```
c19  eval_acc 0.5039  eval_loss 0.7004
silu eval_acc 0.4912  eval_loss 0.7754
```

---

## Quick Start (Micro Assoc Clean)

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
set TP6_PTR_WALK_PROB=0.05
set TP6_PTR_INERTIA=0.1
set TP6_PTR_DEADZONE=0
set TP6_PTR_NO_ROUND=1
set TP6_SOFT_READOUT=1
set TP6_LMOVE=0

set TP6_PANIC_ENABLED=0
set TP6_THERMO_ENABLED=0

python tournament_phase6.py
```

---

## Architecture Overview

```
input -> input_proj -> activation -> GRU -> ring state (scatter_add)
                                   -> head -> logits

pointer control:
  theta_ptr / theta_gate + jump_score -> jump p -> circular lerp -> ptr_float
```

Readout is aligned with the pre-update pointer (read and write are synchronized).

---

## Activation Spotlight: C-19 (Candidate 19)

PRIME C-19 defaults to the C-19 activation function. It is part of the core
research identity of this project and is referenced in the codename.

- Default: TP6_ACT=c19
- Alternatives: TP6_ACT=identity | tanh | silu | relu

Math form (rendered):

<p align="center">
  <img alt="C19 activation" src="https://latex.codecogs.com/svg.image?\\Large%20C_{19}(x)=\\begin{cases}x-L&x\\ge%20L\\\\x+L&x\\le-L\\\\\\pi\\,(sgn\\cdot%20h+\\rho%20h^2)&\\text{otherwise}\\end{cases}">
</p>

Where:
L = 6*pi, s = x/pi, n = floor(s), t = s - n, h = t(1 - t), sgn = (-1)^n
Default rho = 4.0

Reference equation (as implemented):

```
Let L = 6*pi
Let s = x / pi
Let n = floor(s)
Let t = s - n
Let h = t * (1 - t)
Let sgn = +1 if n is even else -1

C19(x) = x - L                      if x >=  L
       = x + L                      if x <= -L
       = pi * (sgn*h + rho*h*h)     otherwise

Default rho = 4.0
```

---

## Controls (Selected)

Pointer dynamics:
- TP6_PTR_UPDATE_EVERY: cadence (key limiter)
- TP6_PTR_SOFT_GATE: soft gate for pointer updates
- TP6_PTR_JUMP_CAP: clamp jump probability
- TP6_PTR_JUMP_DISABLED: disable jump mix (walk only)
- TP6_PTR_WALK_PROB, TP6_PTR_INERTIA, TP6_PTR_DEADZONE

Automation:
- TP6_THERMO, TP6_PTR_UPDATE_AUTO, TP6_PANIC

---

## Known Issues (Active)

- assoc_clean (no-noise recall): gradients restored after pre-update readout fix,
  but hard settings (len=32, keys=4, pairs=2) remain unstable. Cadence sweep in progress.
- seq_mnist eval uses train-subset by default; do not treat as generalization
  unless you switch to a disjoint split.

---

## Evolution Mode (Optional)

Use evolution to explore weight space with short training bursts.

- Enable: TP6_MODE=evolution
- Population: TP6_EVO_POP (default 6)
- Generations: TP6_EVO_GENS (set 0 for infinite)
- Steps per individual: TP6_EVO_STEPS
- Mutation scale: TP6_EVO_MUT_STD
- Pointer-only mutations: TP6_EVO_POINTER_ONLY=1
- Checkpoints: TP6_EVO_CKPT_EVERY (per-gen) and evo_latest.pt
- Resume: TP6_EVO_RESUME=1 (seed new population from evo_latest.pt)

---

## Synthetic Modes (No Download)

Set TP6_SYNTH=1 to use synthetic data instead of MNIST.

- TP6_SYNTH_MODE=markov0: label is last bit.
- TP6_SYNTH_MODE=markov0_flip: label is inverse of last bit.
- TP6_SYNTH_MODE=const0: label always 0.
- TP6_SYNTH_MODE=hand_kv: load data/hand_kv.jsonl.
- TP6_SYNTH_MODE=assoc_clean: no-noise associative recall.
  - Configure with TP6_ASSOC_KEYS and TP6_ASSOC_PAIRS.

---

## Project Structure (High Level)

- tournament_phase6.py: main training and model code
- prime_c19/settings.py: env parsing and config mapping
- artifacts/ab_runs: A/B smoke artifacts and summary CSVs
- tools/: GUI and interactive scripts
- docs/: notes and test summaries

---

## License (Noncommercial)

This project is licensed under PolyForm Noncommercial 1.0.0.
You may use this code for noncommercial purposes (research, education, hobby).
Commercial use requires a separate written license.

Commercial licensing contact:
kenessy.dani@gmail.com

See:
- LICENSE (full PolyForm Noncommercial 1.0.0 text)
- COMMERCIAL_LICENSE.md (commercial licensing note)
- DEFENSIVE_PUBLICATION.md (public disclosure for prior art)
- CITATION.cff (citation metadata)
- NOTICE (copyright + license notice)

---

## Latest Patches

- 2026-01-17: Evolution checkpoints + resume (TP6_EVO_RESUME, evo_latest.pt).
- 2026-01-17: Infinite evolution when TP6_EVO_GENS=0.
- 2026-01-17: Heartbeat logging added inside evolution training loop.
- 2026-01-17: Panic overrides controls only when status is PANIC.
- 2026-01-17: Activation default set to C-19; settings centralized in prime_c19/settings.py.
- 2026-01-17: Pointer math forced to FP32 (sub-bin stability).
- 2026-01-17: Satiety freeze masks state writes for inactive samples.
- 2026-01-17: Optional soft gate for pointer updates (TP6_PTR_SOFT_GATE=1).
- 2026-01-17: Optional jump cap + jump disable (TP6_PTR_JUMP_CAP, TP6_PTR_JUMP_DISABLED).

Full history: CHANGELOG.md
Smoke results: artifacts/ab_runs/proof_ab.csv

---

## Future Research (Speculative)

These are ideas we have not implemented yet. They are recorded for prior art
only and should not be treated as validated results.

- Hyperbolic bundle family: seam-free double-cover or holonomy-bit base, a
  hyperbolic scale axis, structure-preserving/geodesic updates (rotor or
  symplectic), and laminarized jumps. High potential, full redesign
  (not implemented).
- Post-jump momentum damping: apply a short cooldown to pointer velocity or
  jump probability for tau steps after a jump to reduce turbulence. This is a
  small, testable idea we may prototype next.
