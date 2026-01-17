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

## Hypothesis (Speculative)

**The Theory of Thought:** 
**The Principle of Topological Recursion (PTR)**

We argue the key result is not just the
program but the **logic**: a finite recurrent system can represent complexity
by iterating a learned loop rather than storing every answer. In this framing,
capacity is tied to **time/iteration**, not static memory size.

Operationally, PRIME C-19 treats memory as a circular manifold. Stability
(cadence) becomes a physical limiter: if updates are too fast, the system
cannot settle; if too slow, it stalls. We treat this as an engineering law,
not proven physics.

Evidence so far (bounded): the Unified Manifold Governor reaches **1.00 acc**
on micro `assoc_clean` (len=8, keys=2, pairs=1) at 800 steps across 3 seeds, and
the cadence knee occurs at `update_every >= 8`. This supports ALH as a working
hypothesis, not a general proof.

Full narrative (speculative): `docs/HYPOTHESIS.md`

---

## Roadmap Snapshot (Goals + Status)

| Horizon | Goal | Description | Status |
| --- | --- | --- | --- |
| Short | Micro assoc_clean stability | Governor + warmup hits 1.00 acc on len=8 task (3 seeds, 800 steps). | Done |
| Short | Cadence knee measured | update_every sweep shows learning only at >=8. | Done |
| Mid | Hard assoc_clean | len=32, keys=4, pairs=2 reaches >=0.80 acc across seeds. | In progress |
| Mid | Seq-MNIST baseline beat | Beat a standard GRU/LSTM baseline on comparable budget. | Planned |
| Long | Long-range benchmark | LRA/Path-X or Associative Recall at scale. | Planned |
| Long | External reproduction | Independent run confirms results. | Planned |

Checklist:
- [x] Seam-safe pointer interpolation (shortest-arc circ_lerp) implemented.
- [x] FP32 pointer math for fractional kernels (sub-bin gradients).
- [x] Cadence knee documented (update_every >= 8).
- [x] Unified Manifold Governor reaches 1.00 acc on micro assoc_clean.
- [ ] Hard assoc_clean >= 0.80 acc (len=32, keys=4, pairs=2).
- [ ] Seq-MNIST baseline beat on comparable budget.
- [ ] External reproduction confirmed.

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

Fixed cadence sweep (3 seeds, mean acc):

```
update_every  eval_acc_mean
1             0.5176
2             0.5208
4             0.5430
8             0.7129
16            0.7129
```

Governor run (Unified Manifold Governor, 3 seeds):
```
steps  eval_acc_mean  eval_loss_mean
400    0.8867         0.3557
800    1.0000         0.0737
```

Jump-cap alone does not fix "jump every step" failure:
- update_every=1: cap=0.2 and no-cap both ~0.52 acc (3 seeds).

Details: docs/ASSOC_CLEAN_SWEEP.md

Hard assoc_clean (len=32, keys=4, pairs=2) remains near chance:
```
c19  eval_acc 0.5039  eval_loss 0.7004
silu eval_acc 0.4912  eval_loss 0.7754
```

---

<details>
<summary><strong>Quick Start (Micro Assoc Clean)</strong></summary>

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

</details>

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

Math form (plain):

Where:
L = 6*pi, s = x/pi, n = floor(s), t = s - n, h = t(1 - t), sgn = (-1)^n
Default rho = 4.0

<details>
<summary><strong>Reference Equation (as implemented)</strong></summary>

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

</details>

---

Nickname note: we sometimes refer to a compute unit as a **Digital Diamond (DD)**.
This is branding only; the code still uses standard “neuron/unit” terminology.

---

## Small Synthetic Bench (C-19 vs ReLU vs SiLU)

Small, clean synthetic suite (XOR, Two Moons, Circles, Spiral, Sine Regression).
Results show C-19 matching or beating SiLU on the harder geometry/regression tasks
(spiral + sine), while keeping a lighter compute profile (no exp).

See full details: docs/bench_small_prime.md

<p align="center">
  <img alt="Small synthetic bench accuracy" src="https://raw.githubusercontent.com/Kenessy/PRIME-C-19/main/docs/bench_small_prime_acc.svg" width="720">
</p>

<p align="center">
  <img alt="Small synthetic bench sine regression" src="https://raw.githubusercontent.com/Kenessy/PRIME-C-19/main/docs/bench_small_prime_sine.svg" width="720">
</p>

<p align="center">
  <img alt="C19 activation equation" src="docs/c19_equation.svg" width="720">
</p>

---

## RUISS Score (Activation Bench)

RUISS is a relative activation benchmark that compares the total cost of a
candidate activation against a ReLU baseline:

```
ratio = baselineCost / totalCost
RUISS = 100 * ratio / (1 + ratio)   # higher is better
```

Current recorded run (RUISS-Brute v1):
- C-13 (precursor to C-19): RUISS = 59.3395
- ReLU baseline: RUISS = 50.0

Note: C-19 itself has not been re-scored in RUISS-Brute v1 yet.
Details: docs/RUISS.md

---

<details>
<summary><strong>Controls (Selected)</strong></summary>

Pointer dynamics:
- TP6_PTR_UPDATE_EVERY: cadence (key limiter)
- TP6_PTR_SOFT_GATE: soft gate for pointer updates
- TP6_PTR_JUMP_CAP: clamp jump probability
- TP6_PTR_JUMP_DISABLED: disable jump mix (walk only)
- TP6_PTR_WALK_PROB, TP6_PTR_INERTIA, TP6_PTR_DEADZONE

Automation:
- TP6_THERMO, TP6_PTR_UPDATE_AUTO, TP6_PANIC

</details>

---

## Known Issues (Active)

- assoc_clean (no-noise recall): gradients restored after pre-update readout fix,
  but hard settings (len=32, keys=4, pairs=2) remain unstable. Cadence sweep in progress.
- seq_mnist eval uses train-subset by default; do not treat as generalization
  unless you switch to a disjoint split.

---

<details>
<summary><strong>Evolution Mode (Optional)</strong></summary>

Use evolution to explore weight space with short training bursts.

- Enable: TP6_MODE=evolution
- Population: TP6_EVO_POP (default 6)
- Generations: TP6_EVO_GENS (set 0 for infinite)
- Steps per individual: TP6_EVO_STEPS
- Mutation scale: TP6_EVO_MUT_STD
- Pointer-only mutations: TP6_EVO_POINTER_ONLY=1
- Checkpoints: TP6_EVO_CKPT_EVERY (per-gen) and evo_latest.pt
- Resume: TP6_EVO_RESUME=1 (seed new population from evo_latest.pt)

</details>

---

<details>
<summary><strong>Synthetic Modes (No Download)</strong></summary>

Set TP6_SYNTH=1 to use synthetic data instead of MNIST.

- TP6_SYNTH_MODE=markov0: label is last bit.
- TP6_SYNTH_MODE=markov0_flip: label is inverse of last bit.
- TP6_SYNTH_MODE=const0: label always 0.
- TP6_SYNTH_MODE=hand_kv: load data/hand_kv.jsonl.
- TP6_SYNTH_MODE=assoc_clean: no-noise associative recall.
  - Configure with TP6_ASSOC_KEYS and TP6_ASSOC_PAIRS.

</details>

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

<details>
<summary><strong>Latest Patches</strong></summary>

- 2026-01-17: Evolution checkpoints + resume (TP6_EVO_RESUME, evo_latest.pt).
- 2026-01-17: Infinite evolution when TP6_EVO_GENS=0.
- 2026-01-17: Heartbeat logging added inside evolution training loop.
- 2026-01-17: Panic overrides controls only when status is PANIC.
- 2026-01-17: Activation default set to C-19; settings centralized in prime_c19/settings.py.
- 2026-01-17: Pointer math forced to FP32 (sub-bin stability).
- 2026-01-17: Satiety freeze masks state writes for inactive samples.
- 2026-01-17: Optional soft gate for pointer updates (TP6_PTR_SOFT_GATE=1).
- 2026-01-17: Optional jump cap + jump disable (TP6_PTR_JUMP_CAP, TP6_PTR_JUMP_DISABLED).
- 2026-01-17: Added small synthetic bench (xor/two_moons/circles/spiral/sine) + results.
  See docs/bench_small_prime.md.
- 2026-01-17: Publicity roadmap added (docs/roadmap_publicity.md).

Full history: CHANGELOG.md
Smoke results: artifacts/ab_runs/proof_ab.csv

</details>

---

<details>
<summary><strong>Future Research (Speculative)</strong></summary>

These are ideas we have not implemented yet. They are recorded for prior art
only and should not be treated as validated results.

- Hyperbolic bundle family: seam-free double-cover or holonomy-bit base, a
  hyperbolic scale axis, structure-preserving/geodesic updates (rotor or
  symplectic), and laminarized jumps. High potential, full redesign
  (not implemented).
- Post-jump momentum damping: apply a short cooldown to pointer velocity or
  jump probability for tau steps after a jump to reduce turbulence. This is a
  small, testable idea we may prototype next.

</details>

