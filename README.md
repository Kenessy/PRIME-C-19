# PRIME C-19 - Phase-Recurring Infinite Manifold Engine (Candidate 19 Activation)
by Daniel Kenessy

[![Status: Research Preview](https://img.shields.io/badge/Status-Research%20Preview-blue.svg)]()
[![Architecture: Recurrent](https://img.shields.io/badge/Arch-Manifold%20RNN-purple.svg)]()

> "Solving the Gradient Explosion on Circular Manifolds."

Status: PRE-ALPHA (research prototype). This is a proof-of-concept published early
for prior art. It is not production-ready and is not expected to work end-to-end.
Expect breaking changes, unstable results, and incomplete components.

Last updated: 2026-01-17 (local time)

PRIME C-19 is a reference implementation of a recurrent neural memory architecture
designed to navigate a continuous 1D circular manifold (ring buffer). It focuses
on topological and numerical fixes that stabilize gradient descent on closed loops,
eliminating the boundary teleportation glitch found in traditional pointer networks.

---

## Core Mechanisms

1) Shortest-Arc Interpolation (Topology)
Delta = ((P_target - P_current + N/2) mod N) - N/2
This forces error signals to flow through the shortest bridge across the ring.

2) Fractional Gaussian Kernels (Gradients)
Discrete pointers have zero gradients between steps. We use fractional read/write
heads (index 10.4) with truncated Gaussian kernels. The pointer path uses FP32
for stable sub-bin gradients.

3) Mobius Phase Flip (Capacity)
To maximize logical capacity, the architecture tracks a logical coordinate space
[0, 2N). Crossing N/2 flips the retrieved vector sign, allowing anti-features to
share physical memory without interference.

---

## Activation Spotlight: C-19 (Candidate 19)

PRIME C-19 defaults to the C-19 activation function. It is part of the core
research identity of this project and is referenced in the codename.

- Default: TP6_ACT=c19
- Alternatives: TP6_ACT=identity | tanh | silu | relu

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

## Quick Summary

- Pointer moves on a ring with shortest-arc interpolation (no seam teleports).
- Kernel read/write uses circular Gaussian or Von Mises weights.
- Stabilizers: inertia, deadzone, phantom hysteresis, velocity governor.
- Optional auto controls: TP6_THERMO, TP6_PTR_UPDATE_AUTO, TP6_PANIC.

---

## Known Issues (Active)

- assoc_clean (no-noise recall): CE gradients are zero in current diagnostics, so learning signal is blocked. Investigation in progress (see CHANGELOG.md).

---

## Evolution Mode (optional)

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

## Synthetic Modes (no download)

Set `TP6_SYNTH=1` to use synthetic data instead of MNIST.

- `TP6_SYNTH_MODE=markov0`: label is last bit.
- `TP6_SYNTH_MODE=markov0_flip`: label is inverse of last bit.
- `TP6_SYNTH_MODE=const0`: label always 0.
- `TP6_SYNTH_MODE=hand_kv`: load `data/hand_kv.jsonl`.
- `TP6_SYNTH_MODE=assoc_clean`: no-noise associative recall.
  - Configure with `TP6_ASSOC_KEYS` and `TP6_ASSOC_PAIRS`.

---

## Comparison to Standard Methods

| Feature | Transformers (Attention) | Standard RNNs (LSTM/GRU) | Neural Turing Machines | PRIME C-19 |
| --- | --- | --- | --- | --- |
| Context Cost | O(N^2) | O(N) | O(N) | O(1) (Local Kernel) |
| Topology | Flat Sequence | Flat Sequence | Linear Tape | Circular Manifold |
| Boundary | N/A | N/A | Hard Boundary | Continuous Loop |
| Stability | High | High | Low (Unstable) | High (Stabilized) |

---

## Project Structure (high level)

- tournament_phase6.py: main training and model code
- prime_c19/settings.py: env parsing and config mapping
- artifacts/ab_runs: A/B smoke artifacts and summary JSONs
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

- 2026-01-17: Evolution checkpoints + resume (`TP6_EVO_RESUME`, `evo_latest.pt`).
- 2026-01-17: Infinite evolution when `TP6_EVO_GENS=0`.
- 2026-01-17: Heartbeat logging added inside evolution training loop.
- 2026-01-17: Panic overrides controls only when status is PANIC.
- 2026-01-17: Activation default set to C-19; settings centralized in `prime_c19/settings.py`.

Full history: CHANGELOG.md
Smoke results: artifacts/ab_runs/proof_ab.csv
