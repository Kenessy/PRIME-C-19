# PRIME C-19 - Phase-Recurring Infinite Manifold Engine using the Candidate 19 activation function
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

## Activation Spotlight: C-19 (Candidate 19)

PRIME C-19 defaults to the C-19 activation function. It is part of the core
research identity of this project and is referenced in the codename.

- Default: TP6_ACT=c19
- Alternatives: TP6_ACT=identity | tanh | silu | relu

---

## The Core Problem: The Rubber Wall

In standard circular memory architectures, training a neural pointer to cross the
boundary (e.g., bin 2047 -> 0) fails catastrophically.
- Linear interpolation sees a jump of 2047 units instead of 1 unit.
- Result: gradients explode, causing the optimizer to freeze the pointer (the statue)
  or randomize it (the teleport).

## The Solution: Architecture Overview

PRIME C-19 patches the topology issue using three specific mechanisms:

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

## Comparison to Standard Methods

| Feature | Transformers (Attention) | Standard RNNs (LSTM/GRU) | Neural Turing Machines | PRIME C-19 |
| --- | --- | --- | --- | --- |
| Context Cost | O(N^2) | O(N) | O(N) | O(1) (Local Kernel) |
| Topology | Flat Sequence | Flat Sequence | Linear Tape | Circular Manifold |
| Boundary | N/A | N/A | Hard Boundary | Continuous Loop |
| Stability | High | High | Low (Unstable) | High (Stabilized) |

---

## Quick Summary

- Pointer moves on a ring with shortest-arc interpolation (no seam teleports).
- Kernel read/write uses circular Gaussian or Von Mises weights.
- Stabilizers: inertia, deadzone, phantom hysteresis, velocity governor.
- Optional auto controls: TP6_THERMO, TP6_PTR_UPDATE_AUTO, TP6_PANIC.

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

## Recent Updates

See CHANGELOG.md for patch notes and timestamps. A/B smoke results are recorded in
artifacts/ab_runs/proof_ab.csv (short run, not a full benchmark).