# PRIME C-19 - Phase-Recurring Infinite Manifold Engine (Project David)
by Daniel Kenessy

[![Status: Experimental](https://img.shields.io/badge/Status-Research%20Preview-blue.svg)]()
[![Architecture: Recurrent](https://img.shields.io/badge/Arch-Manifold%20RNN-purple.svg)]()

> "Solving the Gradient Explosion on Circular Manifolds."

**Status:** PRE-ALPHA (research prototype). This is a proof-of-concept published early
to establish prior art. It is **not** production-ready and is **not** expected to
work end-to-end yet. Expect breaking changes, unstable results, and incomplete
components.

**Last updated:** 2026-01-17 (local time)

PRIME C-19 is a reference implementation of a recurrent neural memory architecture
designed to navigate a continuous 1D circular manifold (ring buffer). It introduces
topological and numerical fixes that stabilize gradient descent on closed loops,
eliminating the "boundary teleportation" glitch found in traditional pointer networks.

---

## The Core Problem: "The Rubber Wall"
In standard circular memory architectures, training a neural pointer to cross the
boundary (e.g., bin 2047 -> 0) fails catastrophically.
- Linear interpolation sees a jump of 2047 units instead of 1 unit.
- Result: gradients explode, causing the optimizer to freeze the pointer (the statue)
  or randomize it (the teleport).

## The Solution: Architecture Overview
PRIME C-19 patches the topology issue using three specific mechanisms:

### 1) Shortest-Arc Interpolation (Topology)
We replace standard linear deltas with a modular distance:

Delta = ((P_target - P_current + N/2) mod N) - N/2

This forces error signals to flow through the shortest bridge across the ring.

### 2) Fractional Gaussian Kernels (Gradients)
Discrete pointers have zero gradients between steps. PRIME C-19 uses fractional
read/write heads (e.g., index 10.4) with Gaussian/VonMises kernels to keep a
continuous gradient signal.

Engineering note: pointer mechanics are forced to FP32. FP16/AMP rounds the
micro-gradients to zero and paralyzes learning.

### 3) Mobius Phase Flip (Capacity)
To expand logical capacity without more VRAM, the pointer tracks a logical range
[0, 2N). When the pointer crosses the logical horizon, the retrieved vector is
multiplied by -1. This enables "anti-features" in the same physical ring.

---

## Comparison to Standard Methods

| Feature | Transformers | Standard RNNs | Neural Turing Machines | PRIME C-19 |
| :--- | :--- | :--- | :--- | :--- |
| Context Cost | O(N^2) | O(N) | O(N) | O(1) local kernel |
| Topology | Flat sequence | Flat sequence | Linear tape | Circular manifold |
| Boundary | N/A | N/A | Hard boundary | Continuous loop |
| Stability | High | High | Low | High (stabilized) |

---

## Activation Spotlight (C-19)

**Default activation:** `C-19` (`TP6_ACT=c19`)

Candidate-19 is the project's custom activation used to stabilize the pointer
pipeline while preserving continuous gradients. You can override it per run:
`TP6_ACT=identity` (or `tanh`, `silu`, etc.).

## License (Noncommercial)

This project is licensed under **PolyForm Noncommercial 1.0.0**.
You may use this code for **noncommercial** purposes (research, education, hobby).
**Commercial use requires a separate written license.**

Commercial licensing contact:
`kenessy.dani@gmail.com`

See:
- `LICENSE` (full PolyForm Noncommercial 1.0.0 text)
- `COMMERCIAL_LICENSE.md` (commercial licensing note)
- `DEFENSIVE_PUBLICATION.md` (public disclosure for prior art)
- `CITATION.cff` (citation metadata)
- `NOTICE` (copyright + license notice)

## Recent Updates

See `CHANGELOG.md` for full patch notes and timestamps. The latest A/B smoke run
results are recorded in `artifacts/ab_runs/proof_ab.csv` (short run; not a full benchmark).

## Quick Summary

At a high level:
- A continuous pointer moves on a circular memory ring.
- A kernel (Gaussian/VonMises) reads/writes a local neighborhood of the ring.
- Pointer movement is controlled by learned signals plus stabilizers.

For details, see `tournament_phase6.py`, `prime_c19/settings.py`, and `DEFENSIVE_PUBLICATION.md`.

## Optional Auto Controls

- `TP6_THERMO=1` enables flip-rate thermostat (adjusts inertia/deadzone/walk).
- `TP6_PTR_UPDATE_AUTO=1` enables auto pointer update cadence.
- `TP6_PANIC=1` enables loss-based panic reflex (reduces friction on loss spikes).
