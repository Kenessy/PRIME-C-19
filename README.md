# PRIME C-19 — Phase-Recurring Infinite Manifold Engine (Project David)

PRIME C-19: Phase-Recurring Infinite Manifold Engine

"Solving the Gradient Explosion on Circular Manifolds."

PRIME C-19 is a reference implementation of a recurrent neural memory architecture designed to navigate a continuous 1D circular manifold (ring buffer). It introduces a set of topological and numerical fixes that stabilize gradient descent on closed loops, eliminating the notorious "boundary teleportation" glitch found in traditional pointer networks.

⚡ The Core Problem: "The Rubber Wall"

In standard circular memory architectures, training a neural pointer to cross the boundary (e.g., Bin $2047 \to 0$) fails catastrophically.

Linear Interpolation: Sees a jump of 2047 units instead of 1 unit.

Result: Gradients explode, causing the optimizer to either freeze the pointer (The Statue) or randomize it (The Teleport).

🛠️ The Solution: Architecture Overview

PRIME C-19 acts as a patch for this topology issue using three specific mechanisms:

1. Shortest-Arc Interpolation (Topology)

We replaced standard linear delta calculations with a modular distance function.


$$\Delta = ((P_{target} - P_{current} + N/2) \pmod N) - N/2$$


This ensures the error signal always flows through the "shortest bridge" across the ring, effectively turning the linear strip into a true cylinder.

2. Fractional Gaussian Kernels (Gradients)

Discrete pointers (Integers) have zero gradients between steps. We implement Fractional Read/Write Heads (e.g., Index 10.4) using truncated Gaussian kernels.

Engineering Note: This architecture enforces FP32 precision for pointer mechanics. We found that standard FP16/AMP mixed precision rounds micro-gradients to zero, paralyzing the learning process.

3. The Möbius Phase Flip (Capacity)

To maximize the logical capacity of a fixed-size physical ring, the architecture tracks a logical coordinate space $[0, 2N)$. When the pointer traverses the logical horizon ($N/2$), the retrieved vector is multiplied by $-1$. This allows the storage of "Anti-Features" in the same physical space without interference.

📊 Comparison to Standard Methods

Feature

Transformers (Attention)

Standard RNNs (LSTM/GRU)

Neural Turing Machines

PRIME C-19

Context Cost

$O(N^2)$ (Quadratic)

$O(N)$ (Linear Decay)

$O(N)$ (Softmax)

$O(1)$ (Local Kernel)

Topology

Flat Sequence

Flat Sequence

Linear Tape

Circular Manifold

Boundary

N/A

N/A

Hard Boundary

Continuous Loop

Stability

High

High

Low (Unstable)

High (Stabilized)





**Status:** PRE-ALPHA (research prototype). This is a proof-of-concept published early
to establish prior art. It is **not** production-ready and is **not** expected to
work end-to-end yet. Expect breaking changes, unstable results, and incomplete
components.

**Last updated:** 2026-01-17 (local time)

This repository contains experiments and code for a pointer-controlled ring memory
system ("Absolute Hallway") used in sequential tasks. **PRIME C-19** is the
codename for the *Phase-Recurring Infinite Manifold Engine* with the
**Candidate-19** activation function.

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
results are recorded in `proof_ab.csv` (short run; not a full benchmark).

## What is this?

At a high level:
- A continuous pointer moves on a circular memory ring.
- A kernel (Gaussian/VonMises) reads/writes a local neighborhood of the ring.
- Pointer movement is controlled by learned signals plus stabilizers.

For details, see `tournament_phase6.py` and `DEFENSIVE_PUBLICATION.md`.

## Optional Auto Controls

- `TP6_THERMO=1` enables flip-rate thermostat (adjusts inertia/deadzone/walk).
- `TP6_PTR_UPDATE_AUTO=1` enables auto pointer update cadence.
- `TP6_PANIC=1` enables loss-based panic reflex (reduces friction on loss spikes).
