# TECHNICAL NOTE

Title: PRIME C-19 and the Pilot-Pulse Conjecture
Author: Daniel Kenessy
Status: Research prototype (pre-alpha)

## Abstract

PRIME C-19 is a recurrent neural memory architecture that navigates a continuous
1D circular manifold (ring buffer). The system couples shortest-arc pointer
updates with fractional read/write kernels to stabilize gradient descent on
closed loops. We present a control-oriented hypothesis, the Pilot-Pulse
Conjecture, which frames intelligence as navigation efficiency on a structured
manifold rather than pure compute or storage capacity. This document is a
technical note intended for prior art and reproducibility. It does not claim
machine consciousness.

## What is implemented

- Shortest-arc pointer interpolation (seam-safe updates on a ring)
- Fractional read/write kernels with stable sub-bin gradients
- Pointer physics controls (walk probability, inertia, deadzone)
- Cadence control and optional governor for pointer update timing
- Optional soft readout and phantom hysteresis

## The Pilot-Pulse Conjecture (Hypothesis)

Core thesis: intelligence is not only compute or storage, but navigation
performance on a structured manifold. "Thinking" is the control agent (Pilot)
traversing the Substrate (encoded geometry).

Pilot-Substrate dualism:
- Substrate: the learned structure stored in weights.
- Pilot: a persistent pointer that navigates the structure.
A strong Substrate with a poorly tuned Pilot can be dysfunctional. Both must
align to yield stable intelligence.

Law of topological inertia:
- Walker regime: step-by-step verification, slow but stable.
- Tunneler regime: predictive leaps when inertia is aligned, fast but risky.
These are control dynamics, not claims about biology.

Singularity mechanism (insight):
Under low friction and aligned inertia, the Pilot can converge rapidly to the
Substrate structure, moving from search to resonance. This remains a hypothesis.

Scaling rebuttal (soft form):
Larger substrates expand capacity but also search entropy unless the Pilot is
physics-aware. We expect self-governing inertia and cadence control to matter
alongside parameter count.

## Evidence (bounded)

- Micro assoc_clean (len=8, keys=2, pairs=1): 1.00 acc at 800 steps across
  3 seeds using cadence-aware control (see README and docs/ASSOC_CLEAN_SWEEP.md).
- Cadence knee: learning appears only at update_every >= 8 on the same micro task.

These are internal results and not a general proof.

## Limitations

- Pre-alpha; unstable and not end-to-end.
- Results are narrow and task-specific.
- The Pilot-Pulse Conjecture is speculative and testable, not a claim of
  consciousness.

## Reproducibility

See docs/REPRO.md for exact commands and run settings.

## Citation

If you cite this note, use the repository and the Zenodo DOI (once created).

