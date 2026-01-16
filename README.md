# PRIME C-19 — Phase-Recurring Infinite Manifold Engine (Project David)

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

## What is this?

At a high level:
- A continuous pointer moves on a circular memory ring.
- A kernel (Gaussian/VonMises) reads/writes a local neighborhood of the ring.
- Pointer movement is controlled by learned signals plus stabilizers.

For details, see `tournament_phase6.py` and `DEFENSIVE_PUBLICATION.md`.
