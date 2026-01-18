# ZENODO UPLOAD CHECKLIST

## One-time setup

1) Create a Zenodo account.
2) In Zenodo, connect your GitHub account.
3) Enable the PRIME-C-19 repo in Zenodo GitHub settings.

## Release-based workflow (recommended)

1) Create a GitHub release (tag) on the repo.
2) Wait for Zenodo to archive the release and generate a DOI.
3) Edit the Zenodo record metadata:
   - Title
   - Authors
   - Description (use docs/TECH_NOTE.md summary)
   - Keywords
   - License (PolyForm Noncommercial 1.0.0)

## What to upload

- docs/TECH_NOTE.md (and optional PDF export)
- docs/HYPOTHESIS.md
- docs/REPRO.md
- docs/RESULTS.md

## Suggested metadata

- Title: PRIME C-19 and the Pilot-Pulse Conjecture
- Keywords: recurrent memory, manifold navigation, control theory
- Notes: Research prototype; speculative hypothesis; not production-ready

## Optional PDF export

If you want a PDF:
- Convert docs/TECH_NOTE.md to PDF with your preferred tool
- Upload the PDF to Zenodo alongside the release

