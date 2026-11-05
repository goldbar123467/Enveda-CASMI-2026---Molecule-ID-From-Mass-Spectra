# Enveda CASMI 2026 — Molecule ID from Mass Spectra

Private workspace for the Kaggle code competition
[`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
Everything runs on Kaggle; this machine only edits notebooks and drives the Kaggle CLI.

- [`CLAUDE.md`](CLAUDE.md) — working rules, commands, safety patches applied to every variant.
- [`STRATEGY.md`](STRATEGY.md) — how the current best pipeline works, what has been tried, next ideas.
- [`SUBMISSIONS.md`](SUBMISSIONS.md) — every submission with its change and public score.
- `kaggle/forks/<slot>/` — one private Kaggle notebook per experiment; `kaggle/forks/queue*.json` — submission queues.
- `scripts/make_variant.py` (build forks), `scripts/autopilot.py` (push + submit under the daily cap).
- `docs/archive-2026-10-08/` — superseded Phase-1 documents, kept for history only.
