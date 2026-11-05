# Enveda CASMI 2026 — Molecule ID from Mass Spectra

Private workspace for the Kaggle code competition
[`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
Training and labelled scorer evaluation can run on the local RTX 2070 with `.venv-gpu`.
The full candidate-generation pipeline and competition submissions run on Kaggle.

- [`docs/SCIENCE-2026-11-05.md`](docs/SCIENCE-2026-11-05.md) — current research,
  two-round experiment protocol, validation limits and submission decision.
- [`NOTICE.md`](NOTICE.md) — upstream attribution and licensing scope.

- [`CLAUDE.md`](CLAUDE.md) — working rules, commands, safety patches applied to every variant.
- [`STRATEGY.md`](STRATEGY.md) — how the current best pipeline works, what has been tried, next ideas.
- [`SUBMISSIONS.md`](SUBMISSIONS.md) — every submission with its change and public score.
- `kaggle/forks/<slot>/` — one private Kaggle notebook per experiment; `kaggle/forks/queue*.json` — submission queues.
- `scripts/make_variant.py` (build forks), `scripts/autopilot.py` (push + submit under the daily cap).
- `docs/archive-2026-10-08/` — superseded Phase-1 documents, kept for history only.

The current instruction is to submit only after evidence of improvement over the best
submitted method (public MRR@25 **0.477**, checked 2026-11-05). It supersedes the older
daily submission quota policy. Local retrieval/panel MRR and Kaggle public MRR are
different evaluations; their absolute values must not be compared.

Tests use two environments:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -q
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_tail_fusion.py -v
```

The CLI suite skips numerical tests if NumPy is absent. The GPU environment runs them;
it need not install the CLI notebook-authoring dependencies.
