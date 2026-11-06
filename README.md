# Enveda CASMI 2026 — Molecule ID from Mass Spectra

Private workspace for the Kaggle code competition
[`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
Training and labelled scorer evaluation can run on the local RTX 2070 with `.venv-gpu`.
The full candidate-generation pipeline and competition submissions run on Kaggle.

To watch local training, run `.\training-monitor.cmd` for a browser dashboard, or
`.\training-monitor.cmd --terminal` for a live terminal view. Both show progress,
ETA, loss, memory and watchdog health. See [the monitor guide](docs/TRAINING-MONITOR.md)
for options and future runs. Python standard library only; no installs needed.

- [`docs/SCIENCE-2026-11-05.md`](docs/SCIENCE-2026-11-05.md) — current research,
  two-round experiment protocol, validation limits and submission decision.
- [`NOTICE.md`](NOTICE.md) — upstream attribution and licensing scope.
- [`docs/COMPETING-2026-11-05.md`](docs/COMPETING-2026-11-05.md) — resumed active
  objective: continue until an actual submitted public score exceeds 0.477;
  exact raw-spectrum validation and current submission evidence.

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
