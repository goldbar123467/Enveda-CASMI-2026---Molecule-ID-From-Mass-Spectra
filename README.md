# Enveda CASMI 2026 — Molecule ID from Mass Spectra

Private workspace for the Kaggle code competition
[`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
Training and labelled scorer evaluation can run on the local RTX 2070 with `.venv-gpu`.
The full candidate-generation pipeline and competition submissions run on Kaggle.

Latest results (8 November): **P13 scored0.480; P14 v2 scored0.458**. Both
completed without reported errors. P14's positive conditional retrieval result
failed to transfer to the leaderboard; P01/P13 retain the0.480 best. Both approved
scoring slots are used. See [the submission log](SUBMISSIONS.md).

New public scan:247 catalogue entries,20 selected source reviews. The prepared
**C08** run compares four matched320-wide compatible-scorer recipes on two folds:
FP16/FP32 peak storage × default/stronger hard negatives. Batch384 leaves ample
2070 headroom. **Full training started8November17:23 Eastern**, with the30-minute
recovery supervisor and hourly tracker running. The user authorized one
submission after completion and mandatory checks. The completion runner waits
for all8 fits and the novel external full-pipeline panel, selects on development
only, packages the models, runs a real Kaggle smoke and independent audits, then
submits once. Estimated training12–15h,24h cap; validation/scoring take additional
time. Read [the live protocol](docs/C08-CAMPAIGN-2026-11-08.md).
Use `.venv\Scripts\python.exe scripts/finish_compatible_campaign.py --check`
to inspect readiness; use the monitor below to watch the existing run.

Best verified public MRR remains **0.480 (P01/P13)**. P12 ref58071964 finished with
an unhandled hidden-rerun error and **no score**, despite passing its visible
smoke. L06 local retrieval MRR was0.6806 control /0.6463 external; these are not
Kaggle scores. See [the new protocol](docs/FROZEN-TRAINING-2026-11-07.md),
[research](docs/NEXT-LONG-RUN-2026-11-07.md) and [submission log](SUBMISSIONS.md).

To watch local training, run `.\training-monitor.cmd` for a browser dashboard, or
`.\training-monitor.cmd --terminal` for a live terminal view. Both show progress,
ETA, loss, memory and watchdog health. See [the monitor guide](docs/TRAINING-MONITOR.md)
for options and future runs. Python standard library only; no installs needed.

- [`docs/LONG-TRAINING-2026-11-06.md`](docs/LONG-TRAINING-2026-11-06.md) — current shared-fold external-data training and resource/recovery plan.
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
submitted method (public MRR@25 **0.480**, checked 2026-11-08). It supersedes the older
daily submission quota policy. Local retrieval/panel MRR and Kaggle public MRR are
different evaluations; their absolute values must not be compared.

Tests use two environments:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -q
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -p test_tail_fusion.py -v
```

The CLI suite skips numerical tests if NumPy is absent. The GPU environment runs them;
it need not install the CLI notebook-authoring dependencies.
