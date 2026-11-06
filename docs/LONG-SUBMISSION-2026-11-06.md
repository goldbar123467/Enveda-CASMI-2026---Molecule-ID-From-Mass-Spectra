# P12: user-requested L06 leaderboard trial

The user explicitly requested **"Submit it use a submission"** after seeing the
completed experiment's results. This authorizes one leaderboard trial despite
the earlier statistical promotion gate; it does not establish expected gain.
No additional permission is needed after the required execution checks pass.

## Candidate

P12 uses the five control checkpoints from L06, each128 complete epochs, rather
than the external-data models or fitted stack that lost on reserved confirmation.
It averages the five raw candidate-score vectors, then uses the existing
`0.5*z(original public scorer) + 0.5*z(long control mean)` neural recipe. P01's
documentation weights, physical rules, hidden-rerun detection and official-metric
identity deduplication are retained. The previous0.480 notebook is preserved.
These five models each saw80% of training identity groups; they are not full-data
refits. Their OOF retrieval scores are not Kaggle scores.

- Private dataset: `clarkkitchen/casmi26-long-control-20261106`, version1.
- Notebook: `clarkkitchen/casmi26-p12-long-control`, version1.
- Manifest SHA256: `889f459c11a40e326de9564a31c45cd389b6f23acbca7555cca54176cd7f4c5c`.
- Build: `scripts/build_long_anchor.py`; inference: `scripts/score_long_ensemble.py`.
- Independent smoke audit: `scripts/audit_long_smoke.py`.
- Unique submission description marker: `L06-P12-20261106`; maximum one submission.

## Execution evidence

The local GPU inference preflight scored all five pinned models on12 cached real
query molecules/688 candidate rows, with zero input errors and independent mean
error0.0. It took30.62s; peak Torch reserved memory was0.0664GiB. This checks
inference only; it is not a hidden-score or full-pipeline validation result.
The three numerical aggregation tests and three existing submission-guard tests
passed. The notebook schema validates, all code cells compile and are guarded to
remove a frozen fallback CSV on uncaught failure, and all nine embedded module
byte pins match. The private dataset finished creation with status`ready`.

P12 version1 was pushed for a real12-molecule Kaggle GPU smoke. Submission waits
for COMPLETE, valid400-row CSV, exact official identity dedup, applied merge,
all five models actually scoring, and independent mean/public-blend checks.
Final scoring references and audit numbers belong in SUBMISSIONS.md after the
submission is actually accepted. Never report a queued smoke as a submission.

## Accepted submission

P12 version1 completed the real Kaggle GPU smoke in about22.5 minutes. Independent
audit:400 valid rows,12 real scored molecules, zero official identity duplicates
and zero model input failures. All five checkpoint hashes matched. The fivefold
mean error was0.0; independently reconstructed original/control blend error was
4.44e-16. All12 processed rows differ from the frozen0.465 baseline, and the
required physical merge was applied.

**Submitted version1 at2026-11-06T23:50:11UTC; reference58071964.** Exactly one
scoring submission was accepted. The hidden rerun/public score is pending.
Receipt: `docs/evidence/long-2026-11-06/p12_submission.json`.
Independent audit: `docs/evidence/long-2026-11-06/p12_kaggle_smoke_audit.json`.
The previous0.480 P01 submission remains the verified baseline.

## GitHub handoff and wait

The user requested the current project and results be pushed to GitHub, followed
by waiting. The latest API check still lists P12 as pending; there is no new
public score to report. Source, configs, notebook, discussion coverage and compact
numeric receipts are saved together. Dataset/checkpoint binaries, runtime logs,
credentials and virtual environments retain their existing Git exclusions; the
five production weights are already available in the private Kaggle dataset.

Pre-push verification:82 CLI tests passed with28 environment-specific skips;
all7 targeted long-training GPU tests and all3 numerical ensemble tests passed.
No further experiment or submission is queued. Existing Kaggle scoring continues;
local compute waits for a new explicit instruction. The read-only dashboard can
remain available.
