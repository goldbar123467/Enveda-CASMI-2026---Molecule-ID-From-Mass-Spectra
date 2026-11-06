# Shared-fold external-spectrum training — 6 November 2026

The user explicitly resumed project work and requested a longer RTX2070 run,
research of Chris Deotte's solutions and every competition discussion/comment,
and removal of obsolete models. Yesterday's pause is superseded for this work.
The completed [discussion review](DISCUSSION-REVIEW-2026-11-06.md) records the
important host rulings, conflicting claims and resulting implementation choices;
`discussion-coverage-2026-11-06.json` lists all 148 topics across eight pages.

## Baseline and experiment

Kaggle submissions API checked today: P01/ref58032057 scored **0.480**,
P02/ref58032626 **0.479**, P11/ref58036642 **0.468**. The best verified public
baseline is now **0.480**. The NP continuation's local gains failed to transfer;
do not promote another long model from artificial retrieval alone.

[Deotte's repository](https://github.com/cdeotte/kaggle-playground-solutions),
commit `e5a03fffe08a8904feb40c66b85142abb1ec6cc7`, describes six solutions with
shared five-fold splits, reusable feature caches, independent model folders,
OOF prediction contracts, public-model attribution and L2 stacking. The six
solution READMEs plus the common data and stack code were retrieved locally.
The transferable system is the evaluation/ensemble scaffold. Its synthetic
tabular generator models and AUC objectives do not transfer directly to MS/MS.

The new experiment uses five SHA256 folds of stereo-stripped,
tautomer-canonical InChIKey14 identities. The same molecule has the same fold
in competition, MassBank and GNPS data. Both positives and hard negatives,
and auxiliary fingerprint frequency statistics, exclude the query's fold.
Connected components of byte-identical cached spectrum inputs also share a
fold, even when their structure labels disagree. This addresses the
[conflicting-label duplicate report](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744446);
quantized-input equivalence is conservative and is not an acquisition identity.
The two paired conditions use the same architecture/seed schedule/budget:

* `control`: competition cache only.
* `external`: competition cache plus audited MassBank/GNPS spectra and structures.

Each model: width256, four transformer layers, eight heads, embedding192,
batch96, FP16 autocast/GradScaler, original mixed retrieval + fingerprint loss,
128 fixed epochs × 500 steps, cosine LR from 2e-4 to 1e-5. Ten sequential models,
640,000 planned batches. No initialization from yesterday's trained models.
Natural-product examples are evaluated separately; there is no NP oversampling.

The OOF cohort contains all matching public NP examples plus 256 deterministically
sampled competition molecules per fold. Query spectra prefer the exact existing
Enveda native row index when available. Candidate lists are identical in both
conditions: target plus 255 same-formula/near-mass decoys from the original
structure universe. These are **truth-forced retrieval diagnostics**; they do
not measure candidate recall, de novo generation or the full competition pipeline.

The stack uses four standardized score channels (cosine/fingerprint per family),
listwise cross entropy and fixed L2 regularization. A SHA256 20% cohort is excluded
from fitting the stack; a fixed equal blend is also reported. Ordinary OOF is
not nested validation of ensemble construction: other-fold base models can see
an identity assigned to the stack's confirmation cohort. The report states this
limitation. No local number is advertised as a predicted Kaggle score.

## Data permissions and provenance

Read the [official rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules)
and the host's [17 October clarification](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750330).
Free, open external data including NC licenses is permitted; proprietary NIST
and derivatives are generally prohibited. There are explicit exceptions:
[GNPS-NIST14-MATCHES](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745832)
is allowed for library matching/training; the
[ICEBERG PubChem atlas](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745134)
is allowed as downloaded predictions with retrieval/processing code. A question
about training/distillation on atlas predictions remains unanswered; do not infer
permission for that new use. This run uses neither exception.

The [4 November reproducibility clarification](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/753662)
requires traceable data and reproducible intermediates. Public weights need the
original training code and permitted training data; a link to weights alone is
insufficient. [Host answers in 748882](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748882)
allow original released DreaMS/FRIGID and MassSpecGym ICEBERG/GLACIER with their
training references, and PubChem counts, LOTUS and NPAtlas under source licenses.
The new models are trained from scratch using the delivered trainer.

[Aaron Kanzer's disclosure](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/755965)
provides public S3 mirrors. Source snapshots and MD5 manifests are retained in
`outputs/research/2026-11-06/`; raw files are MD5-verified:

* MassBank 2026.03: 96,949 mirrored records, 23,611,171 bytes,
  MD5 `8fbda467165353851eb112054f19b9e6`.
  [Upstream release](https://github.com/MassBank/MassBank-data/releases/tag/2026.03).
  Keep explicit CC0, CC BY4, CC BY-SA4 and dl-de/by-2-0 record licenses/attribution.
* GNPS community snapshot 22 October 2026: 581,722 mirrored records,
  592,400,485 bytes, MD5 `22d30be090f0b088082f5fa23de0c06e`.
  [GNPS libraries](https://external.gnps2.org/gnpslibrary) and
  [upstream documentation](https://ccms-ucsd.github.io/GNPSDocumentation/gnpslibraries/).
  Only the mirror's directly contributed libraries are admitted; additionally
  exclude NIST/METLIN/propagated/suspect/DEREPLICATOR named libraries and require
  MS2 plus library class1/2. Retain source library/accession with every row.

Intake rejects salts, invalid structures, unsupported adducts, fewer than three
peaks and precursor/structure discrepancies above max(0.03Da,50ppm).
It deduplicates external cache spectra by canonical identity and exact cache
features. It does not establish independent acquisition provenance against
competition records; shared copies may remain, and no new-measurement claim is
made. Validation identities are excluded across sources regardless of copying.
Collision energies marked NCE or percent stay unknown rather than being treated
as eV. The [host confirms](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/753419)
that timsTOF energies are nominal and lists indicate merged acquisitions.

## Intake and launch evidence

The intake completed in 1,891 seconds on one low-priority CPU process:
**436,121 spectra, 54,271 canonical identities** (MassBank71,651;
GNPS364,470). Against the existing 273,833-identity cache, **996 identities are
new**, and 53,275 overlap; these are not 54,271 newly discovered structures.
All pinned input MD5s, generated-file SHA256s and binary-array dimensions passed.

The first real-data guard detected 93 overlapping identities whose equivalent
chemical graph spellings yield different Morgan vectors (examples include
S(=O) versus [S+]([O-]) resonance forms). These external identities are excluded
from training. The repeated real RTX2070 smoke passed both conditions, eight
actual optimizer steps each, finite weights, external data actually sampled
(133 external rows), and 48 held-out queries through each scoring path.
Those tiny smoke metrics are execution checks, not scientific evidence.

The complete split audit also removed **80,848** exact duplicate external model
inputs. **354,048** external spectra remain eligible after representation and
input-duplicate exclusions. It found 2,467 cross-label input matches, joining
274,829 identities into 273,875 fold groups. Quantized input equality does not
prove independent acquisition; these are conservative model-input groups.

Run **L06** was launched as a hidden, low-priority detached process after the
smoke. Its ledger refresh runs inside the controller hourly and at completion,
so no perpetual watcher is left behind. Final confirmation diagnostics will be
written back to runs.csv automatically. No Kaggle upload/submission is queued.
The first three full epochs completed 1,500 batches with finite losses and zero
AMP-skipped optimizer steps; approximately22 seconds per epoch. Torch reserved
about0.62GiB VRAM, total device use including the desktop was about1.75GiB,
and free system RAM remained above13.6GiB. The initial pace suggests roughly8h
for the full matrix, subject to later job/evaluation overhead and interruptions.
The active resume checkpoint contains about60MB of model/optimizer/RNG state.

Validation: the CLI suite passed63 tests (25 numerical tests skipped in that
environment). The targeted GPU suite passed7 tests, including an actual CUDA
interruption/resume comparison with identical final weights. A synthetic full
fivefold ensemble test passed and rejected a changed model artifact. The full
GPU-environment discovery also encountered one CLI-only missing `nbformat`
dependency; CLI checks were run in `.venv`, as the project prescribes, rather
than changing the training environment. Real-data GPU smoke and initial full
training passed after the fingerprint-representation exclusions above.

## Resource and recovery contract

One controller, one training job at a time; two Torch CPU threads and one interop
thread; peak arrays are memory-mapped, not preloaded or copied into workers.
PyTorch allocator capped at 70% of 8GB (5.6GB), leaving desktop/driver headroom.
Require at least 4GiB free system RAM at each epoch boundary. A 12h session limit
stops at an epoch boundary; incomplete folds are not marked complete.

Each complete epoch atomically replaces `resume.pt`, containing model,
optimizer, scaler, Python/NumPy/Torch/CUDA RNG and full history. Resume validates
trainer/data/config/evaluation hashes. Final models and OOF predictions have
completion receipts with hashes; completed folds are skipped only after checking
those receipts. Active optimizer states are deleted only after final predictions
and receipts exist. The stack runs after all ten models finish. Nothing uploads
or submits automatically.

### 11:15 a.m. Eastern update: health supervisor and submission approval

The user explicitly renewed purge authorization and approved submission after
completion on 6 November. That approval persists; do not ask for it again.
Training remains on the original pinned recipe. A separate low-priority local
`scripts/long_run_watchdog.py` process now checks every30 minutes. Its first live
check at16:14:58UTC was healthy; next check is16:44:58UTC (11:44:58a.m. Eastern).
The original trainer PID12352 was left running.

The supervisor can restart a dead or stalled trainer from completed-epoch
optimizer/scaler/RNG state, repair its ledger status, archive failed logs and
wait for RAM/VRAM headroom. It verifies code/data provenance before restarting,
refuses unrelated processes or duplicate trainers, limits restarts to3, and
preserves the original global12h deadline (9:52p.m. Eastern). Changed code/data,
repeated deterministic failures or missing recovery state are flagged for repair;
it does not make unvalidated scientific code changes.

Live supervisor state: `outputs/local/long-20261106/watchdog/status.json`.
Settings and persistent user approval: `configs/long-watchdog-20261106.json`.
Eight supervisor tests passed; live health and provenance checks passed.

At219/1280 planned epochs, measured pace22.01s/epoch, the ETA was22:50UTC,
approximately5:50p.m. Eastern, before production packaging/validation overhead.
Use5:45–6:15p.m. as a working training estimate, not a scoring-return promise.
The supervisor writes `completion-review.json` when training finishes. It does
not itself build or submit a Kaggle notebook; Codex still needs to assess the
completed evidence, package the supported candidate, pass the real GPU smoke and
perform the already-authorized scoring submission. No new permission is required.

The renewed purge attempt and an individual-file attempt were both rejected by
automatic execution policy. The4,178,679,179-byte manifest remains undeleted.

```powershell
.venv-gpu/Scripts/python.exe scripts/fetch_external_spectra.py
.venv-gpu/Scripts/python.exe -u scripts/build_external_cache.py
.venv-gpu/Scripts/python.exe -u scripts/long_training.py --smoke
.venv-gpu/Scripts/python.exe -u scripts/long_training.py
```

Live progress: `outputs/local/long-20261106/status.json` and `train.log`.
Configuration: `configs/long-training-20261106.json`. The controller makes a
process-scoped system-awake request on Windows and restores it on exit. Keep
this project environment available while the detached worker runs.

## Promotion requirement

### Completed 6 November, 5:49 p.m. Eastern

L06 completed at22:49:29UTC: all10 models, 1,280 epochs and640,000 training
batches. All10 model/prediction SHA256 completion receipts were independently
checked after completion, as were128 history records and64,000 batches per
model. There were zero watchdog restarts, no failure receipt and empty stderr.
The watchdog observed completion at23:15UTC and exited normally. The trainer
also exited; GPU training load is now zero.

Reserved316-molecule confirmation retrieval MRR@25:

| Method | MRR | Paired delta vs control | 95% bootstrap interval |
|---|---:|---:|---|
| Control | 0.680621 | — | — |
| External | 0.646321 | -0.034299 | [-0.065787, -0.003513] |
| Equal blend | 0.662922 | -0.017698 | [-0.039383, +0.005193] |
| L2 stack | 0.671584 | -0.009037 | [-0.031754, +0.013530] |

On the234-molecule natural-product cohort, the equal blend improved0.470822 to
0.488623, but its delta interval[-0.001132,+0.039022] includes zero. This
cohort mixes development and confirmation identities; it does not overturn the
reserved confirmation outcome. The small development gains did not transfer.

Decision: do not promote external training or a blend from these diagnostics.
The experiment does not establish that any L06 model improves the existing
0.480 Kaggle submission. No new production upload, smoke or competition
submission was made. Submission authorization remains available if a candidate
later passes the required identity-excluded production qualification. Do not
retune on these observed confirmation results and call them a fresh test.

Full local numeric report: `outputs/local/long-20261106/stack/report.json`.

Subsequent user override: **"Submit it use a submission"** authorizes one
leaderboard trial despite these scientific limitations. P12 packages the
control fivefold family with the original50/control50 neural recipe. Execution
validation and exact one-submission accounting still apply. See
[`LONG-SUBMISSION-2026-11-06.md`](LONG-SUBMISSION-2026-11-06.md).

Assess paired molecule deltas, separate NP/general metrics, uncertainty and the
fixed blend. A promising OOF result still requires a realistic identity-excluded
full-pipeline replay, candidate-recall checks and a real Kaggle smoke before a
production model is submitted against **0.480**. Existing easy panels and formula
oracles cannot establish an improvement on hidden data. The discussion review
repeatedly reports the same CV/LB mismatch, source-prior bias and Class3 recall
failure; a bigger model or more epochs alone is insufficient evidence.

## Cleanup

Read-only inventory identified 77 obsolete checkpoint files plus the redundant
3.07GB native raw export, totaling 4,178,679,179 bytes (3.89GiB). The exact paths,
sizes and hashes are in `outputs/research/2026-11-06/purge-manifest.json`.
Original kit and current 0.480 v80 dependencies, essential caches, native row
indexes, source code, metrics, attribution and logs are protected.
Deletion has not occurred: automatic execution policy rejected the bounded
PowerShell purge. Do not report this space as reclaimed.
