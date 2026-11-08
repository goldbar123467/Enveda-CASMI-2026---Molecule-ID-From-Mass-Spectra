# Public scan and prepared C08 experiment — 8 November 2026

P13 returned **0.480** and P14 v2 returned **0.458**, both COMPLETE without a
reported execution error. P01/P13 retain the best public score,0.480. P14's
reserved *conditional* retrieval gain did not predict the leaderboard result.
That is a failed transfer experiment, not a successful production improvement.
The two approved submission slots are used; this preparation does not submit.

The new request authorizes a fresh public scan and preparation of a longer2070
run. C08 is prepared; only its separate short learning/resource/restart pilot
has run. The full24-hour clock and recovery supervisor have not started.

## What the new public scan actually established

Kaggle's newest, score-sorted and vote-sorted first100 listings produced247
unique catalogue entries. Twenty selected notebook sources and their comments,
markdown, embedded modules and attached-asset metadata were examined as text.
This is not an exhaustive review of247 implementations. No downloaded notebook
was executed. [Source/hash index](evidence/scan-2026-11-08/source-index.json)
records the exact snapshots; titles and API sort order do not verify scores.

| Source | Finding | Decision |
|---|---|---|
| [bgunshui V7 model documentation](https://www.kaggle.com/datasets/bgunshui/casmi26-scorer-v7-model) | Compatible320-wide/320-embedding,8-head,3-layer scorer;48 hard negatives from512,100×400 updates,lr2e-4. Competition-only training; raw train SHA matches ours. | Borrow architecture and negative-recipe hypotheses; train our own models. No V7 weights downloaded or used. |
| [Latest peaks-v2](https://www.kaggle.com/code/zz3t3rn41/casmi26-peaks-to-structures-v2) | Public/V2/V7 blend0.4/0.3/0.3 plus library/NP/mass-prior/outsider changes. | Multiple simultaneous changes, not evidence that any one component helps. Do not transplant the whole stack. |
| [New hard-negative trainer](https://www.kaggle.com/code/cheek2005/casmi26-scorer-train-hn) |32 negatives/pool256; otherwise ordinary kit-family training/refit. | Supports a controlled negative-recipe comparison, not a hidden-score claim. |
| [Today's seed trainer](https://www.kaggle.com/code/sardorishiguro/casmi26-scorer-train-seed-20261108) | Ordinary80-epoch refit with a new seed. | A new timestamp is not a new method. Preserve actual configuration, not an unchanged copied default file. |
| [Fable of the Lighter Hand](https://www.kaggle.com/code/prvsiyan/fable-of-the-lighter-hand) / [Weight of an Exact Mass](https://www.kaggle.com/code/prvsiyan/the-weight-of-an-exact-mass) | Extracted code differs in exactly one line: zero versus two extra scorer seeds. | Useful controlled ablation. Extra seeds are not automatically improvements. |
| [Analog propagation baseline](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline) | Author discusses local gains failing on the board and added-scorer regressions. Perfect training-library hits cannot validate rank-one rules. | Require real candidate lists and conservative integration. These are author reports, not our reproduced scores. |
| [0.485 title](https://www.kaggle.com/code/ozertuu/0-485-sota-casmi26-from-peaks-to-structures) / [0.486 title](https://www.kaggle.com/code/ozertuu/casmi26-apex-gold-486) | Identical extracted code SHA. A missing private scorer falls back to public weights; metadata does not supply that missing asset. | Neither title is a reproducible score claim for us. No opaque scorer imported. |
| [Enveda180 experiment](https://www.kaggle.com/code/adityaraj0612/casmi26-exp298-enveda180) | Public reference-library/candidate assets, not automatically novel labelled spectrum training. | Keep retrieval overlays separate from neural training. Audit licensing, provenance and overlap before reuse. |

Other reviewed sources, including calibrated/reranker/FLARE variants, are linked
in the index. Missing dependencies and proprietary/unclear checkpoint provenance
remain reasons not to adopt a model. Existing competition rules/external-data
review is recorded in the6/7November reports. C08 adds **no external dataset or
pretrained checkpoint**; it trains only on the supplied competition train file.

V7's reported0.6671535 is a random molecular holdout diagnostic with256
truth-included candidates. Its final refit includes that former holdout. It is
neither a competition score nor an independent evaluation of its final model.
We do not use those labels/weights as a fold-safe baseline.

## The controlled long run

Use Deotte's [common-fold, independent-model-folder workflow](https://github.com/cdeotte/kaggle-playground-solutions)
with the credited [Saner model/features](https://www.kaggle.com/datasets/sanpier/casmi-spectrum-model-v1).
C08 starts fresh weights and compares four recipes on the same two existing
identity/acquisition folds (eight sequential models):

| Arm | Peak storage simulated at input | Hard negatives | Pool |
|---|---|---:|---:|
| legacy16-default | FP16 round trip, then FP32 |16|128|
| precise32-default | FP32 |16|128|
| legacy16-hard | FP16 round trip, then FP32 |48|512|
| precise32-hard | FP32 |48|512|

All arms share320 width/embedding,8 heads,3 layers,96 peaks,2048-bit Morgan
targets, logical batch384, dropout0.1, temperature0.1 and auxiliary weight0.5.
Fresh paired initialization and separate query/negative RNG streams preserve
the identity/view sequence across recipes. Both negative count and pool size
change together: this tests the **negative recipe**, not negative count alone.
No external-spectrum arm is repeated after L06's negative result.

Each model gets40,000 successful optimizer updates,1,000-step warmup and cosine
decay from2e-4, AdamW decay0.01 and gradient clipping1. This is a larger query
budget than V7's batch96 run; it is **not an exact V7 reproduction**. No epoch or
model is selected using training loss or truth-forced retrieval. Final weights
remain unqualified until actual-pipeline validation. Checkpoints every1,000
updates preserve optimizer, scaler and Python/NumPy/Torch/CUDA RNG states.

The new precision hypothesis is ours: the legacy disk cache rounds normalized
m/z and neutral-loss coordinates to FP16 before Fourier features are computed.
On8,192 spectra/505,060 peaks, the measured m/z round-trip error median/p95 is
0.02456/0.09885Th. This measures representation error, **not ranking benefit**.
FP32 disk inputs preserve those coordinates while both arms retain FP16 AMP
for neural operations. See [precision measurements](evidence/compatible-2026-11-08/precision-audit.json).

## Real data and leakage checks

The separate cache has1,504,015 usable spectra, at most eight acquisition-diverse
views per identity,271,118 mapped identities and270,127 eligible identities after
zero-view/shared-input exclusions. Of2,539,608 raw rows,71,962 raw InChIKey14
values are absent from the preserved canonical catalogue; they are counted and
excluded, never guessed. This mapping/cap is a deliberate difference from V7.
All selected views passed finite/length checks. The original cache is unchanged.

The input-group split SHA is pinned. Shared-input aliases (999 catalogue
identities) are excluded from both positives and negatives; no group crosses
folds. Every pool entry is an eligible training identity, never a held-out one.
Auxiliary class weights use training fingerprints only. Current RDKit rechecked
64 preserved fingerprints/identities with zero mismatches. Source/polarity/adduct
cohort counts are saved. These are existing reused folds, not newly untouched
confirmation data, and are not claimed to be scaffold splits.

One FP32 memory-mapped master cache serves all arms; FP16 controls round only
their gathered batch, avoiding another full disk copy. Fingerprints and pools
are also memory-mapped. CPU/Arrow use two threads. Free RAM floor4GiB; Torch
allocator cap70% of8GB. FP32 here describes stored features, not full neural
precision. [PyTorch AMP guidance](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html)
supports FP16 autocast with scaled gradients and unscale-before-clipping;
[CUDA memory guidance](https://docs.pytorch.org/docs/2.11/notes/cuda.html)
distinguishes allocated/reserved memory from whole-device telemetry.

## Measurements, recovery and launch

The shape benchmark measured batch384 at3,520 spectra/sec and2.24GiB peak
reserved VRAM;768 was only3% faster and used4.40GiB. Those measurements omit
per-update host data loading and the final weighted auxiliary/duplicate mask.
The real pilot measures those costs. Its report and checkpoint-restart result
are in [the execution receipt](evidence/compatible-2026-11-08/execution.json).
Expect roughly12–15hours for the full eight-model run, with a24-hour wall cap;
this excludes later pipeline validation and Kaggle scoring. It is an estimate
from a short pilot, not a completion promise.

The independent supervisor audits every30minutes, notices child completion
promptly, and allows at most three restarts within the original deadline.
Launch preflight verifies every prepared array/pool SHA and seals the fingerprint,
catalogue, selected-row and split bytes used by the pilot.
An OOM starts a fresh child from the last atomic checkpoint, reducing query
microbatch384→192→96 while retaining384 in-batch positive competitors. Repeated
non-resource errors stop for inspection. Recovery does not edit scientific code,
extend the deadline, submit, or refit on validation identities. Microbatch
fallback changes dropout draws; receipts record it, so it is not bit-exact with
a monolithic update. A separate CUDA test checks gradient/loss agreement with
dropout disabled and duplicate masking across microbatches.

```powershell
# Readiness only: does not start training or consume its24-hour clock
.\start-compatible-training.cmd --check

# Start the prepared full run and its30-minute recovery supervisor
.\start-compatible-training.cmd

# Browser or terminal monitoring of this run
.\training-monitor.cmd --web --open --run outputs/local/long-compatible-20261108
.\training-monitor.cmd --terminal --run outputs/local/long-compatible-20261108
```

Preparation/pilot for a restored checkout (requires the pinned local data/kit,
not committed to Git):

```powershell
.venv-gpu\Scripts\python.exe scripts/compatible_long_run.py --prepare-only
.venv-gpu\Scripts\python.exe scripts/audit_compatible_inputs.py
.venv-gpu\Scripts\python.exe scripts/compatible_long_run.py --smoke
```

Future public scans use `scripts/scan_public_strategy.py --refresh --out <new-folder>
--evidence <new-evidence-folder>`. The20-reference list is editable; retain dated
snapshots rather than overwrite this evidence.

## Promotion remains separate

No automatic submission is wired. Before any model can replace or blend with
P01, build labelled **actual pipeline candidate lists**, including cases where
truth is absent (MRR zero), with source/identity/acquisition leakage audits.
Use development-only model/weight selection (initial new-model weights0/.05/.10/.20,
protect rank one), then a separate reserved end-to-end paired comparison.
P14 must be a failed-transfer negative control: a panel that keeps favoring its
known losing recipe is insufficient to qualify another replacement. Cross-fit
any train-library retrieval stage too; leaving the query's library acquisition
in the index would make a misleadingly easy validation query.

Serving must use the selected arm's **same precision round trip, stable peak
ordering, fingerprint implementation and fixed config**. Require local/Kaggle
preprocessing/raw-score parity, full real Kaggle smoke, time/memory budget and
official-identity dedup before an explicitly authorized scoring trial. Preparing
or finishing C08 does not establish a higher score.
