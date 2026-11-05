# Competing toward an observed score above 0.477

The user resumed work on 2026-11-05 with the explicit stopping condition: keep
working until the submitted public score exceeds the current best **0.477**.
An improved artificial retrieval metric is not completion of that objective.

## Operating method

1. Reproduce a strong, available baseline before inventing further changes. Keep
   a clear distinction between an author's displayed score and our observed score.
2. Validate metric-exact identities, real inference preprocessing and actual
   candidate lists. Split at molecule identity, never at individual spectrum.
3. Track distinct deployment cases: reference available, known structure without
   reference, and a structure removed from the candidate pool. The last case is
   still an optimistic novelty simulation, not a sample of genuinely new chemistry.
4. Evaluate retrieval/ranking changes at the retrieval/ranking level. Fingerprint
   or embedding quality alone is not a submission criterion. Use paired molecular
   deltas, error counts, uncertainty and explicit missing-truth denominators.
5. Predeclare a candidate and freeze its settings before hidden evaluation. Do
   not treat many correlated seeds, reused panels, or public notebook clones as
   independent evidence. Record negative results and failures too.
6. Run local GPU scoring and training with two CPU threads. Keep full library
   search and forward models on Kaggle; at most two GPU sessions and at most two
   pending submissions. Do not spend the submission allowance on failed candidates.

Sources: [official metric and notebook requirements](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview/evaluation),
[CASMI practitioner notes](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254),
[MassSpecGym](https://arxiv.org/abs/2410.23326),
[Jürgens et al., selective molecular retrieval](https://arxiv.org/abs/2603.10950v2),
and [MSAlign](https://arxiv.org/abs/2605.19752v2).
The selective-retrieval paper supports assessing uncertainty at the target task
level; we do not claim its risk guarantees transfer to these reused CASMI panels.

## Candidate P01: reproduce the reported public 0.480 method

Source: [alex chilton's 0.480 neural ensemble](https://www.kaggle.com/code/alexchilton/casmi26-lb-0-480-neural-ensemble),
crediting bgunshui, Baidalin Adilzhan, Kunal Desale and the existing public lineage.
The attached [public v80 rebuild](https://www.kaggle.com/code/alexchilton/casmi26-scorer-train-v80)
is COMPLETE. It selected epoch 52 from 56 holdout epochs, then refit on all data
for 52 epochs. Its displayed 0.480 claim predates that rebuilt attachment; an
observed reproduction on our account is still required.

The fixed recipe averages within-query standardized public and rebuilt neural
scores at 0.5/0.5, uses the public leader-safe consensus, fragment merge, fixed
documentation/spectral/NP prior on ranks 2–25, and tautomer-canonical connectivity
deduplication with available backfills. No weights are tuned in this reproduction.

Our private fork changes only robust hidden-rerun detection and a strict final
audit: the second model must be found and the merge must actually apply. It runs
a real smoke; fast placeholder commits do not qualify. Copied source `id_no` was
removed after the first SaveKernel request was rejected before execution.

Prospective submission criterion for this fixed external reproduction: available
assets, exact local neural-score reproduction, positive regression-panel changes
with both holdout checkpoints, supportive natural-product tail-prior checks,
and successful real pipeline smoke with no silent fallback. These justify testing
an expected improvement; they cannot guarantee the unseen Kaggle score. The
previous strict two-seed rule concerned the newly fitted candidate-loss fusion,
which remains rejected. This reproduction is a different, fixed candidate.

## Correcting the validation input contract

The old local cache uses float16 peak tensors and deduplicates model inputs.
The deployed neural scorer constructs float32 tensors from raw query spectra
without that deduplication. The 120-molecule public panel has five query-count
differences against the local cache. Therefore the previous cached-input panel
replays were approximate, even though validation identities were excluded.

We downloaded the public panel's exact `effective_query_spectra.parquet` files,
reapplied the deployed filters and checked every per-query spectrum count.
The kit holdout model now reproduces all stored scores to maximum absolute
error **1.90735e-6**. The old cached-input audit correctly failed instead of
silently accepting a mismatch (PL01); raw replay PL02 passes.

### Fixed public recipe, exact raw inputs, 120 regression molecules

| Rule | MRR@25 | Delta from current-recipe proxy | Improved / harmed |
|---|---:|---:|---:|
| Current 0.477 consensus proxy | 0.912842 | — | — |
| Documentation prior, original scorer | 0.914738 | +0.001896 | 5 / 1 |
| Public 0.480 recipe, v80 best holdout | 0.918796 | +0.005954 | 6 / 1 |
| Public 0.480 recipe, v80 last holdout | 0.914969 | +0.002126 | 5 / 2 |

Best-checkpoint paired bootstrap 95% interval: **[+0.000240, +0.013267]**;
last-checkpoint interval: **[-0.003704, +0.008730]**. Both point estimates improve.
This reused public panel also participated in historical model selection; these
are regression checks, not independent confirmation or hidden-test MRR. The
local replay does not reproduce the full fragment merge or outsider recovery.

Fixed alternatives averaging our two candidate-loss models, or those models and
v80, added only +0.000150/+0.000058 against this public recipe, with wide intervals
and five harmed molecules each. No production refit is justified for those blends.

### Natural-product documentation-prior stress checks

Apply the fixed tail prior to the existing top-25 lists, retaining rank 1 and
keeping all 250 identities in each scenario, including missing truths as zero.
This isolates a tail-prior regression check, not the full 0.480 neural pipeline.

| Cached scenario | Baseline MRR | With fixed tail prior | Delta | Improved / harmed |
|---|---:|---:|---:|---:|
| Reference in other libraries | 0.954830 | 0.958667 | +0.003836 | 7 / 0 |
| Structure in pool, no reference | 0.970641 | 0.972641 | +0.002000 | 4 / 0 |
| Structure removed from pool | 0.360533 | 0.364333 | +0.003800 | 7 / 0 |

The first two scenarios have optimistic absolute accuracy because upstream
models may have seen the identities; deleting a known structure does not create
genuinely novel chemistry. No headline Kaggle claim is based on these numbers.

## Precision hypothesis: tested and rejected as a production change

Float16 storage changes peak positions; high-frequency m/z features make this
a reasonable failure hypothesis. Raw versus quantized inputs changed neural
scores but gave identical final recipe rankings on the 120-molecule panel.

P03 exported **54,665 raw spectra** for the fixed 234 NP holdout identities.
RDKit **2026.03.3** verifies all 278 source SMILES map to their expected canonical
identities. After the exact kit duplicate/filter rule, **49,512 spectra** remain,
with all 234 query counts matching the training cache and no missing molecules.
Median absolute m/z quantization error is **0.02684 Da**, 99th percentile **0.19413 Da**.

On the 2070, three historical NP holdout models compare identical queries and
fixed 256-candidate sets. Quantization minus raw MRR deltas are
**-0.001758 / -0.000270 / +0.000243**. The molecule-paired mean-over-seeds delta is
**-0.000595**, interval **[-0.001582, +0.000148]**. The seeds share the same molecules.
Quantized scores also reproduce the historical NP diagnostics exactly. Thus
quantization is a real input mismatch, but correcting deployment to quantized
inputs has no supported benefit. Keep raw inference; do not promote this patch.

## Run state

P01 real GPU smoke completed: both models loaded, zero input failures, exact
50/50 standardized-score agreement, merge applied, 400 valid rows, all 12 real
smoke molecules processed. Commit output SHA-256:
`5cb2f1ad75c64791fa63cbfabb7e70e29948e05d8cc3987de79d7e137cdc20b9`.
The hidden rerun replaces that visible smoke output and runs every molecule.

**Submitted P01 v1 at 2026-11-05 18:15 UTC, ref 58032057; pending hidden score.**
P02 is the same documentation/dedup recipe with the original neural scorer only.
Its smoke completed with 400 valid rows, 12 real processed molecules and zero
input failures. Its 300 real neural candidate scores exactly match the original
channel in P01 (maximum absolute error zero). Submitted at **18:40 UTC, ref
58032626**. This control separates the new scorer from final-layer changes.

The user requires at least **two submissions reserved for later decisions before
00:00 UTC**, after the initial batch has scored. P01/P02 use two slots; three
remain. No further batch will be sent before the initial scores are available.

The read-only watcher polls every 180 seconds and requires a COMPLETE submission
with a finite public score strictly greater than 0.477; a tie, pending result or
invalid score cannot end the task. Objective remains active until an actual win.

## Correcting the acquisition-domain validation

The old 234-identity NP diagnostic used every reference spectrum of those
identities: 54,665 raw spectra, including 25,771 from Riken, 9,837 GNPS, 6,212
MassBank and 5,039 MoNA. Only **1,097 spectra** are the native Enveda NP timsTOF
queries. The target identities were held out, but pooling acquisition domains
made that diagnostic a poor stand-in for the hidden native query distribution.
For example, NP64 seed9 changes from pooled-reference MRR about 0.5363 to native
MRR 0.3764 on the same familiar 256-candidate groups. These are retrieval
diagnostics, not full-pipeline or Kaggle scores.

P04 additionally builds exact-formula candidate groups whose decoy identities
are absent from **all 273,833 Kit cache identities**. There are 234 targets and
10,724 unique decoys, at most 128 candidates per group. Group sizes range 1–128
(median 28); report the 185 groups with at least eight candidates separately.
The target identity is present first by construction, while decoys are unseen.
These are explicitly **formula-oracle diagnostics**, with historically reused
targets. They do not estimate competition MRR, and a value above 0.477 does not
count as beating the submitted best.

| Existing excluded-NP model | Native familiar MRR | Native unseen-formula MRR | Unseen groups with ≥8 candidates |
|---|---:|---:|---:|
| 16 epochs, seed1 | 0.2400 | 0.3773 | 0.2741 |
| 32 epochs, seed6 | 0.3243 | 0.4007 | 0.2947 |
| 64 epochs, seed7 | 0.3803 | 0.4321 | 0.3475 |
| 128 epochs, seed8 | 0.4401 | 0.4511 | 0.3741 |
| 256 epochs, seed10 | 0.4502 | 0.4933 | 0.4240 |
| Width256, 128 epochs, seed12 | 0.4636 | 0.4802 | 0.4016 |

Native-only raw/q16 checks remain mixed across three NP64 seeds; no precision
patch is promoted. The acquisition finding motivates a controlled sampling
experiment rather than another epoch-only ensemble.

## Concurrent workstation and Kaggle GPU experiments

The RTX 2070 and Kaggle T4×2 run independent fits concurrently. Both sources use
the original training cache and loss, width256 / four layers / eight heads,
128 complete epochs of 500 steps and a fixed final checkpoint. The exclusion
cohort is the union of every original Kit validation identity and the NP234
identities: **27,600 excluded, 246,233 training identities**. Thus the same new
holdout checkpoints can be evaluated on both the public panel and native NP
queries. Compare logical identity hashes across Windows and Linux; text-file
byte hashes differ with line endings.

Two conditions change only spectrum sampling within a training molecule:

- General: original uniform choice from all its cached spectra.
- Native-preferred: if a training molecule has a matched Enveda native view,
  choose from its native views; otherwise retain its original views. Sampling
  frequency per molecule, negative sampling, objective and architecture remain
  the same. Held-out mappings are never modified.

In notation, both minimize an average over the same training molecules,
`E_m E_(s|m) loss(s,m)`. The experiment changes the conditional acquisition
distribution `s|m`, while retaining the molecule distribution. This tests the
acquisition hypothesis without introducing an additional loss or capacity change.
The two paired seeds are 20 (RTX 2070, general then native) and 21 (T4 GPU0
general, GPU1 native). Identical seeds reduce variation; the two seed pairs are
the replication units for the training comparison, while molecule-paired
intervals describe uncertainty over these reused evaluation targets.

The native index streams 1,154,969 raw rows on one low-priority CPU core. It
matches exact key + float16 peak tensor + float32 metadata + adduct digests,
preserving the original cache labels. Of 1,148,546 unique eligible native inputs,
**1,146,540 match** (99.825%); 6,423 raw rows have identities absent from the
canonical cache and are omitted. Native preference therefore applies to
matched inputs; no claim of complete native-only coverage is made.

P06 precomputes the unchanged production fragment scorer in Kaggle, including
optimized/baseline scores and per-energy-chunk votes for all 3,000 listed panel
candidates. All 114 eligible first-12 candidate checks agree with the full-list
batch scores. The lightweight local replay uses this numeric evidence and the
same pinned physical rule; candidate generation and outside-list retrieval
remain outside the replay. Cache the cloud's exact selected spectra as well:
one of 114 molecules has a cross-platform NumPy peak tie-order difference.

The completed exact in-list physical replay supersedes the simpler panel proxy
for this comparison. Raw original-score terms reproduce the source with maximum
errors 3.28e-7 (cosine), 1.49e-8 (fingerprint) and 3.81e-6 (total score). Baseline
physical merge changes one leader, giving panel MRR **0.917009**. With the same
physical rule, P02's fixed recipe gives **0.918905** (delta +0.001896, 95% paired
interval [-0.000965,+0.005671]); P01's v80-best holdout proxy gives **0.918796**
(delta +0.001787, interval [-0.009722,+0.011587], 6 improved / 2 harmed). The
alternate v80-last gives **0.919135** (delta +0.002126, interval
[-0.003704,+0.008730]). The v80-best proxy triggers two physical leader changes,
whereas baseline and v80-last trigger one. Thus the earlier positive-only
interval from omitting the physical rule overstates the strength of local
evidence. The actual submitted hidden scores remain the deciding measurement.

## Current 0.481 public notebook: inspect, reproduce components, reject weak changes

The November 5 source of [zz3t3rn41's notebook](https://www.kaggle.com/code/zz3t3rn41/casmi26-peaks-to-structures-v2)
reports 0.481 and attaches accessible v2/v4 model datasets and a 149,983-structure
[LOTUS dataset](https://www.kaggle.com/datasets/qwertykeypadapple/casmi26-lotus-candidates).
Its published model configurations differ: v2 is 192/6/3 with embedding192 and
52 refit epochs; v4 is 256/8/3 with embedding256 and 109 refit epochs. These are
full-data refits, explicitly including former validation identities. We download
them for provenance/production inspection and **do not score them on the panel**.
The original trainer notebooks return 403. No holdout checkpoint from those
private trainers is assumed available.

Test the fixed final-layer policy separately with excluded-identity raw neural
channels: source weights POP=0.10, SPEC=0.10, NP=0.12, library gate0.78, augmented
LOTUS membership, and low-library NP leader promotion requiring a 0.055 score
margin plus a better neural rank. No threshold is fitted here. Against the
physical baseline panel MRR0.917009, the original-neural control gives 0.906184,
v80-best proxy0.906938 and v80-last0.915797. **Reject this composite prior/promotion
policy on the panel.** The exact published three-model ensemble is not evaluated
by these numbers. Its claimed score and claims of protected leaders cannot be
treated as measured guarantees for our run.

Tensor-level provenance check: the downloaded published v2 full model and P01's
v80 full model have **identical state tensors, maximum absolute error 0.0**, both
at epoch52. Their serialized file hashes differ. This resolves the concern that
the v80 attachment introduced different learned weights; it does not turn the
reported public score into an observed score on our account. No inference on
training molecules was needed for this check.

## First completed wide run and the next controlled test

PL08 completed on the RTX2070 at **20:03 UTC**: 128 complete epochs, **64,000
optimizer steps**, fixed final epoch128, 3,461 seconds including diagnostics.
The saved model is the final checkpoint, although the legacy retrieval monitor
peaked at epoch114. It gives native familiar MRR0.456220, formula-oracle MRR0.484772,
and hard185 oracle MRR0.393802. None of those is a Kaggle score.

On the exact in-list physical panel replay, original50%/general 20-50% plus the
fixed P01 prior gives 0.922160 versus baseline 0.917009. The prespecified
original50%/v80-25%/general 20-25% blend gives 0.922864. Against P01's holdout proxy
0.918796, that latter difference is+0.004067 (paired interval
[-0.000298,+0.012500], one improved / one harmed). Thus the point estimate is
promising but still uncertain and based on reused targets.

Local PL09 starts the matching native-preferred seed20 at **20:04 UTC**. Its
actual manifest matches the general run's training/exclusion hashes. Native
views exist for 163,854 of 246,233 training molecules, with 1,030,691 native rows.
An exposure audit is essential: under the original uniform-by-molecule sampler,
the expected native-view fraction is already **65.8957%**, rising only to
**66.5443%** under native preference. Only3,810 training molecules have both
matched native and other views. The validation acquisition mismatch was real,
but the proposed native sampler changes training exposure only slightly. Finish
the paired experiment and avoid extrapolating a large domain-correction effect.

The second Kaggle session P07 prepares a fixed general 20 production refit on
GPU0 and a union-excluded replica of the published v4 architecture on GPU1.
Version1 failed at argument parsing before any epoch; version2 uses a checked
structured dispatcher. The replica deliberately uses original seed0 with its
different architecture through the training library API. The production model
will never be evaluated on its former holdout.

The next preregistered hypothesis changes **chemical-domain sampling** instead.
The public metadata/LOTUS union annotates 23,076 of 246,233 training molecules
(9.3716%). Hold the parent weights, loss, new AdamW initialization and learning
rate2e-5 fixed, then compare32 full epochs under:

- Original uniform-by-molecule sampling.
- `q(m)=0.5 Uniform(all training identities)+0.5 Uniform(annotated NP training identities)`.

The second condition assigns54.6858% probability mass to annotated NP identities
before sampling each batch without replacement. This is an annotation-defined
subpopulation, not a complete classification of natural products. The original
hard-negative pools, fingerprint auxiliary loss and fingerprint class weights
remain unchanged; the in-batch negative composition naturally follows the
sampled positive molecules. Both conditions reset optimizer state equally.

Use independent parents general 20 (local) and general 21 (Kaggle), preserving all
27,600 exclusions. Compare each continuation with its parent and with the
ordinary continuation on native 234, hard185 and the exact physical panel. Do
not pool two seeds as468 independent targets: report each seed pair, then a
molecule-paired mean over seeds with its limited interpretation. Keep the
published blend weights fixed. Favor a production NP continuation only if the
effect repeats across parents and does not worsen the final panel proxy; no
threshold sweep or new rank-1 prior is part of this test.

PL13 real RTX2070 smoke passes both conditions: four actual optimizer steps per
condition, exact parent load, finite changed parameters, matched source hashes.
The 16-target toy diagnostic is explicitly unqualified and omitted from scientific
results. Smoke checkpoints are rejected by the evaluation loader. P09 is
packaged and will launch after P05 frees a Kaggle session; it has not consumed
a scoring submission.

## Completed T4 pair and an important curriculum caveat

P05 completes both 128×500 trials. General/native training times are 4,045/4,024
seconds, and both selected checkpoints are final epoch128. Training-index and
held-out identity hashes exactly match the local cohort. Native matching is
slightly platform-dependent: local 1,146,540 rows versus cloud 1,147,185, with 826
local-only and1,471 cloud-only rows (Jaccard99.7999%). Raw eligible input counts
agree; view sets do not. The cloud native trial covers163,855 training molecules
versus local 163,854. Each general/native pair is controlled within its platform;
the across-platform seed replication also includes this small preprocessing
difference. It must not be described as identical byte-level preprocessing.

| Seed21 condition | Native familiar MRR | Native unseen oracle MRR | Hard185 oracle MRR | Public50/new50 physical panel |
|---|---:|---:|---:|---:|
| General | 0.486625 | 0.477719 | 0.388844 | 0.926069 |
| Native-preferred | 0.471903 | 0.496541 | 0.418688 | 0.923870 |

Native preference changes familiar MRR by-0.014722 (paired interval
[-0.053119,+0.024562]) and unseen-oracle MRR by+0.018823 (interval
[-0.012981,+0.051631]). These mixed directions and wide intervals do not establish
a win. The local seed20 native trial is still running. General's fixed50/50 panel
blend improves in both independent fits (local 0.922160, cloud 0.926069), retaining
the original public channel. This supports preparing the production general 20
refit, while awaiting actual hidden scores.

A further exposure audit shows **only 44** of the 23,076 annotated NP training
identities have matched native views under this exact public-metadata join.
Their mean native-view probability is 0.001678. The NP curriculum consequently
reduces expected native-view exposure from 65.8957% to **33.0318%**. This is a
combined chemical-population/acquisition intervention, not an isolated chemical
mechanism. The annotations are incomplete and most matched NP training examples
come from public references. Record this before reading continuation results;
judge the available-data curriculum on native held-out spectra and the same
panel, rather than interpreting a reference-only gain as transfer success.

P09 launches at 20:33 UTC after P05's full-epoch and identity exclusions pass
inspection. Its general 21 parent is independent of the native-index discrepancy,
since the general trial uses the unchanged complete cache. Both32-epoch
conditions initialize from that exact same parent. Two daily scoring slots
remain reserved; only P01/P02 have been submitted.

## Local acquisition replication and first curriculum result

PL09 finishes at 21:02 UTC:128 complete epochs×500 steps, fixed final128 checkpoint,
3,470 seconds. Native20 gives familiar0.487527, unseen-formula0.502227 and
hard1850.429574. The unseen-oracle improvement over general 20 is+0.017455,
similar to seed21's+0.018823. However, public50/native 20-50 final panel MRR is
**0.916035**, versus general 20's0.922160 and baseline 0.917009. Native21 also
reduces that panel proxy versus its general parent. Do not promote a native-preferred
production refit: the final-ranking criterion fails despite the
retrieval-proxy gains. The 2070 immediately starts the general 20 continuation
control, followed sequentially by the NP-mixture condition (both seed23).

P09's two actual32×500 continuation runs finish with matching parent, cohort,
source and curated-key hashes. General/NP times are 1,124/1,172 seconds. Both use
the same 128-epoch general 21 parent and a fresh optimizer atLR2e-5.

| Native diagnostic | Ordinary continuation | NP curriculum | Delta | Conditional paired95% interval |
|---|---:|---:|---:|---|
| Familiar256 | 0.496931 | 0.544331 | +0.047400 | [+0.021214,+0.074490] |
| Unseen formula-oracle234 | 0.480416 | 0.541264 | +0.060848 | [+0.039203,+0.084171] |
| Unseen formula-oracle hard185 | 0.396760 | 0.476158 | +0.079397 | [+0.054322,+0.105757] |

The fixed public50/new50 final physical-panel proxy is 0.924107 for ordinary
continuation and **0.926416** for NP curriculum; the parent gives 0.926069 and
baseline 0.917009. The NP variant improves8 baseline molecules and harms0 in this
reused panel. The public50/v8025/NP25 alternative gives 0.922975. Keep both blend
families visible; no weights are fitted to these outcomes.

These are encouraging transfer diagnostics despite the lower native-source
exposure. The unseen formula-decoy identities are absent from all training and
also drawn from natural-product chemistry, so the result is more informative
than favoring a memorized truth over familiar decoys. It still does not isolate
one learned physical mechanism, establish scaffold-novel performance, or measure
hidden competition MRR. Confirm with the independent general 20 parent before
choosing a production curriculum. Full-data continuations may be prepared, but
must never be scored on their training identities.

Measured scheduling result: the first three128-epoch fits complete192,000
training steps in approximately87 minutes. Three serial fits at the measured
local duration would take173 minutes, giving about2× faster time to three models.
This includes staggered starts and native-index preparation, not a hardware-only
benchmark. Median500-step training epochs: RTX 2070: 20.75s; T4 general: 24.42s;
T4 native: 24.33s. Independent fits across the two sources provide the benefit.

## Independent continuation confirmation

PL14/PL15 complete the second parent pair on the RTX 2070: each runs all
32 epochs of 500 optimizer steps at LR 2e-5 with a fresh optimizer, from the
same excluded general20 epoch128 parent. Ordinary/NP times are 951/1,012 seconds.
The final epoch32 is selected in both conditions, without validation epoch selection.

| Parent20 condition | Native familiar MRR | Unseen formula-oracle234 | Hard185 oracle | Public50/new50 physical panel |
|---|---:|---:|---:|---:|
| Ordinary continuation | 0.495191 | 0.510077 | 0.422206 | 0.926632 |
| NP curriculum, q50 | 0.551073 | 0.548595 | 0.475880 | 0.927778 |

The NP continuation improves native retrieval over the ordinary continuation
in both independent parent pairs. Average the two fitted-pair effects within
each target, keeping the 234 targets as 234 observations. The mean unseen-oracle
gain is +0.049683; the molecule-paired 95% interval is [+0.031927,+0.067923].
Resampling 171 Bemis-Murcko target clusters gives a sensitivity interval
[+0.032370,+0.067572]. On the 185 ambiguous groups the mean gain is +0.066536,
with scaffold sensitivity interval [+0.047744,+0.086712]. These intervals are
conditional on the two fitted parent pairs and this reused target cohort.
They neither quantify population uncertainty across training seeds nor make
the test scaffold-disjoint from training.

The mean final-panel proxy is 0.927097 for NP versus 0.925370 for ordinary
continuation; the +0.001727 paired interval includes zero
[-0.000663,+0.005397]. Relative to baseline S3's exact physical-panel 0.917009,
the mean NP gain is +0.010088, with scaffold sensitivity interval
[+0.002586,+0.018822]. Each NP parent blend improves eight baseline targets and
harms none. These are reused-panel results, not a competition score or an
independent confirmation set. They support choosing the fixed q50 curriculum
over the other production-ready candidates, while awaiting Kaggle evidence.

## Final dose sweep and production qualification

The already-running local q25/q75 dose sweep completes at 22:13:42 UTC. Both
conditions use the same general20 parent, RNG seed23, LR 2e-5 and full 32x500
budget. Only evaluation and audits follow the user's request to stop after
this run; no additional training is started.

| Mixture coefficient | Native familiar | Unseen formula-oracle234 | Hard185 oracle | Fixed50/50 physical panel |
|---|---:|---:|---:|---:|
| q25 | 0.512071 | 0.524339 | 0.446371 | 0.926354 |
| q50 | 0.551073 | 0.548595 | 0.475880 | 0.927778 |
| q75 | 0.541699 | 0.572836 | 0.509696 | 0.927778 |

q75 improves the formula-oracle diagnostic, loses familiar retrieval against
q50, and gives no further panel gain. Its effect has only one parent pair;
there is no production q75 refit or real submission-path smoke. Keep it as an
unconfirmed future hypothesis. Select the two-parent-confirmed q50 candidate
for the requested final submission.

P07v2's actual T4 production general20 refit completes all 128x500 steps in
3,431 seconds. The excluded public-v4 architecture replica completes 109x500
in 3,264 seconds. Its native familiar/unseen/hard185 results are
0.422694/0.446936/0.373603; its conservative public50/v450 panel is 0.922168.
This does not support replacing the new NP curriculum with that replica.

P10 completes both full-data 32x500 continuations from the production parent:
ordinary 838 seconds, NP 884 seconds. There are 273,833 training identities,
25,916 annotated NP identities and 54.7321% NP probability mass under q50.
Neither full-data model is evaluated on its training targets. Real RTX 2070
four-step full-parent smokes establish optimizer/code integrity before P10;
their toy results are excluded from scientific comparisons.

P08 and P11 both complete the real 12-molecule Kaggle GPU pipeline smoke.
Each produces 400 valid rows, zero official metric-identity duplicates and
independent 50/50 z-score blend error 4.44e-16. P11's new model changes actual
candidate scores (maximum difference 3.581274), with all 12 real output rows
changed from the frozen fallback. Checkpoint, parent, stage, non-smoke flag,
architecture, epoch count, curated-key hash and full-data population are
guarded. Every guarded extension cell deletes `submission.csv` on failure
before reraising, preventing a silent submission of an earlier frozen CSV.
The pinned frozen module literal is preserved exactly.

P11's production checkpoint SHA256 is
`fd05f0333711c568dccdc551eecfd9123a88b4c306515801f1767763abfbf699`;
parent SHA256 is
`67ab9795ae72213bf89574bb2ca3fd6e6d624ea6b3f62ea53b45ab04f54c74b1`.
Its smoke CSV SHA256 is
`dbffacce6411197c91e0b2a7394d2917cdbf1059396720762aa20508a856b28e`.
The smoke CSV is a qualification artifact. The scoring submission uses the
kernel-version rerun, allowing Kaggle to execute the full hidden pipeline.

Validation totals: 54 unique checks pass across the CLI and GPU environments.
The full CLI suite passes 36 and skips 18 numeric checks; all 18 numeric checks
pass in the GPU environment. Candidate construction and failure-cleanup tests
pass, alongside the actual GPU optimization and production-path smokes.

## End-of-day submission and overnight pause

The latest user directive authorizes one final best-candidate submission and
then a pause. Submit P11 version1 from `submission.csv` at 22:20:35 UTC:
**ref58036642**, description marker `EOD11-20261105`. This is an explicit
one-time exception to the normal two-pending limit while P01/P02 are pending.
The final 22:26 UTC read-only API audit confirms three submissions today and
**two slots unused**. All three scores are pending; the best completed public
MRR remains **0.477**. No observed hidden improvement is claimed.

At 22:21:57 UTC, all project training, experiment controllers, score watchers
and hourly run-update processes are stopped. The process audit finds zero
remaining project workers; RTX 2070 utilization is 1% (desktop activity).
Submitted Kaggle scoring evaluations continue so they can return scores.
No training, automatic restart, submission or monitoring job is scheduled
overnight. Pause the active goal after saving and merging these records.
Resume only on a new user instruction, starting with the three pending
submission results. See the compact
[pause handoff](PAUSED-2026-11-05.md) and
[evidence directory](evidence/professional-2026-11-05/).
