# New public scan and prepared C08 — 8 November

Actual scores: P13 **0.480**, P14 v2 **0.458**. P14 completed without reported
error; the large conditional retrieval gain was a misleading promotion proxy.
Retain P01/P13 at0.480. The two authorized scoring slots are spent.

Reviewed20 selected sources from247 public catalogue entries. V7 documents a
compatible320-wide scorer and48-negative/pool512 recipe, but its0.6672 is a
truth-included random-holdout diagnostic, not a leaderboard score. Two
0.485/0.486-titled notebooks have identical code and an unavailable private-model
fallback; their titles are not reproducible evidence. Prvsiyan's lighter/heavier
pair differs only in extra-seed count and supports an isolated ablation.

C08 prepares fresh competition-only models: four precision/negative recipes,
two common identity/acquisition folds, batch384,40k successful updates each.
Actual data-loading/restart pilot is separate; the full run has not started.
Use [the protocol](docs/COMPATIBLE-LONG-RUN-2026-11-08.md). Real pipeline candidate
lists, absent-truth zeros, source-shift audits and failed-P14 negative control
are required before any promotion. No automatic submission or full-data refit.

---

# Historical: completed F07R1 and two accepted trials — 7 November

F07R1 completed100,000 head steps, zero restarts. The selected seed1/50% P01
fusion improves reserved conditional MRR .637481→.742594 after deterministic
cutoff-tie serving correction. This is not candidate recall or end-to-end MRR.
P14 v2 passed its full real Kaggle smoke, exact fusion and unchanged P01 baseline/
outsider checks,400-row official-identity dedup, unsupported fallback and local/
Kaggle portability audit. Submitted v2/ref58111204 at2026-11-08 01:52UTC.
P14 v1 was not submitted: NumPy1/2 unstable cutoff ties were diagnosed first.
P13 ref58105681 remains the independent outsider-blend trial. Both scores are
pending, with P01 **0.480** retained as champion. The two approved slots are used;
no extra scoring or training is queued. See`docs/P14-FROZEN-SUBMISSION-2026-11-07.md`.

---

# Active F07R1 / public trial — 7 November

F07R1 uses official frozen DreaMS/MoLFormer features with learned heads, fresh
head-excluded identity/input groups and training-only mass-near negatives.
Fragment m/z is filtered to<=1000Th; no superseded spectrum features are reused.
The original24h deadline and exact completed molecule cache are retained.
This tests a new representation rather than more of L06's unsuccessful external
spectrum dose. Truth-forced retrieval diagnostics do not qualify a production
submission; real candidate coverage and pipeline validation remain necessary.

Public-source P13 isolates consistent outsider blending on the verified P01
anchor. Its real12 smoke preserved P01 primary scores exactly and changed no
visible rankings. Independent execution/numeric/dedup audits passed. Submitted
v1/ref58105681 at22:04UTC; score pending. One further scoring slot is approved
for a qualified trained candidate, with no extra candidate authorized beyond two.
Best verified public score remains0.480. See`docs/FROZEN-TRAINING-2026-11-07.md`.

---

# Strategy

## 2026-11-06: shared-fold external-spectrum run

Verified scores: P01 **0.480**, P02 **0.479**, P11 **0.468**. Current best is
**0.480**. Yesterday's NP continuation failed to transfer despite positive local
diagnostics. New work follows Deotte's common-fold/OOF scaffold and compares
competition-only vs audited MassBank/GNPS spectra in ten sequential RTX2070
trainings. See [the plan](docs/LONG-TRAINING-2026-11-06.md). No automatic
submission; full-pipeline validation remains necessary.

L06 completed6 November at22:49UTC with all10 models/1,280 epochs and zero
restarts; final model and prediction hashes were checked. On316 reserved
confirmation molecules, control/external/equal blend/L2 stack retrieval MRR was
0.680621/0.646321/0.662922/0.671584. External's paired delta was-0.034299
(95% interval[-0.065787,-0.003513]); development gains did not confirm. The
NP-cohort blend gain+0.017801 has an interval including zero and is not an
independent confirmation. Do not promote external/blended models on this
evidence. No L06 Kaggle submission was made; the0.480 production baseline is
retained. No direct comparison of L06 control against production has qualified
a replacement. See the plan's completion section for numeric results/limits.

Subsequent explicit user override: one leaderboard trial was requested despite
unproven improvement. P12 original50/control-fivefold50 passed the real Kaggle
GPU smoke (400 valid rows/12 real molecules; exact fivefold mean, independent
blend error4.44e-16, zero metric duplicates) and was submitted23:50UTC on6 November,
version1/reference58071964. Hidden rerun failed with an unhandled error and no score (verified 7 November); root cause unconfirmed.


## 2026-11-05: candidate-based loss improves retrieval, tail fusion fails confirmation

Current best submitted public MRR remains **0.477**, verified from Kaggle submissions.
Two complete local RTX 2070 trainings, seeds 202611051/52, each 32 epochs × 500 steps,
tested candidate-only InfoNCE with 32 training-only near-mass/same-formula negatives.
This adapts the existing Saner model, motivated by MSAlign's candidate-based loss,
rather than implementing the MSAlign foundation-model architecture.

- New standalone kit retrieval MRR: **0.612352 / 0.612572** (top-1 0.481 each),
  versus historical mixed-loss 32-epoch seed-3 **0.577967**. Different seeds and
  negative counts prevent attribution to the objective alone.
- A 16-configuration, development-only fusion sweep selected 25% standardized new
  scorer, fingerprint coefficient 2, no prior correction. The public rank-1 decision
  stays fixed. Freeze that choice for seed 2.
- On 62 confirmation identities, current-recipe proxy **0.910455** → **0.910399 /
  0.910714**. Deltas **-0.000056 / +0.000259**, both bootstrap intervals include zero.
  Both local gates fail. No production refit/upload or competition submission.
- A separate cached NP sweep selected doubled fingerprint weight on 129 development
  identities (+0.008890), but deltas on 105 confirmation identities were
  **-0.003114 / +0.000990 / -0.006121** across three existing NP holdout models.
  Retain the existing coefficient.
- Public notebooks now report **0.480** (standardized neural ensemble plus other
  changes; public rebuilt scorer dependency) and **0.482** (missing overlay assets).
  We examined their sources; neither score was reproduced on our account. The 0.480
  reports replacement alone at 0.460, consistent with our earlier losses.

Conclusion: stronger artificial retrieval remains insufficient evidence for production
ranking changes. Better next evidence would be an identity-excluded, full-pipeline
natural-product replay with realistic candidates and a reserved confirmation cohort.
Do not retune on this experiment's confirmation outcomes and call them a fresh test.
Protocol, sources, math, artifacts and numeric evidence:
[`docs/SCIENCE-2026-11-05.md`](docs/SCIENCE-2026-11-05.md).

Snapshot 2026-10-28 ~19:30 UTC.

## Where we stand

- Our best before today: **0.338** (own pipeline v6, `cloud/workload.py`, 2026-10-08).
- Public board: top **0.516**; ~0.48 is a crowded band. 4,267 teams.
- Best reproducible public notebook: **0.477** — busyaprime
  [*LB 0.477 · Two 0.472 Fixes, Merged the Right Way*](https://www.kaggle.com/code/busyaprime/lb-0-477-two-0-472-fixes-merged-the-right-way).
  Heliosli's page shows ≥0.475 but needs a private dataset (its overlay silently skips without it).
- Nearly every public notebook ≥0.46 is the same lineage: Dmitrii Gluzdov's v4g assets → frozen 0.465 pipeline
  (library search, PubChem channel, GLACIER/ICEBERG forward models, two-engine fusion) → Saner Turhaner's neural
  spectrum↔structure scorer fused by reciprocal rank → leader-safe consensus → fragment check.

## How the 0.477 notebook builds an answer

1. Frozen 0.465 pipeline (SHA-pinned, base64-embedded) → baseline lists.
2. Neural scorer (`sanpier/casmi-spectrum-model-v1`) on every listed candidate + 20 outsiders/molecule.
3. Leader-safe consensus: rank 1 = argmax `1/(3+b) + 0.25/(3+n)`; ranks 2–25 by `1/(3+b) + 0.75/(3+n)`.
4. Fragment check (Roman Tamrazov's Spectral Parallax) on the V2 (0.25/0.50) order: a same-formula contender from
   ranks 2–6 takes rank 1 when the leader's library match < 0.87, ≥3 spectra, and fragment + neural + engine
   evidence agree. Also adjacent swaps and one rank-25 outsider.
5. Merge: keep only the check's leader move, old leader at rank 2, the 0.75 order below.

Knobs the field has already swept (no gain): w_mid 0.70/0.75/0.80/1.00, w_top 0 vs 0.25, RRF offset 1 vs 3,
provenance reranker, long-trained neural model (−0.019), NX isomer reranker (−0.058). Moving rank 1 with the neural
model more aggressively loses (w_top 0.5 → 0.448). **Rank 1 decisions are where points are won or lost.**

## Today's experiments (2026-10-28)

| Slot | Idea | Why |
|---|---|---|
| s1 | 0.477 verbatim | Control; lifts us from 0.338 |
| s2 | Spectral Parallax latest: 0.477 + "disagreement rescue" | Recovers leader moves the 0.477 merge discarded, with extra ion/neutral-loss/neural gates |
| s3 | 0.477 + relaxed merge | The 0.477 merge drops a fragment-check leader move whenever the check also replaced rank 25; keep the move |
| s4 | s3 + 3-slot neural tail rescue | Outsiders the neural model prefers take the last 3 places; tiny but positive expected value |
| s5 | s3 + library gate 0.87 → 0.92 | More molecules eligible for a fragment-check leader move |

## Offline evaluation: what the public panels can and cannot tell

- Saner Turhaner's labelled replay panel (`sanpier/catalogue-experiment-saner`, 120 held-out training molecules with
  pipeline lists, neural features and spectra) is far easier than the hidden test: V2 MRR 0.913 vs ~0.48 on the board,
  answer in the list for 119/120. Replaying the fragment check's rank-1 rule there (`kaggle/forks/x3-rule-replay`)
  fires on 1 molecule under every threshold tried (library gate up to "none", ≥2 spectra, ranks 2–8, neural margin
  0.15, fragment margin 0.07). It cannot rank these knobs; only the leaderboard can.
- Dmitrii Gluzdov's natural-product panel (250 NPs, fold-safe FPNet scores) is closer to the test (33% top-1 for unseen
  NPs) but lacks the pipeline's neural/fragment features; it would need a GPU replay to be useful.

## Tomorrow's queue (2026-10-29, `kaggle/forks/queue-2026-10-29.json`)

| Slot | Idea |
|---|---|
| t1 | s3 + looser rank-1 rule (lib 0.92, ≥2 spectra, ranks 2–8, neural margin 0.15) |
| t2 | s3 + neural scorer averaging `final_model.pt` and `last_validation_model.pt` |
| t3 | s3 + fold-safe FPNet vote: five models' f·z (engine protocol: single + merged views) blended into ranks 2–25 as z(neural) + 0.5 z(f·z); falls back to s3 if the stage fails |
| t4 | t3 at weight 1.0 (dose-response pair with t3) |
| t5 | s3 + looser rule + FPNet vote 0.5 — fallback, held until 12:00 UTC; replace it if the 10-28 scores point elsewhere |

FPNet-vote check on the labelled panel (CPU, `kaggle/forks/x4-fpvote-test`): CUSTOM 0.913 → 0.919 / 0.923 / 0.926 at
weight 0.25 / 0.5 / 1.0; f·z alone 0.778, neural alone 0.435; correlation 0.64. Leaky (panel molecules sit in four of
the five FPNets' training folds), so this only shows the stage works and points the right way.

How to read the 10-28 results: s1 is the control (expect ≈0.477). s3 − s1 = relaxed merge; s4 − s3 = tail rescue;
s5 − s3 = library gate 0.92. s2 − s1 = Parallax's disagreement rescue. Ties within ±0.004 are one molecule.

### 10-28 results (scored 2026-10-29 ~00:50 UTC, ~5 h after submission)

| Slot | Public | Reading |
|---|---|---|
| s1 control | 0.477 | Reproduces the public 0.477 — our new best (was 0.338) |
| s2 Parallax rescue | 0.477 | No public effect |
| s3 relaxed merge | 0.477 | No public effect (rule fires rarely); kept as the base — harmless and principled |
| s4 tail rescue | 0.477 | No visible effect at 3 decimals (ranks 23–25 move MRR by <0.0005) |
| s5 library gate 0.92 | **0.473** | −0.004 ≈ one molecule's correct leader demoted to rank 2. Loosening the rank-1 rule hurts |

Consequence: t1 (looser rule, already submitted 00:00 UTC) is expected ≤ 0.473; the t5 fallback (loose + FPNet) was
dropped. Rank-1 changes need a *new* independent signal, not looser gates on the fragment check.

### 10-29 results (scored ~05:00 UTC)

| Slot | Public | Reading |
|---|---|---|
| t1 looser rank-1 rule | **0.465** | −0.012 ≈ 3 correct leaders demoted. Second confirmation: loosening the fragment check loses |
| t2 final + last-validation average | 0.476 | −0.001: averaging a model with its own 90%-data checkpoint does nothing useful |
| t3 / t4 FPNet vote (buggy) | 0.465 | fallback to the frozen baseline (molecule-id bug); no information about FPNet |
| t3b FPNet vote 0.5 (fixed) | **0.470** | −0.007 ≈ two molecules. The fold-safe FPNet vote hurts ranks 2–25 on NP-like test molecules (its leaky panel check said +0.006); t4b (1.0) dropped |

t2 tempers the seed-ensemble expectation: a better neural score does not automatically move the board under the
0.25/0.75 fusion. The seeds are a different thing (independent full-data refits, +0.046 holdout MRR with 3), but
w1/w3 are the test.

### Slot-5 candidates for 2026-10-29 (pick after t1–t4 score, ~05:00 UTC)

FPNet as a second opinion on the fragment check's leader moves (`fpgate_patches` in `scripts/make_variant.py`): a move
needs the fold-safe FPNet vote to prefer the contender; with that agreement the library gate opens to 0.92. Without
FPNet scores the rule is the 0.87 original.

| Slot | Change vs s3 | Pick if |
|---|---|---|
| u1-fpgate092 | FPNet gate + library gate 0.92 (smoke commit: validates the FPNet stage on real pipeline files) | default |
| u2-fpveto | FPNet gate, library gate stays 0.87 (pure veto) | t1 ≪ s5, i.e. loosening is badly negative |
| u3-fpgate092-fp05 | u1 + FPNet vote 0.5 in ranks 2–25 | t3 > s3 |

FPNet facts (dataset README, `x5-fold-probe`): top-1 97% on molecules a model trained on, 68% on its held-out fold,
**33% on never-seen natural products** (the hidden test is NP-like). None of the five models saw the 250 panel NPs;
`casmi26-identity-disjoint-validation-folds` gives each training molecule's fold for out-of-fold checks.

## In flight (2026-10-29)

- **Smoke results (12 visible molecules, every stage):** u3 (FPNet vote 0.5 + FPNet gate) ran cleanly — FPNet scored
  12/12 molecules in 35 s, all consensus presets built, merge applied; no leader move on those 12. v33 completed but
  its residual never ran: all 12 smoke molecules have library match ≥ 0.87 (`library_protected`), so its scoring path
  is untested on real files (it falls back to the m1 order on any error).
- **Neural scorer seed ensemble.**
  - `n0-scorer-prep` (Kaggle CPU, 2 h): cache matches the public retrain on molecules and fingerprints
    (identical SHA-256); spectra differ by one row out of 2.43 M (dedupe on a newer image), so peaks/metadata
    hashes differ. Irrelevant for new seeds.
  - The kit model is under-trained (holdout diagnostic MRR still +0.01/epoch at epoch 16), but a public long-trained
    version lost 0.019 on the board, so the seeds keep the kit recipe exactly (16 epochs, full-data refit).
  - `n1-scorer-seeds` (GPU, 60 min; COMPLETE 04:07): 4 full-data refits (seeds 20262023…20265023) →
    `seeds/casmi26_seed_<k>.pt`, plus 2 holdout seeds. **Seed averaging is a large retrieval gain** on the kit's
    1000-molecule holdout diagnostic (256 candidates each, same-formula decoys first; not leaderboard MRR):

    | Holdout models | MRR@25 | top-1 | top-5 |
    |---|---|---|---|
    | single (kit seed / seed 1 / seed 2) | 0.484 / 0.489 / 0.483 | 0.344 / 0.353 / 0.334 | 0.654 / 0.662 / 0.665 |
    | average of 2 | 0.516 | 0.377 | 0.693 |
    | average of 3 | **0.530** | **0.393** | **0.716** |

    Not saturated at 3 (+0.015 from the third model). The first refit took 1501 s against ~290 s for the others
    (cold random reads from the mounted cache); n2 preloads the arrays into RAM.
  - `n2-scorer-seeds` (GPU, 21 min; COMPLETE 04:31): seeds 5–8, refits only (RAM preload: 17 s, then ~290 s/seed).
  - `w1-seedens` (smoke commit pushed 04:10): s3 with the neural scorer averaging `final_model.pt` + n1's 4 seeds
    (cos and fp terms averaged per candidate, t2's module patch; checkpoint list printed as `NEURAL CHECKPOINTS`).
    Smoke COMPLETE 04:31: `NEURAL CHECKPOINTS 5`, 688 listed + 240 outside rows scored, 0 input failures, the
    neural stage still takes seconds. First in the 10-30 queue.
  - `w3-seedens9`: w1 + n2's seeds (9 models), smoke commit pushed 04:33; fifth in the 10-30 queue. `w2-seedens-fp05`: w1 + FPNet vote 0.5, built.
  - Open question: with a stronger neural scorer the 0.25/0.75 consensus weights may no longer be optimal (the field
    tuned them for one model; w_top 0.5 lost with one model).
- **v33 real-list residual** (Ahmed Berat Özer, public, no LB score claimed): logistic regression on Saner's 120-query
  panel lists (ranks, engine/PubChem ranks, neural, simulator z-scores) re-sorts ranks 2–25 where library < 0.87;
  leaders fixed; any error restores 0.477. Panel scaffold-OOF 0.917 → 0.924. Fork: `v1-v33residual` (v1 COMPLETE).

## 2026-10-30 slate: the epoch dose-response (all five at 00:00 UTC = 8pm EDT)

The user asked for all five to go at 8pm EDT after a day of local testing (no 07:00 hold). Local holdouts showed
that training the scorer longer beats averaging more 16-epoch seeds, so the slate tests epochs on the board:

| Slot | Neural scorer | Holdout MRR (kit / NP), one model | Commit |
|---|---|---|---|
| w3-seedens9 | 9 × 16 epochs (kit + 8 Kaggle refits) | 0.484 / 0.382 (avg of 3: 0.530 / 0.403) | smoke ✓ |
| w6-seedens-e32 | kit final + 8 × 32 epochs (local, dataset casmi26-seeds-e32) | 0.58 / 0.43–0.47 | fast ✓ (no code change) |
| w7-seedens-e64 | 8 × 64 epochs (4 local + 4 Kaggle n6), no kit final | 0.66 / 0.55 | smoke ✓ (8 checkpoints, 0 failures) |
| w8-seedens-e128 | 4 × 128 epochs (Kaggle n7a/n7b), no kit final | 0.71 / 0.60 | smoke ✓ (4 checkpoints, 0 failures) |
| v1-v33residual | public v33 residual re-sort of ranks 2–25 (independent) | — | ✓ |

w4 (w3 + ranks 2–25 at neural weight 1.00) and w5 (17 × 16 epochs) hold slots 4–5 in the queue until w7 and w8
pass their smoke commits. Dropped: w1 (5 seeds; seeds saturate by 3 on natural products), u2 and t4b (FPNet lost
on 10-29). Reading the results: if score rises with epochs, the hidden test behaves like the holdouts and the next
step is longer training (and more seeds of it); if it falls, Özer's −0.019 generalises and the 16-epoch kit stays.

## Local GPU (RTX 2070, allowed from 2026-10-29)

`scripts/local_seeds.py` (venv `.venv-gpu`: torch 2.11+cu128, pandas 2.3.3 — pandas 3 is blocked by Windows
Application Control) trains and evaluates scorer seeds from the n0 cache (`outputs/local/n0/cache`, 2.5 GB) and the
kit (`outputs/local/kit`). It reproduces n1's holdout diagnostic exactly (kit 0.4844, seeds 0.4893 / 0.4826,
average of 3 0.5303); per-molecule z-scoring before averaging changes nothing (0.5304).

Training data by library (`x6-lib-keys`, Kaggle CPU): `enveda-180` (timsTOF, the test instrument) is 1.15 M of
2.43 M spectra and 183 k of 274 k molecules, 98.7% of them in no other library; pluskal (Orbitrap) 47 k molecules,
GNPS 46 k, RIKEN 16 k, others smaller. `enveda-np-examples` = the 250 natural products of Dmitrii Gluzdov's panel;
they also sit in RIKEN/GNPS/MoNA/… but not in enveda-180. 234 map to cache keys
(`outputs/local/seeds/np_holdout_keys.txt`); only 17 were in the kit's random validation split.

**NP split** (`--split np`): those 234 natural products leave training entirely and become the validation set (the
kit's random validation molecules go back into training). It is the closest available stand-in for the hidden
test's chemistry, and decides the refit recipe (ensemble size, epochs) on natural products instead of the random
holdout. Local holdout runs use the refit recipe (fixed epochs, last epoch kept, no early stopping): the kit's
`train_stage` keeps the best epoch on the validation set, which on the NP split selected on the scored molecules
(one seed stopped at epoch 10 with its epoch-6 model).

**The kit scorer is badly under-trained; seeds matter less than epochs** (2026-10-29, local diagnostics):

| Holdout | 16 epochs, single | 16 epochs, average of N | 32 epochs, single (epoch 16 → 32) |
|---|---|---|---|
| kit random 10% (1000 mol.) | 0.484 (mean of 3) | 0.517 (N=2), 0.530 (N=3) | 0.472 → **0.578**, still rising |
| natural products (234 mol.) | 0.382 (mean of 5; 0.366–0.401) | 0.399 / 0.403 / 0.404 / 0.405 (N=2–5) | 0.356 → **0.428** (peak 0.450 at 28) |

On natural products the seed ensemble saturates by N≈3 (+0.02), while 32 epochs give +0.05–0.07 per model; on the
random holdout 32 epochs give +0.10. 64-epoch runs, epochs 16 / 32 / 48 / 64:

| Holdout | 16 | 32 | 48 | 64 |
|---|---|---|---|---|
| kit random (seed 4) | 0.484 | 0.583 | 0.644 | 0.661 (peak 0.669) |
| natural products (seed 7) | 0.379 | 0.467 | 0.524 | 0.550 (peak 0.552) |

Three runs on two holdouts agree: the 16-epoch kit scorer is far from converged and gains continue past 64 epochs.
128-epoch runs (`n5-e128-curves`, Kaggle T4, seed 8), epochs 16 / 32 / 64 / 96 / 128:

| Holdout | 16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| kit random | 0.484 | 0.580 | 0.658 | 0.693 | 0.709 (max 0.711 @114) |
| natural products | 0.374 | 0.469 | 0.549 | 0.580 | 0.600 (max 0.628 @126) |

Flattening on the random holdout, still rising on natural products.

Averaging long-trained models still pays (local L7, three 64-epoch holdout seeds per split): kit singles
0.661 / 0.665 / 0.680 → average of 2 0.695, of 3 **0.708**; NP 0.550 / 0.518 / 0.539 → 0.561 → **0.575**. On natural
products the ensemble gain at 64 epochs (+0.04) is twice the 16-epoch gain (+0.02); three 64-epoch models ≈ one
128-epoch model. Next build: many 128-epoch refits averaged.

Cosine LR decay to 0 over 64 epochs (`n8-e64-cosine`) **loses**: kit 0.631, NP 0.497, against 0.658–0.661 and
0.549–0.550 at the kit's constant 2e-4. The model is under-trained, so decaying the rate only slows it;
`n10-e64-lr` tries the other direction (constant 4e-4 and 6e-4). The 2026-10-30 slate is the epoch
dose-response on the board: w3 (16 epochs, 9 seeds), w6 (32), w7 (64), w8 (128), plus v33. The contrary data point (busyaprime's notes in the 0.477 notebook): Ahmed Berat Özer's long-trained
`final_model.pt` scored 0.718 on Saner's random holdout but lost 0.019 on the board (e1 0.472 → e3 0.453). If it was a
full-data refit, as `final_model.pt` is in this kit, that 0.718 was measured on molecules it trained on; our holdout
runs never see their validation molecules, and the NP holdout is natural-product chemistry. Still a real risk: the
board is the only test of transfer to timsTOF natural products. `w6-seedens-e32` (final_model.pt + eight 32-epoch full-data refits trained locally)
tests it on the board; 64-epoch holdout runs check where the gain stops.

## 2026-11-02 board result: every ensemble of our refits loses ~0.018

| Slot | Neural scorer | Public | NP holdout (one model) |
|---|---|---|---|
| s3 | kit `final_model.pt` alone (16 ep) | 0.477 | 0.38 |
| w3 | kit final + 8 of our 16-ep refits | **0.460** | 0.38 (avg 0.40) |
| w6 | kit final + 8 × 32 ep | **0.458** | 0.43 |
| w7 | 8 × 64 ep | **0.453** | 0.55 |
| w8 | 4 × 128 ep | **0.459** | 0.60 |
| v1 | v33 residual, kit neural | 0.477 | — |

Flat in epochs and **below the frozen pipeline with no neural term (0.465)**; t2 (kit final + Saner's own
last-validation checkpoint, same seed) lost only 0.001. Holdout quality does not transfer to the public board.
Hypotheses:
1. **Public-LB selection.** The 0.477 recipe (fragment-check gates such as nmargin ≥ .23 SD, cos +.003, fp −.012;
   w_mid 0.75; merge rule) was tuned by many people on these ~125 molecules with exactly this kit model; any other
   scorer regresses. Then private may disagree, and neural work cannot be judged on public.
2. **A defect of our refits on test chemistry**, e.g. favouring structures seen in training. Panel check (run L14):
   our full refits memorize their training molecules more with epochs (neural-alone top-1 on panel molecules they
   trained on: e32 46, e64 65, e256 78 of 120; kit final 38, holdout e64 44). The NP holdout argues against this
   being dominant (its decoys are all training molecules and longer training still wins there).

Not a recipe mismatch: the kit's `training_result.json` (printed by n0) shows holdout best_epoch 16 and a 16-epoch
full refit — exactly what `local_seeds.py refit` does (same cache, code, config; only the seed differs). So w3's
loss comes from swapping/averaging seeds, not from how we train.

Diagnostics built (fast commits): `w20-seed1-e16` (one of our 16-epoch refits alone — same recipe as the kit, other
seed), `w21-seed221-e128` (one 128-epoch refit alone), `w22-kit50-e128x12` (kit model at half weight with 12 × e128).
w20 ≈ 0.46 → hypothesis 1 (public is tied to the kit seed); w20 ≈ 0.477 but w21 low → our long refits are worse on
test. Proposed as the first 2-submission batch of 11-03 (w20 + w21). The long-ensemble candidates w9–w19 are on hold
until these read.

## 2026-11-02: epoch slate submitted, longer refits training

The 10-30 slate never went out (work stopped 10-29 20:40); it was submitted unchanged on 11-02 16:51 UTC (w3, w6,
w7, w8, v1). Scores expected ~22:00 UTC.

Unrecorded curves from 10-29 (7-epoch means around each point):

| Run | Holdout | 16 | 32 | 64 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|
| n9 seed 10, 256 epochs | kit random | 0.482 | 0.589 | 0.666 | 0.715 | 0.721 | 0.719 | 0.720 (max 0.732 @170) |
| | natural products | 0.358 | 0.424 | 0.535 | 0.582 | 0.579 | 0.598 | 0.607 (max 0.634 @227) |
| n10 seed 11, LR 4e-4 | kit / NP at 64 | 0.511 / 0.424 | 0.612 / 0.501 | **0.645 / 0.540** | | | | |
| n10 seed 11, LR 6e-4 | kit / NP at 64 | 0.513 / 0.414 | 0.589 / 0.480 | **0.629 / 0.521** | | | | |

- The random holdout plateaus from ~160 epochs; natural products still gain a little to 256 (+0.025 over 128).
- A higher constant LR learns faster early but ends lower at 64 epochs (2e-4: 0.658–0.661 / 0.549–0.550). Keep 2e-4.

Training launched 11-02 (GPU quota reset 10-31; 0 h used at the start):

| Run | What | For |
|---|---|---|
| n11 (Kaggle T4 ×2) | refits 225–228 at 128 epochs | w9 = 8 × e128 (with n7a/n7b) |
| n12 (Kaggle T4 ×2) | refits 241–242 at 256 epochs | w10 = 4 × e256 |
| L9 (local 2070) | refits 243–244 at 256 epochs → dataset `casmi26-seeds-e256` | w10 / w11 |
| n13 (Kaggle, queued) | capacity: width 256 / 4 layers and width 384 / 4 layers, 64-epoch holdouts on both splits | does a bigger scorer learn more per epoch? (kit model: width 192, 3 layers, 3 M params) |

**Panel replay of the rank-1 weight** (`scripts/panel_replay.py`, run L10). All 120 molecules of Saner's labelled
panel sit in the kit's validation split, so kit-split holdout scorers (kit best-validation + n1 seeds at 16 epochs,
local 32/64-epoch holdouts, n5 seed 8 at 128, n9 seed 10 at 256) never trained on them. Their neural scores replace
the panel's and the CUSTOM order is replayed (no fragment check / merge). Candidate fingerprints: `x8-panel-fps`.

| Scorer | neural-alone top-1 | 0.25/0.75 MRR | 0.35/0.75 (leaders fixed / broken) | 0.50/0.75 |
|---|---|---|---|---|
| none (baseline) | — | 0.916 | — | — |
| e16 × 3 | 36/120 | 0.910 | 0.914 (+1/−0) | 0.903 (+3/−5) |
| e64 × 3 | 49 | 0.918 | 0.926 (+3/−1) | 0.905 (+4/−7) |
| e128 | 43 | 0.920 | 0.928 (+2/−0) | 0.907 (+3/−6) |
| e256 | 56 | 0.919 | 0.927 (+2/−0) | 0.923 (+3/−2) |
| e128 + e256 | 53 | 0.921 | 0.929 (+2/−0) | **0.933 (+4/−1)** |
| e128 × 3 (seeds 8/13/14, run L11) | 49 | 0.919 | **0.931 (+3/−0)** | 0.919 (+3/−3) |
| e128 × 3 + e256 | 52 | 0.920 | **0.932 (+3/−0)** | 0.932 (+5/−2) |

With w_top 0.30 / 0.40 as well: 0.30 never breaks a leader for any scorer; 0.40 is neutral for e128 × 3 (+3/−3) and
positive with the e256 model added (+3/−0). **0.35 is the robust choice** (gains for every scorer, worst case +2/−1);
0.50 only pays with the strongest mix.

At 0.25 the neural never moves a leader here. 0.35 gains with every scorer; 0.50 loses with 16-epoch models, which
matches the board (the field's w_top 0.5 → 0.448), and turns positive only for the strongest ensemble. The panel is
easy (baseline top-1 105/120) and small, so these are 1–4 molecule effects — but the direction agreed with the board
on the one rank-1 knob the board has already judged.

Ensembles still pay at 128 epochs (run L12, kit holdout, seeds 8/13/14): singles 0.709 / 0.705 / 0.709 → average
of 2 0.739, of 3 **0.745** (top-1 0.63). Kit holdout, 3-model averages: 16 epochs 0.530, 64 epochs 0.708, 128 epochs
0.745.

**Capacity** (`n13-capacity`, 64-epoch holdouts, seed 12; `local_seeds.py --set key=value`): width 256 / 8 heads /
4 layers (5.0 M params, kit 3.0 M) reaches kit 0.519 / 0.629 / 0.670 / **0.686** at epochs 16 / 32 / 48 / 64 and NP
0.391 / 0.476 / 0.520 / **0.546** (last epoch 0.571), against 0.658–0.680 and 0.518–0.550 for kit-width seeds, at
1.3× the time. A real but smaller gain than doubling epochs (+0.05). Width 384 crashed (shared run folder, fixed).
Follow-ups: `n15-capacity2` (width 384 retry; width 256 at 128 epochs on NP), `n16-w256-refits` (two width-256 refits
at 128 epochs) → `w18-e128x12-w256x2`, which patches the pipeline's `load_model` to build each checkpoint from its
stored `cfg_overrides` (smoke commit).

From 2026-11-02 the user's rule is **at most 2 submissions at a time, spread across the day** (CLAUDE.md); the
10-30 slate's five went out together before that rule. On 11-03: 2 at 00:00 UTC, the other 3 later in the day.

2026-11-03 slate candidates (`scripts/make_variant.py`, all fast commits — only attachments or a CUSTOM
constant differ from w8's smoke-tested notebook): w9-e128x8, w10-e256x4, w11-long12 (8 × e128 + 4 × e256),
w12-e128x8-mid10 (ranks 2–25 neural 1.00), w13-e128x8-top35 (rank-1 neural weight 0.35: the #2 candidate takes rank 1
when the neural ranks it first and the leader 7th or lower; at 0.25 the leader must fall to 18th), w14-mix25 (kit
final + 8 × e32 + 8 × e64 + 8 × e128), w15-long12-top35 and w16-long12-top50 (w11 at w_top 0.35 / 0.50). Ready fallbacks if long training loses on the board: w5-seedens17 (17 × 16
epochs), w4-seedens9-mid10.

## Ideas backlog (next days)

1. **Fold-safe FPNet evidence** (`dmitriigluzdov/casmi26-fold-safe-fpnet`, already attached): heliosli's private
   same-formula isomer reranker built on it gave +0.007 on the 0.468 base. Train our own on Kaggle GPU using the
   natural-product panel benchmark as labels; apply only to same-formula contenders at ranks 1–6 with the FREEZE rule.
2. **Our own Phase-1 signals as rank-1 votes**: library entropy similarity, analog propagation, MetFrag-lite, popularity.
3. **Recall**: answers missing from the 25-list can never score; check list lengths and fill short lists from the
   PubChem/COCONUT channel.
4. Combine whichever of s2–s5 wins.
