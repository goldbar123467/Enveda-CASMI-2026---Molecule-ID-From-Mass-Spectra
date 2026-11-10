# C12 deeper candidate pools — 10 November 2026

P16/ref58181280 remains the only pending public submission at05:23UTC. The
additional user-approved trial remains unconsumed. C12 reuses existing private
full-pipeline query artifacts and unchanged champion weights; it is not another
training fit. Preserve P16 and the0.480 champion.

## Development diagnosis and fixed rule

On the40 already-used C10 development queries, final P01 lists contain20truths;
the existing20-outsider pool adds2, while all earlier engine/PubChem/ICE/GL/N1
pools together contain25. These are oracle presence counts, not achievable MRR.
Ten supplied engine/PubChem keys disagree with official identity, so the new
pool recomputes RDKit2026.03.3 metric identities and deduplicates them.

The fixed recipe uses the complete P01 plus fixed P16 reference parent. It
seeds the parent/original neural candidates, then takes a deterministic round
robin from earlier sources, capped at128 identities/query. The exact P01 public
and v80 checkpoints each contribute0.5 after joint-pool per-query score
standardization. RRF weight0.20/offset3 may insert at most3 outsiders, retaining
rank1 and the exact relative parent prefix, dropping only the old tail. Library
confidence>=0.90 is protected. There was no coefficient search.

Config SHA256: `0c4ad92554b8d1f91dd79c0df6949f1f58d306d08327ea1401e215a9aa42143a`.
Source-contract SHA256: `e4f9208dbd1decfede25290f3ed47e1bcd4e347c77e396982ab25a8be70a8be5`.
Do not change the frozen config/prepare/score/prefix implementation during the
confirmation check. Four meaningful merge tests pass, including insertion
position, prefix preservation, maximum count and exact protected/fallback paths.

Development preparation/scoring completed04:54/04:57UTC:40queries,
3713candidates,zero errors. Raw baseline scores match the real Kaggle outputs:
public maximum absolute difference1.90735e-6, v80 2.38419e-6, under1e-4.
Two CPU threads,70%VRAM ceiling,4GiB free-RAM floor and a bounded scoring
deadline apply. Development peak Torch reserved memory was0.0332GiB;
that is not total Windows GPU allocation.

## Development result and prospective confirmation

Evaluation completed05:01UTC: P01/P16 MRR0.2722727273→C12 0.2735802720,
delta+0.0013075448,95%paired bootstrap interval[-0.0004891304,0.0044117647],
1query improved/1harmed. Truth coverage20→21;one missing truth recovered,
zero truths lost. There were98insertions across38queries, at most3/query;
one query had no selected outsider and one was library protected.

The positive point estimate plus one new truth satisfies the predeclared
development selection rule. Its interval crosses zero: it does not qualify a
submission. Selection and source pins froze at05:01UTC; confirmation protocol
froze at05:08UTC before any reserved prediction metrics were read.

All160 label-blind preparation and2070 scoring completed05:23UTC:
14184candidate rows/13673uniqueSMILES,zero invalid strings or input errors.
Both models support160queries; raw local/cloud parity maxima2.38419e-6 and
2.86102e-6. Before reading120 metrics, the evaluator verified source/data hashes,
support/parity, and exact40dev output equality, then sealed prediction inputs.

The fixed gate requires a positive paired95% lower bound versus both complete
P01 and P16; input-alias-excluded sensitivity must be nonnegative. The two
reference-match views were identical before labels, withzero bounded input
aliases. All missing-truth zeros remain in the denominator. Report P14/P15
negative controls regardless of their signs. P14's wrong development sign
relative to its known public regression limits hidden-score calibration.

The once-only120 evaluation completed05:23:29UTC and **failed** every gate.
Complete P01/P16 MRR0.2809014920→C12 0.2808782919,delta-0.0000232001,
CI[-0.0016050362,0.0017152707],2improved/8harmed. Truths80→80,with2new
truths and2lost. Alias sensitivity is identical. P14delta+0.0012751 again has
the wrong sign; P15delta-0.00130585 has the known regression direction but its
interval crosses0. These control limitations remain in the sealed report.

C12 is rejected, not submitted. No changed-path smoke is warranted for this
failed recipe. All C12 processes exited; do not restart completed work. The120
metrics are consumed, cannot be reselected on or called fresh, and any next
candidate requires independent confirmation data. The extra approval remains
unconsumed. Latest05:23UTC actual rank934/5048,score0.480; cutoff252/0.483.

Local artifact root: `outputs/local/c12-deep-candidates-20261110/`.
Compact evidence is under `docs/evidence/c12-2026-11-10/`; raw labels,
predictions, spectra, candidate caches and models stay outside Git. The panel
is positive-only external CASMI2022 with unknown collision energies; bounded
overlap audits do not prove universal historical pretraining exclusion.
