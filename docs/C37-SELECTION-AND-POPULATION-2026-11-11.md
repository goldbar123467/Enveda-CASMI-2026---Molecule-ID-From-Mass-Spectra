# C37: selection losses and validation population support

The CPU-first investigation is complete with a **no-go for GPU work, a new
retention policy, or scoring submission**. The best public score remains0.480.
No complete-pipeline improvement was qualified and fresh C13 was not opened.

The user cleared the old paused silver objective and authorized this new goal.
Its conditional path explicitly allows a validation-population audit when no
defensible selection mechanism emerges. That is the outcome here. We did not
invent a quota change from the known missing answers or consume the reserved
holdout to search for a favorable result.

## What the26 losses actually are

Existing COMPLETE C10v1 and C20v2 outputs were downloaded read-only. No cloud
job was started. The embedded frozen notebook payloads match their execution
receipt hashes. The saved candidate inputs and final frozen submissions also
match the corresponding run receipts. C36's consumed240-query population and
official identity convention are retained, including every missing-truth zero.

| Observed selection decision | Positive | Negative | Total |
|---|---:|---:|---:|
| Five PubChem slots already occupied by earlier proposals |7|3|10|
| Main list truncated from top60 to top25 |3|1|4|
| PubChem interleaving displaces main rank22 |1|0|1|
| Reciprocal fusion places answer outside top40 |4|0|4|
| Same-formula GL ranking demotes answer across rank25 |2|0|2|
| Same-formula fragment-gap ranking demotes answer across rank25 |0|1|1|
| Answer already outside25, still absent after GL ranking |2|0|2|
| Answer already outside25, still absent after fragment-gap ranking |0|2|2|
| Total |19|7|26|

All10 PubChem losses have library confidence below0.90: library protection did
not block them. Their saved merged lists contain five earlier PubChem proposals,
so the observed cause is the fixed insertion budget. Main physical reranking is
disabled in these executed sources (`BASE_ICE=False`); four saved main answers
are below25 before interleaving, while main rank22 is displaced by five proposals.
The four fusion losses are verified against the untruncated fixed fusion order.

The seven physical-stage witnesses must not be described as seven incorrect
demotions. Only three cross the25 boundary; four start outside25. Saved lists
establish these boundaries. Internal reranked positions26–40 and the original
negative fragment-score dictionary were not retained, so exact within-tail
positions or numerical score causes are not claimed.

The exact executed Engine2 rank-2 overlay was replayed on all240 saved E006
outputs, matching every saved SMILES list. Although it changes membership on
three positive queries, **it displaces zero true answers and adds zero true
answers**. The later CUSTOM/P01 stages preserve membership on all240. This
rejects the suspected late-overlay explanation for the26 losses.

Expanded saved main/overflow/source lists still contain176 true answers,150 in
the final lists. None of the former64 route-absent answers is newly located.
This does not establish absence from the full original databases.

These fixed cutoffs and ranking choices are explicit policies. Their errors do
not establish an implementation defect or a beneficial replacement. No
label-independent correctness invariant justifies a new retention rule from
this trace, so no rule or hyperparameter sweep was frozen/tested. A future
proposal must count the answers it displaces as well as those it recovers.

## Confirmed formate compatibility defect

C20's receipt-pinned legacy `chem.py` recognizes `[M+CH2O2-H]-` and `[M+HCOO]-`
but rejects equivalent `[M+FA-H]-`. Both the main worker and PubChem worker
compute neutral mass through this helper and skip queries with no finite mass.
All25 affected negative validation queries have no saved PubChem record. This
is a genuine lookup/eligibility defect, separate from the26 selection policies.

`scripts/legacy_formate_compat.py` prepares a narrow source adapter for the
legacy chemistry and fragment-offset boundaries. It preserves shared query
tokens, the original model vocabulary and every existing index. The separate
encoder's adduct handling is not rewritten. Unrecognized tokens retain NaN
mass and the unknown model index; no proton default is introduced.

The CPU audit tests all59 supported adducts,177 supported mass vectors, scalar
inverse lookups and fragment offsets. Existing215 query masses are byte-exact;
finite masses increase215→240. Formate fragment offsets now match the canonical
spelling. Five regression tests cover vocabulary shape, charged round trips,
unknown-token behavior, fragment equivalence and refusal of source drift or a
second patch.

**The adapter is prepared locally, not installed in production or executed in a
new model run.** Missing logits/candidates cannot be reconstructed from saved
final lists. The complete public test file contains zero `[M+FA-H]-` tokens;
its formate spectra already use the recognized spelling. Repairing this
validation discrepancy therefore supplies no demonstrated hidden-test gain.

The credited source files come from the existing public reproduction asset
dataset, with exact hashes matched to both mounted legacy copies in C20's
inventory. Source provenance and patch hashes are in the compact audit receipt.

## A more representative population must preserve acquisitions

This audit uses the complete public unlabelled test file, not the12-query
visible smoke output. It reads query spectra/metadata only, without prediction
quality or prior leaderboard outcomes to choose the population.

| Query/acquisition property | Public test | Used primary | Historical source |
|---|---:|---:|---:|
| Molecule queries |400|240|80|
| Spectra |1213|240|302|
| Positive-only queries |276|160|30|
| Negative-only queries |27|80|14|
| Mixed positive/negative queries |97|0|36|
| Metadata instrument |timsTOF|FT|LC-ESI-QFT/Orbitrap|
| Queries with exactly one spectrum |55|240|6|
| Queries with multiple adducts |108|0|36|

Target dominant adducts are347 protonated,38 deprotonated,8 formate,5 sodium
and2 ammonium queries. Primary has87 protonated,47 deprotonated,25 formate,
36 sodium,37 ammonium and8 chloride. Dominant adduct is a deterministic
plurality summary; full adduct sets and mixed queries remain explicit locally.

Mode/dominant-adduct strata cover only75.75% of target queries in primary and
96.75% in the historical source. Covariate importance weights would have
effective sample sizes108.52 and48.83 respectively. These are support diagnostics;
no weighted MRR was computed. Exact target instrument support is zero in both
panels. Weights, repeated rows and acquisition subsampling cannot supply the
missing acquisition source. These discrepancies are evidence of limited support,
not proof of a single causal explanation for leaderboard behavior or a claim
that an unseen chemical population is now characterized.

The prospective protocol is frozen in
`docs/evidence/c37-20261111/prospective-validation-protocol.json`. It requires:

- Complete original usable acquisition sets per molecule, including both
  polarities; no convenient one-scan surrogate.
- At least two documented external acquisition sources, including timsTOF,
  with one source excluded from candidate/model selection.
- A deterministic draw within target metadata strata, independent of scores
  and candidate truth coverage; report missing support rather than fabricate it.
- Chemistry-group isolation and exact/near acquisition and model-exposure
  audits, with unknown pretraining exposure disclosed.
- All-query MRR@25, missing truths zero, recovered/displaced answers, correct
  first-choice gains/losses, and paired chemistry-cluster uncertainty by source
  and actual query acquisition mode.
- One frozen complete-pipeline candidate before source outcomes, followed by
  untouched confirmation only after the earlier checks pass.

No qualifying new source panel was constructed or claimed. Fresh C13 remains
sealed; this metadata audit does not establish that it supplies the missing
target instrument support.

## Corrected historical polarity reporting

C36 assigned each historical query the mode of its first record. Its
“negative31” subgroup contains14 exclusively negative queries plus17 mixed
queries; “positive49” contains30 exclusively positive plus19 mixed. These labels
must not be interpreted as pure polarity subgroups.

The already frozen C36 primary ranker's predictions were recounted with all
acquisitions. No new fit, prediction, selection or holdout access occurred.

| Correct source subgroup | Queries | MRR delta | Grouped95% interval |
|---|---:|---:|---|
| Positive-only |30|+0.063889|[-0.030882,+0.165179]|
| Negative-only |14|-0.072066|[-0.166667,0.000000]|
| Mixed |36|-0.003968|[-0.064060,+0.035992]|

All80 delta+0.009561 and the previously fixed clean60 delta−0.023909 are
unchanged, including their intervals. C36 remains rejected. Small subgroup
intervals are not evidence of reliable improvement or a reason to choose
another favorable subgroup. Original sealed C36 artifacts remain unchanged.

## Verification and decision

Eighteen targeted tests pass:10 prior validation safeguards,5 formate adapter
regressions and3 mixed-query/support tests. A separate arithmetic review verifies
all240 overlay outputs, membership conservation into the final parent, all26
counts, current source hashes and public target counts. A fixed64-structure
sample of the extended identity cache matches the actual executed official
identity helpers. This is a spot check, not a new full-database identity audit.

**Go:** use the corrected query-mode reader and the frozen source-support
protocol for the next validation-data work; retain the tested formate adapter
for controlled future integration. **No-go:** new retention-policy promotion,
fresh C13 consumption, GPU training/inference, cloud execution or submission.
No new public improvement or silver position is claimed.

Reproducible code: `trace_c37_selection.py`, `refine_c37_boundaries.py`,
`audit_c37_formate.py`, `legacy_formate_compat.py`, `audit_c37_population.py`,
`relabel_c37_source_modes.py`, `review_c37_selection.py` under `scripts/`.
Compact aggregate receipts are under `docs/evidence/c37-20261111/`. Raw source
downloads, candidate lists, identity cache, per-query witnesses and unlabelled
test data remain ignored/local under `outputs/local/c37-selection-trace-20261111/`.
