# Why the current approach has stalled at 0.480

Research completed 11 November 2026 using existing artifacts and public primary
sources. Training, cloud launches and submissions remain paused. No new model
was trained or evaluated on fresh reserved labels.

The evidence points to an unreliable validation proxy, missing candidates and
restricted isomer ranking. It does not establish a hard 0.480 ceiling or identify
the hidden molecules responsible for each regression. At 15:12 UTC the live
authenticated board had a top score of 0.540 and 891 teams above 0.480; our best
remains 0.480. These are leaderboard observations, not reproducible recipes.

Both requested past-work trials have finished: P18/C22 ref58212322 and P19/C18
ref58219034 each scored **0.478**, COMPLETE without a reported error. The observer
sent the second only after the first result, then exited at 05:34 UTC and restored
the pause. Fresh account reconciliation at 15:10 UTC found zero pending entries.

**Our local improvements have repeatedly failed to predict Kaggle.**

| Submitted change | Offline evidence before submission | Actual public result vs 0.480 |
|---|---|---|
| P14 frozen-feature channel | Truth-included conditional retrieval: 0.637481 to 0.742594 | 0.458; -0.022 |
| P16 external references | Full external80 pipeline: 0.713859 to 0.837445 | 0.480; no reported gain |
| P15 compatible tail | Reserved64: +0.000254; interval includes zero | 0.479; -0.001 |
| P18/C22 peak-drop dose | Used240: +0.001171; 95% interval [-0.000479, 0.002943] | 0.478; -0.002 |
| P19/C18 peak drop | Used240: +0.000815; 95% interval [-0.000895, 0.002607] | 0.478; -0.002 |

These offline panels measure different tasks and populations; their absolute
scores must not be compared directly. P14's conditional diagnostic guaranteed
the truth was among 256 candidates and omitted real candidate generation. Its
large gain could never establish candidate recall. Later complete used240
validation still gives P14 a positive point estimate (+0.001659) despite its
known public regression. The negative-only P14 control is an exact fallback,
so it supplies no evidence about negative-mode model quality.

The original C08 acquisition audit also found cached model-input group overlap
for 67 of 128 queries, with large source differences: 55/64 truths present in
one source versus 9/64 in the other. The C09 raw-spectrum audit flagged 20/80
queries, all from its GNPS cohort. These are bounded overlap warnings, not proof
that every label leaked. Even acquisition/identity exclusions do not establish
the same chemistry, public-pretraining exposure or class mixture as hidden data.

Selecting the two largest point estimates among 13 already-used recipes adds
selection bias. Both selected confidence intervals already included zero, and
their prior promotion gates had failed. The two entries were explicit scoring
trials; choosing the best available retained fits did not make them reliable
improvements. More training time cannot repair this measurement problem.

**The recent changes cannot address most observed error types.**

A new CPU-only calculation independently checked the exact production parent
and submitted C18/C22 recipes against the already-consumed 240-query panel:

| Parent outcome | Queries | Consequence for current tail-only changes |
|---|---:|---|
| Correct at rank 1 | 46 | Already scores 1; protected |
| Correct candidate present, below rank 1 | 104 | Can move within the tail, but cannot reach first |
| Correct candidate absent from final25 | 90 | Remains zero regardless of neural ranking |

Thus 37.5% of this panel is unreachable without changing candidate membership.
The other major problem is distinguishing structures: **96 of the 104** wrong
first choices have the same molecular formula as the correct candidate. Better
mass estimates or formula filters alone cannot distinguish those structures.
This is a measured used-panel result, not a hidden-test error distribution.

Candidate attrition is partly avoidable with existing assets. The earlier full
positive160 audit found the truth in 100 final lists but in 119 deeper existing
pools: 19 answers were available earlier and did not survive final selection.
The remaining 41 were absent from those deeper pools too; we have not fully
separated database absence from mass/adduct filtering or earlier truncation.
LOTUS did not increase the union beyond 119, so merely adding that database is
not a new solution supported by our own coverage audit.

The integration rule further limits the model. It preserves the first candidate
and uses `1/(3+baseline_rank) + 0.2/(3+neural_rank)` for the tail. Even a perfect
neural order cannot move an original rank25 answer above rank10 under this rule;
an original rank10 answer cannot rise above rank6. The metric rewards a 2-to-1
promotion by `0.5/N`, versus `0.1/N` for a 10-to-5 promotion, where N is the
number of scored molecules. The recent tail stage excludes the former action.
This follows directly from the [official MRR@25 definition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/).

On this used panel, answer-aware mathematical ceilings are 0.625 with arbitrary
ranking of the existing candidates, 0.40833 with the first candidate locked,
and 0.31935 under the actual fixed0.2 rule, versus parent0.28375. The last
calculation maximizes over every possible neural order and was checked against
exhaustive enumeration on 21 small cases. These use known answers and are not
attainable performance estimates. **0.31935 is not a ceiling on Kaggle MRR**;
this panel has a different distribution from the hidden test.

C18 and C22 changed 237/240 lists, but only 48 truth ranks changed for either
model. C22 improved26 and harmed22; C18 improved24 and harmed24. The two models
give the truth the same rank on 227/240 queries. They share architecture, parent
fits, augmentation and integration; C22 continues C18 for another10,000 updates
per fold. Their many changed strings do not imply independent useful evidence.
Equal rounded public scores do not prove their hidden lists are identical.

**The evidence does not point to failed GPU training as the main explanation.**

The completed training/checkpoint receipts show real updates and finite models.
The new Kaggle execution audits independently reproduce mean-of-standardizations
and RRF arithmetic, exact final lists locally versus T4, unchanged parent scores,
official identity deduplication, rank-one protection and fallback. Maximum raw
local/cloud differences were around 1.9e-6. The real smoke exercised12 queries;
its 400 valid output rows include388 exact fallback rows. It verifies execution,
not quality on400 hidden queries.

There are observability limits. In the retained inference source,
`ValueError`/`TypeError` can skip an acquisition while `input_errors` is written
as literal0. Unsupported molecules are reported separately. Hidden per-query
diagnostics are unavailable here. We therefore cannot use that zero counter to
prove all hidden acquisitions were usable. The demonstrated preflight cleanup
gap was fixed before C18; there is no evidence that it executed in P18. Neither
the successful smoke nor the aggregate0.478 alone proves the exact hidden loss
mechanism. No scoring source was changed during this research.

Public research supports these concerns, while supplying no automatic fix.
[MassSpecGym](https://arxiv.org/html/2410.23326v3) identifies instrument/preprocessing
heterogeneity and shows that identity-only splits can retain near-duplicate
chemistry. A [CASMI validation discussion](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748524)
reports sensitivity to source, public-pretraining exposure and class assumptions;
its estimated hidden class shares are hypotheses and are not adopted here.
Another [firsthand discussion](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254)
describes isomer errors and conflicting panel conclusions, consistent with our
independent formula audit. A [2 November participant report](https://nihalsulaiman.com/blog/leak-free-validation-casmi/)
describes exposed-panel gains reversing on disjoint full-pipeline validation.
A [public experiment log](https://raw.githubusercontent.com/shiferaxa/casmi26/main/docs/plan.md)
also documents better fingerprint retrieval failing to improve the full pipeline.
Those are supporting observations, not measurements of our hidden test.

**The next justified work, if training is resumed, is different from more C08 doses.**

1. Repair the validation decision process first: use complete pipeline outputs,
   absent-truth zeros, disjoint structure/acquisition groups and source/mode
   breakdowns. Use P14 as a known harmful control, then assess a frozen protocol
   independently; do not keep redesigning a panel until old leaderboard signs fit.
   The fresh C13 reserved metrics remain unread and must stay separate from design.
2. Attribute candidate losses by stage, starting with the19 already present in
   deeper pools. Measure recall before filtering, after each prune and in final25.
   Candidate membership improvement must survive an actual ranking test; oracle
   presence alone does not qualify a change.
3. Target isomer discrimination and train any future fusion/ranker on features
   produced by the actual serving models and candidate lists, using out-of-fold
   scores. Test controlled top1 promotions in validation. Simply increasing a
   fusion weight is unsafe: P14 already demonstrated that risk.
4. Count acquisition failures and unsupported queries honestly and retain reason
   codes. Keep execution qualification separate from evidence of a quality gain.
5. Freeze one primary comparison and selection rule before confirmation. Require
   a material repeatable gain over the champion, not merely a better matched
   weak control or the largest tiny gain after many trials.

No new training, cloud job, submission, model adoption or fresh reserved-label
evaluation was performed. The score observer exited normally; the two remaining
local dashboard workers were stopped, and the hourly automation is PAUSED.
The obsolete C09 coverage kernel still reports CANCEL_REQUESTED; termination is
not acknowledged, and it has not been requeued. The silver goal remains paused.

Reproducible analysis: `scripts/research_score_plateau.py` and
`scripts/research_plateau_isomers.py`. Compact account, board, source-review and
numeric receipts are in `docs/evidence/plateau-20261111/`. Raw labels, per-query
results, models and caches remain outside Git.
