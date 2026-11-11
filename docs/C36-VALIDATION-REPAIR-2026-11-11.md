# C36: validation safeguards, candidate losses and CPU isomer ranking

The user's 11 November instruction authorized validation repair, locating
candidate losses and testing isomer ranking before further GPU spending. This
cycle used retained serving features, two CPU threads and small linear
calibration fits. No encoder training, GPU inference, cloud job, submission or
fresh C13 reserved-label evaluation occurred. The champion remains 0.480.

The evaluator and admission safeguards are repaired. **A trustworthy proxy for
the leaderboard is still not established.** The CPU experiment demonstrates
why: a statistically supported development gain failed the separate source
checks and was rejected. No production model or scoring notebook was changed.

**What changed in validation.**

Candidate membership and label-free features were frozen before targets were
joined. Every query stays in MRR@25, including zero for an absent truth.
Duplicate official identities and missing queries raise errors. The reused
240-query CASMI2022 panel has 169 chemistry components, largest size8. Components
join equal nonempty Murcko scaffolds, equal truth formulas, Morgan Tanimoto>=0.70
and identical rounded acquisition signatures, with transitive closure. Five
outer folds each contain48 queries. The known structures are used for split
isolation, not as candidate-ranking features or truth injection.

Each outer fold's calibration weights and mixing coefficient are fitted using
its training queries only. Mixing is selected in three inner chemistry-grouped
folds, with smaller coefficients preferred on ties. The primary variant,
regularization, coefficient grid, three comparisons and endpoints were fixed
before new scores. Models train only on existing same-formula contrasts; a
missing true candidate is never added for training or evaluation.

A separate historical80-query external-source panel is a fixed stress check.
Twelve CASMI queries sharing chemistry components with that panel are excluded
from its model fits, leaving228 training queries. Its previously established
20 raw-signature-flagged queries and60 unflagged queries are reported separately.
Neither stress outcomes nor outer outcomes select coefficients. These panels
were already consumed before C36; nested fitting does not turn them into fresh
confirmation or exclude every public model's pretraining exposure.

The admission checker now explicitly rejects incomplete/truth-forced metrics,
held-out fitting/selection leakage, unsupported primary gains, adverse polarity
or clean-source results, wrong-sign P14 calibration and missing independent
confirmation/provenance. These checks prevent another positive local number
from being treated as a qualified leaderboard improvement.

**Where candidates disappear.**

The audit joins retained routes by the official tautomer-canonical connectivity
identity, not raw supplied keys. Local per-query witnesses retain route presence,
input/output availability, final rank and mass diagnostics.

| Used panel | Truth in final25 | Truth in retained route union | Retrieved but dropped | Absent from all retained routes |
|---|---:|---:|---:|---:|
| Positive160 |100|119|19|41|
| Negative80 |50|57|7|23|
| All240 |150|176|26|64|

All26 missing-final truths occur in retained ICEBERG input lists. Of the19
positive losses,16 have actual ICEBERG scores;12 occur in scored GLACIER lists
(these sets overlap). Six are already in Engine2 lists and eight in PubChem
lists. Thus at least part of the loss is candidate selection/fusion after
retrieval, rather than a need to generate new molecules.

The other three positive losses with no direct ICEBERG truth score are all
ammonium-adduct queries. They also occur in the retained N1 proxy input route.
The seven retrieved negative losses have no direct ICEBERG/GLACIER truth scores;
the production negative features have no signal from those channels. Input-list
presence must not be mistaken for actual supported inference. These witnesses
do not establish that every missing output was a runtime failure.

The64 truths absent from retained routes are not proven absent from the full
original databases. Earlier mass windows, database membership and unretained
candidate budgets cannot be distinguished from these saved files alone.

The mass audit found a diagnostic defect: cached `nm` is missing for25 formate
queries. The independent validation reader derives neutral mass from precursor,
charge and a recognized adduct, refusing unknown or polarity-inconsistent adducts.
This recovers mass diagnostics for all240, with maximum truth-mass error
7.95047ppm and zero above30ppm. It does not change ranking inputs, original
pipeline files or predictions. Missing diagnostic metadata is not evidence that
the production candidate generator used a wrong precursor mass.

**Ranking tests and result.**

All variants retain candidate membership and each formula's occupied positions.
Candidates may exchange positions only with candidates sharing their formula;
rank1 can therefore change within that formula. Parent outputs with library
confidence>=0.90 are protected. The CPU pairwise ridge fits use cached neural,
route-rank and physical features; the ablation removes physical features and
selects its coefficient independently within training folds.

| Fixed comparison | Used240 MRR | Delta vs0.283748 parent | Grouped95% interval |
|---|---:|---:|---|
| Physical-only formula reordering |0.285467|+0.001719|[+0.000087,+0.004169]|
| Primary CPU pairwise ranker |0.312032|+0.028284|[+0.005327,+0.051536]|
| CPU ranker without physical features |0.298615|+0.014867|[-0.007276,+0.036825]|

The primary improves48 query scores and harms40. It changes55 first choices,
gaining8 correct first choices and losing7. All missing-truth zeros are retained.
Against the physical-feature ablation, its incremental used240 gain is0.013418,
grouped95% interval[0.002356,0.025676]. This compares complete calibration
procedures, including their independently selected coefficients; it does not
isolate a causal effect of one physical score.

The held-out-source behavior blocks promotion:

| Primary CPU ranker stress check | Delta vs parent | Grouped95% interval |
|---|---:|---|
| Full historical80 |+0.009561|[-0.039357,+0.054392]|
| Predeclared raw-alias-free60 |-0.023909|[-0.085608,+0.020258]|
| Source negative31 |-0.037154|[-0.088955,+0.003079]|

The full80 positive point estimate hides the clean60 regression. By exact
weighted decomposition, the preflagged20 have a +0.109970 point gain; their
exposure warning was fixed before C36 outcomes. This is a population-sensitivity
diagnostic, not proof of leakage or an acceptable subgroup to target. The
negative stress interval includes zero, so the result is not a statistically
established harm; it also supplies no reliable evidence of nonnegative transfer.

The physical-only change makes no improvement on clean60 and changes no first
choices. Its primary gain is below the predeclared0.003 materiality threshold.
The nonphysical ranker also has an uncertain primary gain and adverse clean-
source/negative-source point estimates. None of the three qualifies for GPU
work or submission. No alternate variant was selected after viewing stress data.

P14 remains a wrong-sign control on used240: local delta+0.001659 versus actual
public0.458 against0.480. P15 is locally near null/negative; P18/P19 retain their
small uncertain local gains despite public0.478. Correct grouping and arithmetic
cannot repair the population mismatch by themselves. The admission report
retains that failure rather than filtering queries until old leaderboard signs
look right. Fresh C13 metrics remain available for a future frozen, defensible
candidate, not for rescuing this failed source check.

**Independent verification and disposition.**

Ten tests cover missing-query/absent-truth handling, duplicate official keys,
formula-slot preservation, group isolation, held-out-target invariance, weighted
cluster bootstrap, wrong-sign control blocking, source harm hidden by aggregate
gains and charge/adduct mass handling. The independent audit recomputes12 frozen
fits with augmented least squares rather than normal equations: maximum weight
difference1.78e-15. It separately reconstructs640 learned prediction lists and
their metrics exactly. All inner/outer/source training exclusions pass. Original
source/model qualification artifacts remain unchanged.

Code: `validation_repair.py`, `validation_admission.py`, and the C36 preparation,
route audit, ranking, review and independent-weight scripts under `scripts/`.
Settings are in `configs/c36-cpu-validation-20261111.json`; compact receipts are
in `docs/evidence/c36-20261111/`. Raw labels, coefficients, per-query predictions
and candidate-loss witnesses remain ignored/local in
`outputs/local/c36-validation-repair-20261111/`.

The CPU cycle is finished. Training, GPU inference, cloud runs, submissions,
hourly automation and the silver campaign remain paused. The result justifies
further work on cross-source calibration and the observed26 candidate-selection
losses, without launching another GPU training dose from this evidence.
