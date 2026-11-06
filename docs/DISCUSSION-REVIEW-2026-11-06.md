# CASMI discussion review — 6 November 2026

Reviewed all 148 topics in the eight pages of the competition discussion index,
sorted by recent comments, including all available comment text and collapsed
reply chains. Brief rate-limited/empty loads were revisited. Deleted comments
cannot be recovered. This is a dated research review, not endorsement of every
participant's claim. The machine-readable topic checklist is alongside this file.

## What changed the plan

1. **The actual new baseline is 0.480.** Our own Kaggle submissions API now shows
   P01 0.480, P02 0.479 and P11 0.468. The longer NP curriculum improved reused
   local diagnostics and lost on hidden public data. Today's experiment compares
   external data against a matched control and preserves the existing champion.
2. **Use common folds and genuine OOF scores.** Deotte's six solution writeups
   use one fold definition across models, cached shared inputs, independent model
   directories and regularized stacking. Here folds follow canonical molecular
   identities across every spectral source; they are not row-level stratified folds.
3. **Conflicting labels can cross an identity split.**
   [744446](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744446)
   reports exact spectra assigned to different constitutional isomers. The new
   split audit connects labels sharing identical cached model inputs and keeps
   each connected component in one fold. Quantized equality is conservative;
   missing acquisition provenance still prevents an acquisition-disjoint claim.
4. **Use the newly disclosed open spectral snapshots, with record checks.**
   [755965](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/755965)
   provides MassBank/GNPS mirrors, COCONUT, LOTUS and Enveda-180. MassBank/GNPS
   supply labeled experimental spectra; COCONUT/LOTUS supply structures and
   associations, not additional measured spectrum labels. The claimed 18M
   [candidate database](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/753724)
   is likewise a structure pool, not 18M training spectra. No unsupported
   spectrum-to-structure labels are manufactured from it.
5. **Label and adduct quality matter.**
   [742087](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742087)
   reports library adduct errors, especially formate labels; comments question
   how much low instrument accuracy explains. External intake validates neutral
   mass against the parsed structure, retains explicit adduct aliases, rejects
   unsupported species and excludes low-confidence/propagated GNPS annotations.
6. **Instrument match and chemistry match differ.**
   [742055](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742055)
   notes that timsTOF largely selects synthetic Enveda-180 chemistry. Its comments
   report stronger fingerprint diagnostics after timsTOF fine-tuning but a worse
   public score. Evaluate native NP and broader cohorts separately; do not assume
   another instrument-weighted curriculum will transfer.
7. **Ranking and recall need different measurements.**
   [752735](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/752735),
   [748524](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748524)
   and [743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254)
   distinguish absent truth, out-of-top25 truth and incorrect ordering. The new
   OOF candidate diagnostic forces the truth into its list; it cannot establish
   candidate recall. A production decision still needs an end-to-end replay with
   missing truths scored zero and all fitting paths excluded by identity.
8. **Do not tune repeatedly on the old NP panel and call it confirmation.**
   [748068](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748068)
   and comments in [742055](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742055)
   show strong curated/public-spectra bias. In [743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254),
   a participant's selected fragmentation gain shrinks from about .065 on tuning
   molecules to .023 on independent panels. The shared-fold sample expands the
   development population; reused NP examples remain explicitly labeled.
9. **De novo generation is a separate unresolved lever.**
   [753400](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/753400)
   reports weak recovery from hand-coded transformations when close analogs are
   removed. [755089](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/755089)
   finds that generated isomers' stability does not explain their poor ranking.
   [741659](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741659)
   explains the MRR cost of displacing a correct leader. Today's spectrum model
   does not claim to solve Class3 or add a validated de novo generator.

## Current host rulings

| Question | Current evidence and implication |
|---|---|
| Free public external data, including NC/SA | [750330](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750330), [745615](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745615), [748363](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748363): permitted subject to source rules; NC is not automatic prize/medal exclusion. |
| List of approved names exhaustive? | [748552](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748552): no; other compliant public sources can be used. |
| Newer GNPS, MassBank, MoNA | [745228](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745228), [750684](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750684): current releases and CC BY-SA records permitted. |
| Proprietary NIST and derivatives | [744470](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744470): generally prohibited. A permissive label on weights does not fix impermissible training data. |
| GNPS-NIST14-MATCHES exception | [745832](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745832), confirmed in [749217](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/749217): explicitly allowed for matching/training; excludes a blanket inference that new NIST derivatives are allowed. Today's new cache omits it. |
| ICEBERG PubChem atlas | [745134](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745134): host permits prediction subsets/membership with retrieval/processing code. Training/distillation on atlas outputs remains an unanswered follow-up; do not extend the ruling. |
| CFM-ID4 / public predicted spectra | [743774](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743774), [741912](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741912): explicitly allowed, with documented versions. Not used in this new training. |
| MassSpecGym, DreaMS, FRIGID, MIST | [748882](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748882), [746430](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/746430), [751373](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/751373): identified official releases allowed with original training/data references. This does not approve arbitrary unaudited fine-tunes. |
| Third-party fine-tunes/weights | [750612](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750612), [745072](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745072), [753662](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/753662): original training code and permitted training data needed. Published original code can suffice without a new pretraining run; equivalent self-retraining with deliverable code is another route. |
| Generated intermediates | [754292](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/754292), [753662](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/753662): must be reproducible and provenanced. Linking an opaque ranker-feature dataset is insufficient. |
| Bit-for-bit reproduction | [751373](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/751373): not required when source versions change; provide code, permitted inputs and version/download date. |
| PubChem structures/counts, LOTUS, NPAtlas | [749209](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/749209), [750461](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750461), [748882](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748882): permitted under source licenses. Following PubChem references to other content needs that source's terms. |
| Private processed datasets | [749209](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/749209): offline PubChem snapshots can remain private during competition, with deliverables available for review. |
| Enveda-180 Zenodo | [743569](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743569), [745148](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745148): public release allowed; missing structures likely preparation filtering, not evidence about hidden identities. |
| EnvedaDark / BMDMS-NP | [754732](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/754732): host doubts EnvedaDark was published; public releases would follow normal rules. BMDMS spectral-data licensing question in 748882 has no answer. Neither added. |
| SIRIUS / CSI:FingerID | [751373](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/751373): host says not allowed in the stated restricted-training-data case. A beginner roadmap mentioning it is not approval. |
| Automated cross-test processing | [752297](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/752297): automated label-free cross-spectrum processing permitted. Unanswered weight-adaptation follow-up should not be treated as approval. |
| CCS / ion mobility | [743368](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743368), [743569](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743569): deliberately absent from evaluation inputs; do not build an inference dependency on it. |
| Electron-mass correction | [748882](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748882): hidden test corrected, train/Enveda-180 release not corrected; positive m/z adjustment is −0.000548Da. No precision-sensitive new fragment scoring added here. |
| Collision energies | [753419](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/753419): nominal timsTOF values; lists mean merged acquisitions. External NCE/percent values are kept unknown. |
| Metric / canonicalization | [744556](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744556), [746878](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/746878): remove stereo before tautomer canonicalization and InChIKey14. Empty guesses discarded per [742274](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742274). |

## Claims to treat cautiously

The host does not publish the class shares or public/private stratification.
Participant estimates of 16/45/39, 28/20/52, or a Class3-heavy private set are
assumptions, not facts. The public board uses hidden public spectra, not the
downloadable placeholder; [741404](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404)
contains a direct host correction, and [745715](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745715)
contains staff confirmation that dummy and rerun IDs do not overlap. Do not build
target-specific static candidate artifacts from the placeholder's masses
([744172](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744172)).

Noise simulations in [753525](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/753525)
and [753511](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/753511)
illustrate how tiny public improvements can arise by selection, but their assumed
rank distributions are not measured hidden-test uncertainty. Compare paired
molecule outcomes and report uncertainty without converting diagnostic MRR into
a promised leaderboard gain.

Several older threads claim valid tail guesses can poison an unchanged leader.
Current host guidance in 748882 says the first-match scorer cannot lower that
row's MRR when valid distinct guesses are appended after an unchanged rank1.
Execution failures, fallback CSVs and reordering can explain apparent contradictions.
Pinned runtime checks and real-path smokes are essential; [749908](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/749908)
also reports environment drift changing reproduced scores.

Administrative/team-up/learning threads were read for completeness. No contacts
were messaged and no team membership, public post, model upload or submission
was changed during this review.
