# Submissions

Public LB = MRR@25 on ~125 public molecules (one rank-2→1 flip ≈ 0.004). Scoring reruns take hours.

C09 P16v2 accepted10November01:22UTC, **ref58181280**, marker
`C09-P16-EXTREF-20261109`. Actual public score is pending; authenticated API
reportsPENDING and native UI showsNotebook Running. This is the ONLY pending
submission globally. Do not resubmit or send another trial until its actual
score resolves. P01/P13 remain champion0.480. P15/ref58163163 is COMPLETE0.479,
no reported error; never resubmit P15. Durable intent was resolved to58181280.
The10November01:37UTC approval for one further trial is reserved/unconsumed.
C10's running private inference is validation only, not a leaderboard submission.

| Date (UTC) | Slot | Kernel | Change | Ref | Public LB |
|---|---|---|---|---|---|
| 2026-11-10 01:22 | P16 | casmi26-p16-external-reference-c09 v2 | Unchanged pinned P01 plus fixed measured-reference promotions; no newly trained model | 58181280 | **0.480** (COMPLETE) |

P16 (`casmi26-p16-external-reference-c09` v2) completed its private real12/400
Kaggle smoke and independent execution/identity/fallback/local-cloud audit.
V1 failed after P01 on a checkpoint-audit bundle alias; no leaderboard trial was
spent. V2 preserves the production rule/models/references, explicitly audits
both checkpoints and exercises7 promotions/2 insertions on30 isolated real
development fixtures. The preview12 are protected pipeline-library matches.
Frozen80 confirmation is complete:MRR.713859→.837445,13 improved/3 harmed,
truths71→77;raw-clean60 MassBank delta+.063889,95%CI[.005556,.133333]. All20
GNPS queries are raw-signature flagged,so there is no clean GNPS evidence.
Reference-input alias exclusion gives identical results;0 detected matches.
The full80 panel does not detect P14/P15 public regressions. Three harms include
one NH4/H adduct mismatch; the predeclared rule was not tuned on confirmation.
Review/qualification and independent execution/source/identity/fallback/parity
receipts are retained in `docs/evidence/c09-2026-11-09/`. One actual-score trial
is authorized by the current C09 campaign; hidden benefit remains unproven.

P15 was accepted at12:01UTC after its real Kaggle smoke and independent audits,
spending C08's single-trial authorization. C09 is separately authorized.
**P13=0.480; P14 v2=0.458**, both COMPLETE without reported errors. Best remains
0.480 (P01/P13). P14's conditional validation gain failed to transfer. P15's
reserved external-panel change is too small/noisy to establish improvement.
Receipts: `docs/evidence/compatible-2026-11-08/`.

| Date (UTC) | Slot | Kernel | Change vs parent | Ref | Public LB |
|---|---|---|---|---|---|
| 2026-10-08 | v6 | enveda-casmi-2026-cpu v6 | Own pipeline, popularity prior 0.30 (previous best) | 56967624 | 0.338 |
| 2026-10-28 | s1 | casmi26-s1-p477 v1 | Fork of public 0.477 (busyaprime m1) | 57747549 | 0.477 |
| 2026-10-28 | s2 | casmi26-s2-parallax v1 | Spectral Parallax latest: 0.477 + disagreement rescue | 57747570 | 0.477 |
| 2026-10-28 | s3 | casmi26-s3-relaxmerge v1 | s1 + keep leader moves when the check also changed rank 25 (fast commit) | 57747571 | 0.477 |
| 2026-10-28 | s4 | casmi26-s4-relaxmerge-tail3 v1 | s3 + 3-slot neural tail rescue (fast commit) | 57747598 | 0.477 |
| 2026-10-28 | s5 | casmi26-s5-relaxmerge-lib092 v1 | s3 + fragment-check library gate 0.92 (fast commit) | 57747601 | **0.473** |
| 2026-10-29 | t1 | casmi26-t1-loose v1 | s3 + looser rank-1 rule (lib 0.92, ≥2 spectra, ranks 2–8, neural margin 0.15) | 57751749 | **0.465** |
| 2026-10-29 | t2 | casmi26-t2-ckptens v1 | s3 + neural scorer averages final and last-validation checkpoints | 57751756 | 0.476 |
| 2026-10-29 | t3 | casmi26-t3-fpvote v1 | s3 + fold-safe FPNet vote 0.5 in ranks 2–25 | 57751766 | 0.465 (fallback) |
| 2026-10-29 | t4 | casmi26-t4-fpvote10 v1 | s3 + fold-safe FPNet vote 1.0 in ranks 2–25 | 57751772 | 0.465 (fallback) |
| 2026-10-29 | t3b | casmi26-t3b-fpvote v1 | t3 with the molecule-id fix: s3 + FPNet vote 0.5 in ranks 2–25 (fast commit; code smoke-tested by u3) | 57756861 | **0.470** |
| 2026-11-02 | w3 | casmi26-w3-seedens9 v1 | s3 + neural scorer averaged over 9 × 16-epoch models (kit final + 8 refits) | 57915477 | **0.460** |
| 2026-11-02 | w6 | casmi26-w6-seedens-e32 v1 | s3 + neural = kit final + 8 refits at 32 epochs (fast commit) | 57915482 | **0.458** |
| 2026-11-02 | w7 | casmi26-w7-seedens-e64 v1 | s3 + neural = 8 refits at 64 epochs, no kit final (smoke ✓) | 57915486 | **0.453** |
| 2026-11-02 | w8 | casmi26-w8-seedens-e128 v1 | s3 + neural = 4 refits at 128 epochs, no kit final (smoke ✓) | 57915488 | **0.459** |
| 2026-11-02 | v1 | casmi26-v1-v33residual v1 | public v33 real-list residual: logistic re-sort of ranks 2–25 where library < 0.87 | 57915490 | 0.477 |

t3/t4 v1 scored 0.465, the frozen baseline: their consensus patch reused the name `mid` (the molecule id) for the
blended scores, so the V2-control rebuild raised. Kaggle then scored the baseline `submission.csv` that the frozen
pipeline had already written — **a crash after the pipeline stage scores 0.465, it does not error the submission**.
Found by the u1 smoke commit; fixed in `t3b-fpvote` / `t4b-fpvote10` / `u*` (variable `fp_mid`).
t1 = 0.465 is most likely genuine: the looser rule (contenders from ranks 2–8, ≥2 spectra, lib 0.92, neural margin
0.15) has no crash path in `decision()`, and s5 already lost one molecule with the library gate alone.

2026-10-29 slot 5: the t5-loose-fpvote fallback was dropped after s5 lost (its looser rule goes the same way).
Replaced by `t3b-fpvote` (t3 with the fix; same FPNet code that the u3 smoke commit ran cleanly on 12 real molecules),
so the FPNet vote still gets a clean measurement today. Submitted 03:09 UTC. Scored 0.470 (−0.007 vs s3 ≈ two
molecules): the FPNet vote in ranks 2–25 hurts, so its 1.0 twin t4b was dropped from the 10-30 queue.

2026-11-02: the 10-30 slate (epoch dose-response w3/w6/w7/w8 + v1) was built and committed on 10-29 but never
submitted (the user stopped work on 10-29 20:40). Submitted unchanged 16:51 UTC on 11-02; every kernel was COMPLETE
with a `submission.csv` (w6's holds fast-commit placeholders; its scoring rerun runs every stage).
The user did not want all five sent at once (rule since 11-02: at most 2 at a time, spread across the day). They
could not be withdrawn: the Kaggle API (kaggle 2.2.4 / kagglesdk 0.1.37) has no cancel or delete for submissions,
and a submission counts toward the daily 5 when it is made.

10-30 slate results (scored 21:22–21:42 UTC): every ensemble of our scorer refits lost ~0.02 against the kit model
alone (s3 0.477): w3 0.460, w6 0.458, w7 0.453, w8 0.459 — flat in epochs although holdout MRR rose 0.38 → 0.60 on
natural products. v1 (v33 residual) 0.477 = no effect. Analysis and the w20/w21 single-model diagnostics: STRATEGY.md.

Queue and push/submit state: `kaggle/forks/queue.json` (10-28), `kaggle/forks/queue-2026-10-29.json`;
orchestrator log: `outputs/autopilot.log`.

Earlier 2026-11-05 session: **no new submission**. Two full 32-epoch local RTX 2070 candidate-loss
training/evaluation rounds completed. The selected tail-only fusion changed reserved
confirmation MRR by -0.000056 / +0.000259; both 95% paired intervals include zero.
The natural-product coefficient sweep also failed confirmation. Best submitted public
score remains **0.477**. No Kaggle scoring run or extra GPU quota was used.
See [`docs/SCIENCE-2026-11-05.md`](docs/SCIENCE-2026-11-05.md).

## Resumed professional competition run — 2026-11-05

| Date (UTC) | Slot | Kernel | Change and qualifying evidence | Ref | Public LB |
|---|---|---|---|---|---|
| 2026-11-05 18:15 | P01 | casmi26-p01-public480 v1 | Fixed public 0.480 recipe: public + rebuilt v80 neural z-score ensemble, documentation prior, exact metric-identity dedup. Exact raw-panel score reproduction (max error 1.91e-6); panel proxy 0.912842 → 0.918796; alternate checkpoint also positive; NP tail checks positive; COMPLETE real smoke, zero input errors, both models and merge verified. | 58032057 | **Pending** |
| 2026-11-05 18:40 | P02 | casmi26-p02-public479 v1 | Original Saner neural scorer with the same fixed documentation/dedup recipe. COMPLETE real smoke: 400 valid rows, 12 real molecules, zero input errors; all 300 real neural scores exactly equal P01's original channel (maximum error 0). | 58032626 | **Pending** |
| 2026-11-05 22:20 | P11 | casmi26-p11-np-anchor v1 | Original50/NP-curriculum50; full-data general20 parent128 + fixed32 NP-mixture epochs. Both independent excluded parents improve native retrieval and final-panel point estimates. COMPLETE real GPU smoke: 400 valid rows, 12 real molecules, zero metric duplicates; independent blend error 4.44e-16; exact checkpoint/parent/source guards. | 58036642 | **Pending** |

The latest user instruction explicitly requested one best-candidate submission
after the current run, followed by an overnight pause. This authorizes one
end-of-day exception to the two-pending limit; it does not authorize another
submission. P11 was submitted from kernel version1, output `submission.csv`,
at 22:20:35 UTC (unique description marker `EOD11-20261105`). The 22:26 UTC API
check confirms **3 of 5 daily submissions used; 2 remain unused**. P01, P02 and
P11 are all pending. The best completed public score is still **0.477**.

These are expected-improvement tests; local diagnostics do not establish a
hidden-score improvement. The goal is **paused for today at the user's request**.
All training, experiment controllers, score watchers and the hourly run updater
are stopped. Submitted Kaggle evaluations continue to return scores. No automatic
restart or overnight submissions are scheduled. Resume only when the user asks.
Protocol: [`docs/COMPETING-2026-11-05.md`](docs/COMPETING-2026-11-05.md).

## Scores verified 2026-11-06

| Submission | Kernel | Public MRR@25 | Interpretation |
|---|---|---|---|
| 58032057 | P01 public480 v1 | **0.480** | New best submitted baseline |
| 58032626 | P02 original-scorer control v1 | **0.479** | Documentation/dedup control |
| 58036642 | P11 NP curriculum v1 | **0.468** | Local gains did not transfer; not promoted |

Verified through the Kaggle submissions API. No submission made on 6 November.

## User-requested L06 trial — submitted 2026-11-06

| Date (UTC) | Slot | Kernel | Change and execution evidence | Ref | Public LB |
|---|---|---|---|---|---|
| 2026-11-06 23:50 | P12 | casmi26-p12-long-control v1 | Original scorer50 + mean of five128-epoch L06 control fits50; fixed P01 documentation/physical/metric-dedup recipe. COMPLETE real12-molecule Kaggle GPU smoke,400 valid rows, zero input errors/metric duplicates, all5 pinned models used; independent raw-score mean error0.0 and public-blend error4.44e-16. Explicit user-requested leaderboard trial despite unproven hidden-score benefit. | 58071964 | **Error; no score** |

Accepted at23:50:11UTC with unique marker`L06-P12-20261106`. This is exactly one
submission; no other L06 candidate is queued. It preserves the verified0.480 P01
baseline. The completion/OOF diagnostics did not qualify an expected improvement;
the user's explicit **"Submit it use a submission"** overrides that promotion
gate for this one trial. Scoring is pending; the smoke is not a hidden-test result.
Protocol and receipts: [`docs/LONG-SUBMISSION-2026-11-06.md`](docs/LONG-SUBMISSION-2026-11-06.md).

Verified 7 November: P12 is COMPLETE **with an error and an empty score**.
Kaggle reports an unhandled error in the hidden rerun; no detailed hidden
traceback is available. Its exact root cause is not established. This is not
a leaderboard measurement. P01 remains the verified0.480 baseline.

## Authorized public trial — 7 November

| Date (UTC) | Slot | Kernel | Change and execution evidence | Ref | Public LB |
|---|---|---|---|---|---|
| 2026-11-07 22:04 | P13 | casmi26-p13-outside-blend v1 | Exact P01 primary scores and pinned v80; public50/v80-50 outsider scoring consistency. Real12 GPU smoke,400 valid rows, zero input failures/metric duplicates. Independent primary error4.44e-16, outsider error2.08e-7; visible12 rankings unchanged from P01. | 58105681 | **0.480** |

First of two newly approved submissions accepted. Marker`F07-P13-20261107`.
Scored0.480, tying P01; no demonstrated gain from this outsider blend.

## Second approved trained trial — submitted 8 November UTC

| Date (UTC) | Slot | Kernel | Change and execution evidence | Ref | Public LB |
|---|---|---|---|---|---|
| 2026-11-08 01:52 | P14 | casmi26-p14-frozen-channel v2 | P01 plus fixed50% F07R1 seed202611071 final-list channel; unchanged P01 outsider scorer and exact unsupported fallback. Development-selected fusion; reserved conditional robustness MRR .637481→.742594. COMPLETE real12 GPU smoke,400valid, zero input errors/metric duplicates; exact independent fusion; stable peak ties and local/Kaggle parity0.000489 verified. | 58111204 | **0.458** |

Both new user-approved scoring slots are now accepted: P13 ref58105681 and
P14 v2 ref58111204. P14 v1 was never submitted: its platform-dependent cutoff
selection was diagnosed and fixed before spending the remaining slot. Marker
`F07-P14-20261107`; accepted at2026-11-08T01:52:30UTC (7 November20:52 Eastern).
No further scoring submission is queued. Best verified public score remains
P01/P13 **0.480**. P14 returned0.458, below baseline despite its local gain.
Conditional retrieval MRR is not a leaderboard score. Actual pipeline validation
must explain this negative control before another learned channel is promoted.

## C08 authorized trial — submitted 9 November UTC

| Date (UTC) | Slot | Kernel | Change and execution evidence | Ref | Public LB |
|---|---|---|---|---|---|
| 2026-11-09 12:01 | P15 | casmi26-p15-compatible-c08 v1 | Complete P01 plus0.20 tail-RRF from two independently trained320-wide legacy16-hard fits,40000 updates each. Rank1 and candidate set protected. Development-only selection; reserved external MRR .3617255→.3619792, interval includes zero. Real12 Kaggle GPU smoke,400 valid rows; zero input errors/metric duplicates. Independent two-fit error0, unchanged P01 primary error4.44e-16, local/Kaggle maximum raw-score difference2.86e-6. | 58163163 | **0.479** (COMPLETE; verified21:06UTC) |

Accepted at2026-11-09T12:01:22UTC /07:01 Eastern, unique marker
`C08-P15-20261108`. The API independently confirms exactly one accepted marker,
PENDING with an empty score. All8 C08 fits completed320000 successful updates
in11h22m with zero restarts. Only the development-selected same-arm pair is used
by P15. The explicit single-trial authorization is exhausted; no extra candidate
or training is queued. P01/P13 at0.480 remain the champion while scoring runs.
No statistical promotion is claimed. Full evidence is in
`docs/evidence/compatible-2026-11-08/`.
