# Submissions

Public LB = MRR@25 on ~125 public molecules (one rank-2→1 flip ≈ 0.004). Scoring reruns take hours.

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

This is an expected-improvement test, not a claim that the displayed public
notebook score has reproduced. The objective remains active until a completed
submission actually exceeds the baseline **0.477**. At most two may be pending.
Three daily slots remain. Keep at least two reserved until the initial scores
are available; do not spend the remaining slots before reviewing that batch.
Protocol: [`docs/COMPETING-2026-11-05.md`](docs/COMPETING-2026-11-05.md).
