# Experiment log

Every run executes on Kaggle (private notebook `clarkkitchen/enveda-casmi-2026-<device>`, internet off).
Each run writes `run_report.json` next to `submission.csv` with the configuration, package versions,
timings, the validation grid, and the selected weights. Download it with `Kaggle.ps1 output --device cpu`.

Validation runs in the cloud on `enveda-np-examples` (~250 natural products on the test instrument):

- **C1** (Class-1 simulation): the query spectra are removed from the library; other libraries keep the compound.
- **C2** (Class-2 simulation): every spectrum of the query structures is removed; the structure stays reachable
  through the candidate pool.
- **Objective** = 0.25 · C1 + 0.75 · C2 for v1–v3, and 0.45 · C1 + 0.55 · C2 from v4 onward, used to choose blend weights. It is a filter for broken changes, not a
  leaderboard predictor; public notebooks report that local validation and the leaderboard disagree on small deltas.

Public LB = MRR@25 on the hidden test (~400 molecules). Five submissions per day.

| Version | Kernel | Change | C1 | C2 | Objective | Public LB |
|---|---|---|---|---|---|---|
| v1 | cpu v1 | Baseline: entropy library search + hybrid analog propagation over train ∪ COCONUT (712k structures); selected A=0, P=3 | 0.712 | 0.609 | 0.635 | **0.268** |
| v2 | cpu v2 | Gated library boost A·lib·[lib≥τ] + public fingerprint models (f·z, z-scored) + analog prefilter (top 4,000 by a cheap bound); selected A=4, τ=0.8, P=2, B=0.05 | 0.878 | 0.606 | 0.674 | **0.299** |
| v3 | cpu v3 | LightGBM lambdarank (3 seeds) on 19 library/analog features, trained in-cloud on 1,305 simulated queries (884 Class-2-style, 421 Class-1-style) from structures outside validation; + B=0.4 f·z | 0.633 | 0.810 | 0.766 (w1=0.25) | 0.266 |
| v4 | cpu v4 | v3 ranker + MetFrag-lite features (frag, frag_gap, frag_rank) + gated library boost and f·z on top of the ranker; objective w1=0.45; selected A=8, τ=0.8, B=0.8 | 0.868 | 0.726 | 0.790 (w1=0.45) | 0.320 |
| v5 | cpu v5, submission 56963849 | E2: PubChem popularity prior, weight 0.15 | 0.8968 | 0.8015 | 0.8444 (with prior; gate base 0.7895) | 0.331 |
| v6 | cpu v6, submission56967624 | E2b: double popularity weight0.15→0.30 only | 0.9079 | 0.8173 | 0.8581 (with prior; gate base0.7895) | 0.338 **(best)** |

## Notes

- **v1 (2026-10-07).** Python 3.13.15, numpy 2.1.3, numba 0.61.2, RDKit 2026.03.3 (offline wheel), 25 min on
  Kaggle CPU. 400 test molecules, median 107 candidates (max 604), no fallbacks. An ungated library term trades
  Class 2 for Class 1: A=0.5 gives C1 0.889 but C2 0.392, because same-mass impostors in the library get promoted.
- **v2.** Gating the library term at τ=0.8 keeps most of the Class-1 gain without the Class-2 loss. The fingerprint
  models add about +0.025 C2 at the best weight; larger weights hurt (B=0.4 → C2 0.542). 21 min on CPU.
- **Calibration from v1 and v2:** solving LB ≈ α·C1 + β·C2 gives α ≈ 0.19, β ≈ 0.22. That's two points, so it's
  only a rough guide, but it suggests about 19% of hidden molecules behave like Class 1, and that a Class-2 gain is
  worth slightly more than the same Class-1 gain.
- **v3.** The ranker lifts C2 from 0.606 to 0.810 but drops C1 from 0.878 to 0.633. Trained mostly on Class-2
  queries, it learned that candidates visible as their own analog (`best_tan` = 1, i.e. library structures) are
  usually wrong. That's right for molecules without public spectra but wrong for Class 1. Top features by gain:
  `ap3_rank`, `best_tan`, `ap3_gap`, `lib_rank`. Under the v1/v2 calibration, v3 and v2 predict about the same LB
  (~0.30); it was submitted to measure whether the C2 gain transfers. Run time 43 min.
- **v3 result: 0.266**, below the 0.2965 the v1/v2 calibration predicts using the unrounded report metrics. Compared with v1 (ΔC1 −0.079 ≈ −0.015 LB),
  the +0.201 C2 gain was worth only about +0.013 LB. That's roughly 0.065 LB per unit of ranker-driven C2 MRR, versus
  the 0.22 implied by v1's level. **The Class-2 simulation overstates ranker gains about 3×.** Treat C2 deltas
  from the ranker as weak evidence until the simulation is fixed (see `PHASE_1.md`).
- **v4.** Gated library boost (A=8, τ=0.8) on the ranker restores C1 (0.868) while keeping much of the C2 gain
  (0.726). Fragmentation features carry real weight (`frag_gap`, `frag_rank` are the 3rd–5th largest by gain).
  Using the v3 transfer rate, it should land around v2 + 0.008 ≈ 0.305. Run time 47 min (fragmentation ~9 min).
  Kernel version 4 completed and was submitted on 2026-10-08 (submission 56956009). The result is **0.320**, an improvement of **0.021** over v2; keep the ranker pipeline as the Phase 1 base.

- **E1 complete (2026-10-08).** Submitted completed CPU kernel version 4 after verifying 400 molecules, zero fallbacks, zero missing logits, total 2,849 s and objective 0.7895. Submission 56956009 completed with public LB **0.320** at the 14:27 UTC observation. No submissions were used today before E1; keep the Phase 1 cap at four per UTC day. User requested new training on the SSH server at 24.196.244.156:42272; authentication is unresolved. No training has run on that server.

- **E3 preparation (2026-10-08, not an experiment result).** Added a disabled `use_bio_db` switch and ChEBI/LIPID MAPS loading before training-structure fingerprints. It rejects incompatible array formats and skips structures already present, including repetitions within the new source. Two synthetic tests cover source order and deduplication; all ten local tests and both template checks pass. The dataset is not attached and the feature remains disabled pending E2.

- **E3 input preparation (2026-10-08).** Downloaded only the named ChEBI/LIPID MAPS files directly to the authorized server while E2 verifies. The guarded cloud check confirmed 62,744 rows, uint8 fingerprints shaped 62,744×867 with 6,930 selected bits, compatible metadata, and zero non-finite masses. All 62,744 keys are new relative to COCONUT; training-structure overlap will be handled when E3 builds its pool. Active config remains `use_bio_db=False`; proposed E3 configs are isolated under ignored `.local/` until E2's leaderboard decision. This prepared inputs, without launching a scoring experiment.

- **E3 semantic check.** Recomputed fingerprints for64evenly spaced Bio structures on the authorized server using the baseline RDKit generator and selected bits. All64matched the packed input bytes exactly; no sampled SMILES failed, and the maximum sampled mass discrepancy was3.49e-10ppm. This validates the sampled encoding beyond array dimensions; the full source remains subject to E3's normal gate and leaderboard test.

## Calibration after E1

Least-squares bookkeeping for `LB = a*C1 + b*C2`, without an intercept, using the report metrics rather than rounded table values:

| Version | C1 | C2 | Public LB | Ranking |
|---|---|---|---|---|
| v1 | 0.7122 | 0.6085 | 0.268 | linear |
| v2 | 0.8777 | 0.6055 | 0.299 | linear |
| v3 | 0.6332 | 0.8099 | 0.266 | ranker |
| v4 | 0.8677 | 0.7255 | 0.320 | ranker |

- All four versions: **a ≈ 0.2557, b ≈ 0.1318**. Fitted scores are 0.2623, 0.3042, 0.2687, and 0.3175; no absolute residual exceeds 0.015.
- Linear versions v1–v2: **a ≈ 0.1912, b ≈ 0.2166**.
- Ranker versions v3–v4: **a ≈ 0.2720, b ≈ 0.1158**.
- Each separate fit has only two points and is descriptive, not a reliable predictor of small leaderboard changes. The prior linear fit predicts v4 at 0.3231, versus its actual 0.320; the discrepancy is under 0.015. v3's prior-fit discrepancy remains approximately −0.0305, consistent with its overly optimistic Class-2 simulation.
- Keep `objective_class1_weight` at 0.45 for E2 so popularity is the only new scoring change.

## E2 complete; E2b next

`v5-pubchem-popularity` retains the v4 ranker and fragmentation with `pop_weight = 0.15` and the named popularity dataset. Its Vast run completed in 215 seconds and passed the gate. Private Kaggle CPU version 5 completed in 2,772 seconds, with 400 predictions, no fallbacks/missing logits and base C1/C2/objective **0.8677/0.7255/0.7895**, exactly matching v4. With the prior, Kaggle C1/C2 are **0.8968/0.8015**. Submission **56963849** completed at the 17:58 UTC observation with public LB **0.331**, a **0.011** gain: keep the prior. The new champion is preserved under `.local/phase1-best/`; the prior v4 champion is archived under `.local/phase1-history/v4-best/`. Consecutive non-improvements reset to zero. E2b will test **0.30**, changing only the prior weight; two of the four allowed daily slots remain. E3 remains disabled with its inputs ready.

## Calibration after E2

Cloud least squares over all five submitted versions gives **a=0.2552371, b=0.1310609**. Linear v1–v2 remains **a=0.1912374, b=0.2165993**; learned v3–v5 gives **a=0.2643802, b=0.1210309**. For v5, use final with-prior C1/C2 0.8968/0.8015 rather than the unchanged base metrics. The all-version fitted v5 LB is 0.33394 versus actual 0.331; no fitted residual exceeds 0.015. The separate linear group still has only two points. Keep objective weight0.45 unchanged. Numerical evidence is in ignored `.local/calibration-after-e2.json`.

E2b's guarded Vast run passed in **228 seconds**, with 400 molecules and zero fallbacks/missing logits. Base objective0.7849 matches the prior server runs; with-prior server C1/C2 are0.9079/0.8202. All CPU pilot evidence arrays and cached fragment scores remained exact. Four query workers×six threads stayed fastest; peak projectCPU24.10logical cores, PSS5.12GiB, no addedthrottling/memoryerrors. Private Kaggle CPU version6 is now running; `.local/watch_e2b.py` performs the same gate/daily-cap/single-submit/score handoff. v5 remains the best until this experiment's LB result.

Kaggle CPU version6 completed in **2,752 seconds** and passed all gates, with base C1/C2/objective0.8677/0.7255/0.7895 exactly matching v5/v4. Final with-prior C1/C2 were0.9079/0.8173. The watcher submitted it as **56967624** at18:57UTC; scoring is pending. Three daily slots have been used (E1, E2, E2b), leaving one within the Phase1 cap.

## Server performance preparation (2026-10-08)

Verified rented allocation: Ryzen 9 7945HX (16 physical cores, 32 SMT threads), CPU quota 30.72 logical-core equivalents, memory limit 27.9 GiB, RTX 5080 with 16 GiB VRAM. A dedicated agent records project CPU/memory, GPU contention and stage throughput. Initial settings are 24 fingerprint/fragment workers and Numba/LightGBM threads, BLAS threads 1, Torch host threads 4, and float32 GPU inference batches of 32 spectra. These settings require measured validation; no speedup is claimed yet. Other server workloads are left untouched.

Historical v4 stage durations: pool 323 s, library 90 s, validation 505 s, training-query search 1,019 s, test 335 s, fragmentation 527 s, LightGBM fit 40 s, final ranking 10 s. Search and fragmentation are the primary performance targets. New timings separate GPU inference, fragment generation and fragment matching. The Kaggle guard remains unchanged; the additional server guard rejects Windows/WSL and unverified instances.

The monitored server v4 reproduction completed in **217 seconds**, versus historical Kaggle v4 **2,849 seconds** (13.13× throughput across this hardware/runtime change). It produced all 400 molecules with zero fallbacks/missing logits. Fingerprint pool: 712,199 structures, zero failed training SMILES. Server C1/C2/objective were 0.8626/0.7214/0.7849, a small drift from Kaggle champion 0.8677/0.7255/0.7895; this is not an exact ranking reproduction. Same selected A=8, gate=0.8 and B=0.8. The CPU search pilot verified every evidence array exactly; float32 GPU batch comparison had max absolute logit difference 0.000061. Public LB still decides experiments.

E2 CPU pilots compared 4×6, 6×4, 8×3 and 4×4 outer/inner threads on the same 32 queries. All evidence arrays matched exactly; **4×6 was fastest** (2.285 s vs serial 2.609 s), while 4×4 took 2.948 s. The matching cache also passed an exact-score gate (0.186 s vs 0.279 s in its 32-query pilot); full matching fell from 22.083 s to 14.909 s. Test GPU inference took 0.579 s for 400 molecules. Peak v4 project CPU was 24.04 logical-core equivalents; cgroup RAM peaked at 46.11% with no OOM events. Reports/backups: ignored `outputs/server/v4-20261008T160433Z/` and `outputs/server/v5-20261008T161500Z/`. The resource-monitor agent and a 15-minute chat heartbeat continue the authorized workflow.

Operational fixes verified before accepting results: preserved ZIP-based `.pt`/`.whl` files during transfer; passed fingerprint-bit selection into spawned workers; added a valid-SMILES initialization gate and refused a severely incomplete pool. The rejected partial attempt was stopped and was not submitted. All 18 synthetic local tests and notebook metadata checks passed.

E2 orchestration is handled by ignored `.local/watch_e2.py`: retrieve the completed Kaggle report, apply `scripts/phase1_gate.py`, recheck the daily submission count, record an exclusive submit intent, submit version 5 once, and poll its score. Inspect `.local/e2-submit-intent.json` before any retry after an uncertain submission outcome. The gate has synthetic tests; all 21 local tests now pass. `cloud/server_calibration.py` performs subsequent calibration fits on the authorized server, and its E1 check reproduced all existing coefficients (`outputs/server/preparation/calibration-e1-cloud.json`).

## E2b result and user-requested pause (2026-10-08)

E2b scored **0.338**; the +0.007 gain meets the 0.003 keep threshold, so retain popularity 0.30 and v6 as best. Result observed at 2026-10-08T19:47:10.161254+00:00. Server training finished in 228 seconds; private Kaggle CPU version 6 finished in 2,752 seconds. Both passed the 400-molecule/zero-fallback/zero-missing-logit gates. Tested source/config/report/result are preserved in `.local/phase1-history/v6-tested/`. The accepted best configuration is restored or retained in `cloud/workload.py`; no reproduction run is launched because the user requested stopping. Calibration after E2 remains the latest fit; the six-version update is deferred until cloud work resumes. The project tunnel is disconnected, the heartbeat is paused, and E3-E5 remain pending.
