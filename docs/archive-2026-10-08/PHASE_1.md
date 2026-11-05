# Phase 1

A runbook for an AI agent to continue improving the Enveda CASMI 2026 submission. Follow it in order. Each experiment
changes **one thing**, runs on Kaggle, is submitted, and is logged before the next one starts.

## 0. Ground rules (read first)

Later user authorization: perform Phase 1 training on the verified rented Vast instance (`24.196.244.156:42272`, container `54749798`) and use its GPU for pretrained fingerprint inference. This is a narrow exception to the Kaggle-only compute rule; local training stays prohibited. Use the instance-bound `cloud/server_run.py` guard and a dedicated resource-monitor agent. Kaggle CPU submission reruns, submission limits, scoring gates, and experiment order still apply. Resource tuning must preserve scoring semantics and record measured throughput.

**Latest user instruction (2026-10-08): stop after the current E2b run and stop for the day.** E2b is closed and its keep/revert decision is recorded. Phase 1 is paused for the day. Do not launch E3 or any further server jobs. The automatic monitor is paused and the project SSH tunnel is disconnected. Resume only after a new user request; Phase 1 remains incomplete.

- Read `AGENTS.md` and obey it. **Never train or run the pipeline locally.** All compute runs on Kaggle through
  `scripts/Kaggle.ps1`. Locally you may only edit files, run syntax checks and `tests/`, and call the Kaggle CLI.
- Never print, log or commit the Kaggle credential. Do not push to GitHub. `git` is not installed on this machine.
- The user has authorized cloud runs and competition submissions for this phase. Use **at most 4 submissions per UTC
  day** (the limit is 5), so one slot stays free for the user.
- Poll Kaggle no faster than once per 60 s; faster polling returns HTTP 429. Run long waits as background commands.
- Public LB is the ground truth. Local validation (C1/C2 below) is a filter for broken changes, not a predictor. The
  Class-2 simulation overstated the v3 ranker gain about 3× (see `EXPERIMENTS.md`).
- Stop and ask the user before: GPU runs, attaching a dataset not named in this file, fusing with another person's
  notebook, or anything that would take longer than 3 hours per run.

## 1. Where things stand (2026-10-08)

| Version | What | C1 | C2 | Public LB |
|---|---|---|---|---|
| v1 | Library search + hybrid analog propagation, train ∪ COCONUT pool | 0.712 | 0.609 | 0.268 |
| v2 | + gated library boost (A=4, τ=0.8) + public fingerprint models (f·z) | 0.878 | 0.606 | 0.299 |
| v3 | LightGBM lambdarank on library/analog features | 0.633 | 0.810 | 0.266 |
| v4 | v3 + MetFrag-lite features + gated library boost on the ranker | 0.868 | 0.726 | 0.320 (submission 56956009, 2026-10-08) |
| v5 | v4 + PubChem popularity prior, weight 0.15 | 0.8968 | 0.8015 | 0.331 (submission 56963849, 2026-10-08) |
| v6 | v5 + popularity weight doubled to 0.30 | 0.9079 | 0.8173 | 0.338 **(best)** (submission 56967624, 2026-10-08) |

- Code: `cloud/workload.py` (single file, embedded into the notebook by `Kaggle.ps1 prepare/run`). Config is the
  `CONFIG` dict at the top. The kernel is `clarkkitchen/enveda-casmi-2026-cpu`. Kernel version N = experiment N.
- E1, E2 and E2b are complete. E2b scored **0.338**; the +0.007 gain meets the 0.003 keep threshold, so retain popularity 0.30 and v6 as best. Three submissions were used today. The accepted best source/config/report/result are preserved in `.local/phase1-best/`. E3 inputs are verified but disabled; E3-E5 remain unrun. **Paused by user request; no automatic continuation.**
- Validation runs inside the notebook on `enveda-np-examples` (250 natural products, the test instrument):
  **C1** = query spectra removed from the library; **C2** = every spectrum of the structure removed.
  Objective = 0.45·C1 + 0.55·C2.
- Calibration after E2: all five versions give LB ≈ 0.25524·C1 + 0.13106·C2. Linear v1–v2 gives 0.19124·C1 + 0.21660·C2; learned v3–v5 gives 0.26438·C1 + 0.12103·C2. These groups have only two/three points; use LB to decide experiments. v5 uses its final with-prior metrics. See `EXPERIMENTS.md`.
- Typical costs: CPU run 25–50 min; a submission rerun on the hidden test takes 45–90 min more.

## 2. Procedure for every experiment

1. **Edit** `cloud/workload.py` (and `configs/project.json` if a dataset is attached). Set `CONFIG["version"]` to
   `vN-short-name`. Change one thing only.
2. **Check locally** (no compute):
   ```powershell
   .\.venv\Scripts\python.exe -c "import ast; ast.parse(open('cloud/workload.py',encoding='utf-8').read()); print('ok')"
   .\scripts\Kaggle.ps1 build-templates
   .\.venv\Scripts\python.exe -m unittest discover -s tests
   .\scripts\Kaggle.ps1 check
   ```
   All must pass. `test_workload_refuses_local_execution` must keep passing. Do not add heavy imports at module level.
3. **Run in the cloud**: `.\scripts\Kaggle.ps1 run --device cpu`. Note the printed kernel version number.
4. **Wait** in a background command:
   ```powershell
   for ($i=0; $i -lt 150; $i++) { Start-Sleep 60; $s = (.\.venv\Scripts\python.exe -m kaggle kernels status clarkkitchen/enveda-casmi-2026-cpu) | Out-String; if ($s -match 'COMPLETE|ERROR|CANCEL') { $s; break } }
   ```
5. **Read the result**: `.\scripts\Kaggle.ps1 output --device cpu`, then open the newest
   `outputs/cpu/<timestamp>/run_report.json` (keys: `validation`, `ranker`, `final_ranking`, `final_params`,
   `test`, `timings`). If the status is ERROR, read `outputs/cpu/<timestamp>/*.log`, fix the cause, and rerun.
   Do not submit a failed or partial run.
6. **Sanity gate** before submitting: `test.molecules` = 400, `test.fallback` = 0, `test.no_logits` = 0, total time
   under 3 h, and the objective not more than 0.02 below the current best. If the gate fails, log it and skip the
   submission.
7. **Submit**: `.\scripts\Kaggle.ps1 submit --device cpu --version N --message "vN: <change>; C1 x C2 y"`.
8. **Get the score**: poll `.\scripts\Kaggle.ps1 submissions` every few minutes until COMPLETE.
9. **Log** a row in the `EXPERIMENTS.md` table (version, kernel, change, C1, C2, objective, LB), plus a short note on
   what was learned. Update the calibration in section 4.
10. **Keep or revert**: keep the change only if LB improves by ≥ 0.003 over the current best. Otherwise revert that
    one change in `cloud/workload.py` before the next experiment. Seed noise in this competition is about ±0.006, so
    treat smaller deltas as ties.

## 3. Experiments, in order

### E1 — Submit v4 as is
- Kernel version 4 is already complete. Run steps 7–9 only: `submit --version 4`.
- Complete (2026-10-08): submission 56956009 scored **0.320**, exceeding the 0.302 keep threshold. The ranker remains the base. E1 is logged and calibration is updated; do not repeat this submission.
- Decision: if v4 ≥ 0.302, the ranker pipeline is the new base. If v4 < 0.299, set `CONFIG["use_ranker"] = False`
  (the linear v2 ranking, with fragmentation computed but unused) for the remaining experiments and note it.

### E2 — PubChem popularity prior
- Why: the top public notebooks add `μ · (log1p substances + log1p PubMed)` and report +0.004 to +0.010 LB. Already
  coded (`Pool.load_popularity`, `pop_weight`).
- Do: add `"dmitriigluzdov/casmi26-pubchem-popularity-prior"` to `dataset_sources` in `configs/project.json`. Set
  `CONFIG["pop_weight"]`: **0.15** if the ranker is the base (its scores are z-scored), **0.01** for the linear base
  (its scores span about 0–1).
- Validation can't choose this weight (the validation compounds are common, so popularity looks too good). The report
  shows `with_popularity` for information only. Decide on LB.
- If it helps, try one more weight (double or half) as E2b, budget permitting.

### E3 — Wider candidate pool: ChEBI + LIPID MAPS
- Why: Class-2 answers missing from the pool can never score, and pool recall is the likely reason simulation gains
  shrink on LB. The top public pipelines include ChEBI/LIPID MAPS (+62,744 structures).
- Do: add `"prvsiyan/chebi-lipidmaps-casmi26"` to `dataset_sources`. Add `CONFIG["use_bio_db"] = True`. In
  `build_pool`, when enabled, load `bio_fp.npy`, `bio_mass.npy` and `bio_meta.pkl` (`keys`, `smiles`; same format
  and same 6,930-bit fingerprints as COCONUT, found with `find_input("bio_fp.npy")`). Skip keys already present, then
  append before the training structures. Log the added count.
- Expect validation C2 to dip slightly (more decoys); judge on LB.

### E4 — Make the ranker less provenance-driven (only if the ranker is the base)
- Why: v3 showed the ranker leans on `best_tan` (a candidate equal to one of its own analogs is a library structure),
  and its C2 gains did not transfer.
- Do one variant only, chosen in this order:
  1. Rebalance training: `train_queries_class1` 450 → 900 (equal to class 2).
  2. If E4.1 does not help, drop `best_tan` and `close_analogs` from `rank_features` and `RANK_FEATURES`.
- Keep the gated library boost on top of the ranker unchanged.

### E5 — Variance reduction
- Do: `ranker_seeds` [0, 1, 2] → [0, 1, 2, 3, 4]; `train_queries_class2` and `train_queries_class1` × 1.5.
  Check that the run stays under 2 h.
- Expect a small gain and a more stable LB. Keep it if it is not worse.

## 4. Calibration bookkeeping

After each LB result, refit `LB ≈ a·C1 + b·C2` by least squares over all submitted versions. Fit linear-blend
versions and ranker versions separately when there are enough points. Record the coefficients in `EXPERIMENTS.md`.
If a change's predicted LB (from the fit) and its actual LB differ by more than 0.015, write a one-line hypothesis
for why. Do not change `objective_class1_weight` without at least four calibration points.

## 5. When Phase 1 is done

Stop when E1–E5 are finished, or after three consecutive experiments without improvement, or when the next step needs
something outside these rules. Then:

1. Make sure `cloud/workload.py` holds the best LB configuration. If you reverted, run it once more and confirm the
   report matches.
2. Update `EXPERIMENTS.md` with all results and the final calibration, and update the "Where things stand" table above.
3. Report to the user:
   - the best LB score and its kernel version;
   - what helped and what did not;
   - which two submissions to pick as final (the best LB, plus the most robust alternative);
   - the proposed Phase 2.

   Phase 2 candidates: fusing with the strongest public pipeline (`bobthebot369/enveda-casmi-2026-v18-sovereign-zenith`,
   LB 0.425, GPU, multi-hour rerun), a PubChem candidate channel, or retraining the fingerprint model. All need the
   user's approval first.

## Pause after E2b (2026-10-08)

E2b scored **0.338**; the +0.007 gain meets the 0.003 keep threshold, so retain popularity 0.30 and v6 as best. The project SSH tunnel is disconnected and the monitor heartbeat is paused. No new server or Kaggle runs will start until the user resumes work. Calibration including v6 is deferred to a cloud run after resuming. This is a user-requested pause, not completion of Phase 1.
