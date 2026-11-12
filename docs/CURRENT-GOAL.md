# Current project goal — active C38/C42 cycle

Complete the two newly approved scoring trials, provisionally one CPU-fitted
ridge alignment and one GPU-fitted nonlinear alignment. Both use frozen GPU
encoders and a complete GPU parent; CPU-fitted does not mean CPU-only serving.
Observe the first actual terminal score before releasing the second. Preserve
the 0.480 champion and all earlier evidence. No new paid compute, hourly
campaign, exhausted P18/P19 resubmission, truth insertion or favorable-cohort
selection. No repeat submission approval is needed. Two approvals remain unused.

The native goal remains ACTIVE. Current authorization is
`.local/c38-active-goal.json` and
`outputs/local/c38-source-support-20261111/authorization.json`. Keep the old
`.local/compute-paused.json` so historical workers stay inactive. Current controlled
workers use the explicit new authorization. Do not restart old campaigns.

## Live work, verified 12 November 2026 01:36 UTC

- `clarkkitchen/casmi26-c41r3-full-masked-validation`, version1, kernel144498742:
  RUNNING. This is the complete352-query/1292-acquisition C39 reference-masked
  P01/P14 inference, not a scoring submission.
- `clarkkitchen/casmi26-p20-c42-cpu-alignment`, version1, kernel144500368:
  COMPLETE and independently engineering-audited.11changed real lists,1unsupported
  exact fallback,388unprocessed exact fallbacks;400valid rows;0rank1/membership/
  formula-slot/identity-duplicate changes. Parent exactly matches prior P16;
  local/cloud final lists exact, raw score difference maximum6.02e-5.
- `clarkkitchen/casmi26-p21-c42-gpu-alignment`, version1, kernel144503174:
  RUNNING as a private production engineering test, not a scoring entry.

Two detached observers have hidden Windows startup and file logs:

- `scripts/watch_c41_validation.py`, actual PID27720, observes the same full v1.
  Latest state/logs/attention are under
  `outputs/local/c41r3-masked-pipeline-validation-20261111/`.
  It will collect/audit full outputs, verify all1292original acquisitions, score
  both sealed heads locally and evaluate the fixed C39 gates. Deadline06:56UTC.
- `scripts/watch_c42_production_smokes.py`, actual PID5312, has audited P20v1 and
  released P21v1 after its engineering pass. It now observes/audits P21v1. State/logs/attention are under
  `outputs/local/c42-two-new-submissions-20261111/`. Deadline07:32UTC. An initial auditor-only SDK-environment failure
  was fixed by scripts/inspect_c42_remote.py in the existing SDK venv; no cloud
  rerun or model/recipe change was needed. Prior attention/logs were archived.

Neither observer can submit to the leaderboard or open C13. Revalidate actual
API/process handles before treating a wait as live. Timeout is not termination;
never requeue merely because observation expired. Never duplicate a durable push
or submission intent; reconcile its exact handle first.

## Completed evidence

C39 has352queries/1292original acquisitions and273chemistry groups. Development
and source groups are disjoint. New C40R1 fits use174328identities/600815acquisitions;
all identified original-row aliases and C39 chemistry were excluded. Initial
C40heads and C41/R2smokes are retired/quarantined, without quality evaluation.

C41R3smoke v1 is COMPLETE and collected: actual dualT4;409162main reference rows
and59583Engine-owned analog representatives removed;408950Engine2rows removed;
zero forbidden rows; exact P01replay; all12queries/37original acquisitions exact.
The guard must cover actual `Engine._reps`, not only Library.representatives.
Sixteen targeted tests and the exact downloaded-method fixture pass.

Both sealed C42R1heads execute on that real12 parent:8changed lists each,
4unsupported exact fallbacks, zero rank1/membership/formula-slot changes.
Independent RRF reconstruction is exact. This is engineering proof, not quality.

The current immutable head bundle is
`outputs/publish/c42r1-crossmodal-heads-20261111/` with manifestSHA
`a534d8654ce8d407b6cee9b1c9b6fa7c2e8231dde84e6a232957d097fee8bed6`.
P20/P21notebooks are frozen under `kaggle/forks/`. Their P16parent differs only
by the tested legacy scalar formate source copy and the new required head stage.
No validation-reference masks or query labels are mounted in production.

C39quality and fresh C13 metrics remain UNREAD as of this state check. No new
scoring entry has been released. Full data/labels/weights stay in ignored local
storage. Compact receipts and the source study report are in
`docs/evidence/c38-c42-20261111/` and
`docs/C38-C42-SOURCE-AND-HEADS-2026-11-11.md`.

## Next required work

After full collection, independently audit its actual C42ranking with
`scripts/audit_c42_ranking.py`, then independently recompute all352MRR ranks and exact rational directional
  signs with scripts/audit_c42_quality.py, and review the fixed C39gates. Do not weaken or
fabricate admission. The study is controlled in-corpus validation: public encoder
and parent historical exposure remains known or unknown; it is not universally
unseen validation or established leaderboard calibration.

Only if both fixed recipes pass their prior/source checks, use the prospective
paired policy `configs/c43-paired-confirmation-20261111.json` and
`scripts/confirm_c43_pair.py`: prepare, score, then explicitly evaluate --consume.
The first two phases do not open labels. The last consumes both fixed predictions
jointly under a global C13marker; no recipe/cohort/model revisions follow it.
Exact crash recovery must retain the same source/policy/prediction seal. Report
all-query zeros, query modes, grouped uncertainty, fitting identity/neighbor
exposure, and P14control. C13is FT/HCDsingle-acquisition source stress and does
not substitute for timsTOF target support. The closed-gate fixture passes.

Actual P20/P21production smokes must pass
`scripts/audit_c42_production.py` (remote source, preserved parent, identities,
independent math, local/cloud agreement and fallback). Once all evidence passes,
`scripts/qualify_c42_pair.py` seals the pair. `scripts/submit_c42_pair.py` enforces
two accepted entries maximum, a single pending account entry, durable intents,
CPUfirst and its actual score before GPUrelease. These are exploratory trials;
only actual public scores establish improvement. No scoring command has run.

The source-bound `scripts/watch_c42_scores.py` is ready but has NOT been started.
Start it with `.venv` and `--release-second` only after qualification and actual
CPU acceptance. It observes real account status, waits for a numeric terminal
CPU score, releases the same qualified GPU entry through the submission guard,
and stops after both actual results. Missing observations never imply failure
or authorize a repeat submission. A real scoring error stops for investigation.
Five temporary-account tests pass: delayed score propagation, interrupted API
observation, another pending entry, missing qualification, and terminal failure.
Ambiguous accepted-response recovery requires the exact durable qualification,
source and kernel intent, and preserves the qualification pin in the ledger.

Finish the goal only after both approved entries have actual terminal results,
or reconcile a demonstrated external blocker under the native blocked-audit
rules. Keep the full original objective intact across continuation turns.

The residual C44raw-GNPS audit covers the9492records outside previously admitted views. After conservative known-identity exclusion,30records remained:16invalid primary SMILES,8official identities already known,6outside prespecified target adducts. Zero surviving identities; no new panel or qualification claim. C13inventory keys were used only for global exclusion; its reserved outcomes were not accessed.

Prospective C43also rejects prior C39identity overlap and reports its chemistry neighbors, rather than dropping exposed queries after outcomes. Its point-sign gates use exact reciprocal-rank fractions. The independent C39quality audit is required before C43can be prepared. All these criteria are frozen before any C39quality outcome.
