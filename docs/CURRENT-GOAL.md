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

## Live work, verified 12 November 2026 01:58 UTC

- `clarkkitchen/casmi26-c41r3-full-masked-validation`, version1, kernel144498742:
  RUNNING. This is the complete352-query/1292-acquisition C39 reference-masked
  P01/P14 inference, not a scoring submission.
- `clarkkitchen/casmi26-p20-c42-cpu-alignment`, version1, kernel144500368:
  COMPLETE and independently engineering-audited.11changed real lists,1unsupported
  exact fallback,388unprocessed exact fallbacks;400valid rows;0rank1/membership/
  formula-slot/identity-duplicate changes. Parent exactly matches prior P16;
  local/cloud final lists exact, raw score difference maximum6.02e-5.
- `clarkkitchen/casmi26-p21-c42-gpu-alignment`, version1, kernel144503174:
  COMPLETE and independently engineering-audited at 01:57 UTC. The GPU head also
  changes 11 real lists, with exact local/cloud rankings and zero protected
  invariant changes. This private test is not a scoring entry.

The exact-job observers use hidden Windows startup and file logs:

- `scripts/watch_c41_validation.py`, actual PID27720, observes the same full v1.
  Latest state/logs/attention are under
  `outputs/local/c41r3-masked-pipeline-validation-20261111/`.
  It will collect/audit full outputs, verify all1292original acquisitions, score
  both sealed heads locally and evaluate the fixed C39 gates. Deadline06:56UTC.
- `scripts/watch_c42_production_smokes.py`, former PID5312, has completed both
  engineering audits and exited normally. Its completion receipt, state and logs
  are under `outputs/local/c42-two-new-submissions-20261111/`.
  An initial auditor-only SDK-environment failure
  was fixed by scripts/inspect_c42_remote.py in the existing SDK venv; no cloud
  rerun or model/recipe change was needed. Prior attention/logs were archived.

Neither observer can submit to the leaderboard or open C13. Revalidate actual
API/process handles before treating a wait as live. Timeout is not termination;
never requeue merely because observation expired. Never duplicate a durable push
or submission intent; reconcile its exact handle first.

The source-sealed `scripts/continue_c42_approved_pair.py --release-approved-pair`
coordinator is now live as actual PID21248 (started 02:10 UTC, bounded to 24 hours).
It observes the same full job and existing validation observer, then independently
audits ranking and exact quality. Only if BOTH recipes pass those original gates
and BOTH production audits pass can it execute the one paired C43 confirmation,
qualify the unchanged pair, submit CPU, observe its actual score, release GPU and
stop after the second actual result. It cannot fit, tune, select a cohort, launch
another cloud run or create a third entry. Failed gates stop with a no-go receipt;
ambiguous or failed mutations stop for investigation without automatic retries.
Logs, state, launch/source seal and attention receipts are under the C42 folder.
Do not edit its sealed scripts/configs while it is running. Do not concurrently
run the same confirmation or scoring steps. The native goal still requires root
inspection of actual account results and the full completion audit.

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
The new coordinator will start this observer only after actual CPU acceptance.

Before any C39 quality or C13 outcome was available, the C43 policy received an
explicit operational amendment allowing this bounded, source-bound coordinator
to execute confirmation after the existing gates. All scientific admission
criteria, models, recipes, cohorts and the one-evaluation limit remain identical.
The earlier manual-only execution field was changed transparently in the policy;
the prior closed-gate fixture is preserved, and a new fixture proves the amended
code still refuses missing C39 quality before creating C13 preparation or consuming
labels. Ten temporary-account/gate tests pass. This implements the authorized
continuous goal; it grants no additional entries or compute budget.

Finish the goal only after both approved entries have actual terminal results,
or reconcile a demonstrated external blocker under the native blocked-audit
rules. Keep the full original objective intact across continuation turns.

The residual C44raw-GNPS audit covers the9492records outside previously admitted views. After conservative known-identity exclusion,30records remained:16invalid primary SMILES,8official identities already known,6outside prespecified target adducts. Zero surviving identities; no new panel or qualification claim. C13inventory keys were used only for global exclusion; its reserved outcomes were not accessed.

Prospective C43also rejects prior C39identity overlap and reports its chemistry neighbors, rather than dropping exposed queries after outcomes. Its point-sign gates use exact reciprocal-rank fractions. The independent C39quality audit is required before C43can be prepared. All these criteria are frozen before any C39quality outcome.
