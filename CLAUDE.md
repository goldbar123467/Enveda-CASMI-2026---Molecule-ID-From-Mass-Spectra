# Enveda CASMI 2026 — working rules

## Current authorization — C09, 2026-11-09

Latest user override at approximately16:38UTC: **run more2070 training** while
Kaggle has a user-reported notebook-start/connectivity incident. C09T is already
complete:2 folds x5000 additional updates, zero restarts/AMP skips; matched old
development MRR ties C08 at.740000. Do not relaunch it. C09CT now trains the
missing matched cosine-only control in `outputs/local/c09-control-20261109`.
`scripts/run_2070_followup_queue.py` waits for that existing trainer to finish,
then pilots/resumes C09H online-hard-negative training before its longer run.
Only one local GPU workload at a time; never launch either trainer manually
while the supervisor/queue is alive. C09H:2 folds x20000 updates,48 hardest of128,
four5000-update schedule cycles/milestones,6h cap. Pilot failure stops the queue.
Dashboard http://127.0.0.1:8766 follows its active-run pointer. No submission is
authorized without the unchanged campaign execution/scientific qualification
and global one-pending gate. P15 was still PENDING/no score at16:34UTC.

The user's attached handoff explicitly resumes a **fresh improvement campaign**
using the local2070 and free Kaggle concurrently. Training/recovery/private model
uploads/qualified sequential submissions/GitHub pushes are authorized. Historical
C08/F07 exhausted-authorization and pause instructions below do not limit C09.
Exactly **ONE pending leaderboard submission globally** is allowed, including
P15/ref58163163, still pending at14:45UTC. Never automatically resubmit P15.
Use `configs/c09-campaign-20261109.json`, `.local/c09-submission-ledger.json`,
`docs/C09-CAMPAIGN-2026-11-09.md`; inspect processes before any restart.
C09T continues existing C08 fits with a new served-score objective; C09V is a
separate private P14 full-pipeline negative control and candidate-coverage audit.
The existing128-query panel is already used, not an untouched new holdout.
Keep P01/P13 at0.480 champion until an actual better nonempty score returns.
Use `scripts/c09_submission_guard.py` for fail-closed account status, not the old
two-pending submitters. No paid/rented compute. All original resource floors,
real smoke, source/artifact/numeric/identity/fallback/parity requirements remain.

## Current handoff — C08 complete/P15 accepted, 2026-11-09

C08 completed all8 x40000 fits at09:45UTC with320000 successful updates and
zero restarts. Development-only selection froze legacy16-hard /0.20 tail RRF;
reserved external delta+0.0002537 has95% interval[-0.0022017,0.0028206], so no
reliable improvement/statistical promotion is established. P15 v1 passed the
real12 Kaggle smoke and independent execution/artifact/arithmetic/identity/
local-Kaggle audit, then was accepted at12:01UTC, ref58163163, unique marker
`C08-P15-20261108`. The API confirms exactly one match, PENDING/no score.
The one new authorization is exhausted; do not re-submit or start another fit.
Training supervisor, hourly tracker and completion runner exited normally.
Keep P01/P13 at0.480 as champion while scoring runs. Read-only score updates are
allowed; source/results/GitHub sync remains authorized. See
`docs/C08-CAMPAIGN-2026-11-08.md` and its receipts. The startup instructions below
are historical and must not trigger a duplicate campaign.

## Latest resumption — 2026-11-08

Latest user override: **"Start it now and track it with a runner every hour then
submit when finished"**. Launch C08 now, keep its bounded30-minute recovery
supervisor plus hourly read-only runner, and carry through validation/packaging/
real Kaggle smoke to **one** trained-model scoring submission. No further approval
is needed for that single trial, including if scientific improvement remains
unproven (as with the prior explicit trial override); preserve the0.480 baseline
and report the evidence honestly. Execution/artifact/numeric/smoke checks remain
mandatory. Campaign authorization: `configs/compatible-campaign-20261108.json`.
Do not edit the pinned scientific trainer/config while it runs. Continue in this
task through submission/score rather than creating a job-tracking app automation.

The following preparation-only status predates that latest authorization.

User requested a fresh public-notebook strategy scan and preparation of a long
2070 run. This supersedes the older no-training-queued scope for **preparation
and short pilots**. C08 is prepared at `outputs/local/long-compatible-20261108`;
the full run/supervisor/24h clock have not started. Its isolated C08S pilot tests
all8 recipes/folds and actual checkpoint recovery. Follow
`docs/COMPATIBLE-LONG-RUN-2026-11-08.md` and its final readiness receipt. Do not
retrain F07 or rewrite its scientific sources/checkpoints.

Latest actual scores: P13 ref58105681 **0.480**, P14 v2 ref58111204 **0.458**,
both COMPLETE/no reported error. P01/P13 remain best0.480. Conditional F07 gains
did not transfer. Both previously approved scoring slots are spent; no new
submission is authorized or queued. Training completion is not qualification.
Public catalogue247 entries/20 selected sources, no downloaded code executed.
The prior request to save code/results to GitHub remains authorization to sync
this preparation; model/data/runtime/credential files remain excluded.

The dated sections below are historical and do not override this scope.

## Active resumption — 2026-11-07

Current checkpoint —7Nov20:52 Eastern: training is COMPLETE (both50k-step heads,
zero restarts). **Both approved scoring submissions are accepted:** P13 v1
ref58105681 and P14 **v2 ref58111204**, scores pending. The two-slot budget is
exhausted; no further candidate or training is queued. P14 v1 was not submitted.
V2's full real12 Kaggle smoke and independent model/fusion/fallback/dedup checks
passed; deterministic NumPy sorting fixed cross-platform peak selection and
local/Kaggle raw-score parity is0.000489. The saved budget ledger and receipts
make retries idempotent. Preserve P01's verified0.480 baseline pending scores.
The earlier active/reserved instructions below describe history; do not re-submit
or start another model based on them.

The user explicitly requested starting the proposed frozen-feature training,
public mining while it starts, and approved **two new submissions**. This
supersedes the wait below. Allocate one submission to a reproducible public-code
trial and reserve the other for the new trained candidate after validation.
Real Kaggle smoke and exact artifact checks remain required; no extra approval
is needed for those two submissions. Avoid additional candidates beyond that cap.

P13 v1 is now submitted, ref58105681 at2026-11-07T22:04UTC; score pending.
Independent real12 smoke/400valid and exact P01 primary/outsider-blend/dedup
checks passed. One trained-model slot remains approved; no second submission is
queued. The current training is **F07R1**, output
`outputs/local/long-frozen-20261107-r1`. It retains the original24h deadline,
reuses the byte-verified molecule cache, fixes fragment m/z<=1000 before DreaMS,
and passed the1000-step learning pilot. The superseded F07 was stopped before
any head fit and is retained for provenance. Follow
`docs/FROZEN-TRAINING-2026-11-07.md`; do not restart the superseded run.


P12 ref58071964 has finished with an unhandled hidden-rerun error and **no score**;
do not treat COMPLETE without a nonempty score as a successful leaderboard run.
The verified best remains P01 **0.480**. Keep that working production baseline.

Update7Nov20:29 Eastern: F07R1 finished both50k-step heads at18:22, zero
restarts. P14 v2 (`clarkkitchen/casmi26-p14-frozen-channel`) is running its
mandatory real Kaggle smoke for the second/last approved slot. It uses P01 plus
seed202611071 at50% on final lists; P01 outsiders are unchanged. Development-only
selection and fresh reserved conditional fusion check passed (+.104787 MRR,
CI[.070643,.140729]); this is not candidate recall or end-to-end validation.
Before submitting, use `audit_frozen_smoke.py` on downloaded real outputs and
verify actual COMPLETE,400 valid rows,12 real inputs, byte pins, exact fusion
and unsupported fallback. Budget ledger `.local/f07-submission-budget.json` has
P13 used and P14 reserved. Marker for P14: `F07-P14-20261107`; version2 only if
it passes, no extra approval needed, maximum two scoring submissions total.


## Current handoff — GitHub sync and wait, 2026-11-06

The latest user instruction is **"Update and push everything it GitHub with
results then wait"**. Save the current code/configs/notebook/research notes and
compact numeric evidence to GitHub, then wait. Do not launch more training,
experiments, submission controllers or extra scoring submissions until an
explicit request to resume. P12 v1/ref58071964 is already submitted and still
pending at the latest API check; its existing Kaggle scoring rerun may finish.
The verified public best remains P01 **0.480**. The local pause marker records
this handoff after the push succeeds. Read-only status updates remain allowed.

Completed evidence: L06 ten models/1,280 epochs/640,000 batches, zero restarts;
control/external reserved retrieval MRR0.680621/0.646321. P12 real Kaggle GPU
smoke and independent fivefold/50:50-blend/metric-dedup checks passed. See
`docs/LONG-SUBMISSION-2026-11-06.md` and `docs/evidence/long-2026-11-06/`.

## Current resumption — 2026-11-06

The user explicitly resumed work: research Deotte's systems and all CASMI
discussions, launch a longer local2070 training and purge obsolete artifacts.
Yesterday's end-of-day pause is superseded for that scope. Current verified best
submitted public MRR is **0.480** (P01); P02 .479 and P11 .468. Follow
`docs/LONG-TRAINING-2026-11-06.md` and its resource/recovery/validation contract.
No automatic submission is queued. Obsolete-file deletion was blocked by execution
policy; see the exact inventory in outputs/research/2026-11-06/purge-manifest.json.

The user's latest explicit instruction after completion is **"Submit it use a
submission"**. This authorizes one L06 leaderboard trial despite the earlier
statistical promotion gate. Package the control fivefold family, the strongest
reserved-confirmation family, as P12 with the fixed original50/control50 neural
recipe. Required real Kaggle smoke, exact checkpoint and numeric ensemble checks
still apply. Submit once after those execution checks; do not ask again or infer
an authorization for additional candidates.

That one trial is now submitted: **P12 v1, ref58071964, 2026-11-06 23:50UTC**.
The real Kaggle GPU smoke and independent numeric/metric-dedup audit passed.
Score pending; no additional candidate is queued or authorized by this request.

Kaggle code competition [`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
Account `clarkkitchen`. Deadline 2026-12-14 23:59 UTC. Metric MRR@25 on ~400 hidden molecules; the public
board uses ~31% (~125 molecules), so one rank-2→rank-1 flip ≈ +0.004 public.

These rules come from the user's instructions of 2026-10-28 and replace everything in `docs/archive-2026-10-08/`
(old README, AGENTS.md, PHASE_1.md, EXPERIMENTS.md — historical only, do not follow them).

## Compute

- Training, inference, preprocessing and evaluation run on Kaggle cloud (CPU or GPU notebooks) **or on this
  workstation's GPU (RTX 2070, 8 GB)** — allowed by the user on 2026-10-29. Local GPU jobs use `.venv-gpu`
  (torch + CUDA); the plain `.venv` stays CLI-only.
- **Keep the workstation CPU light**: no CPU-heavy pipelines locally (the frozen 0.465 pipeline, simulators,
  library search stay on Kaggle); local jobs use few threads. Submissions always come from Kaggle notebooks, so
  locally trained weights go up as private Kaggle datasets.
- The rented Vast server from Phase 1 is **not** used any more.
- GPU quota: **30 h/week** (T4 ×2 or P100, one bucket) + TPU v3-8 20 h/week, both reset Saturday 00:00 UTC. The
  account page (kaggle.com, "Your Kaggle Quota") is authoritative; the API's `totalTimeAllowed` reports 21600 s (6 h),
  which is wrong, but its `timeUsed` is right:
  `.venv\Scripts\python.exe -c "from kaggle.api.kaggle_api_extended import KaggleApi as K; a=K(); a.authenticate(); print(a.quota_view())"`
- A T4 ×2 session has **two** GPUs and is billed per session: run two trainings in parallel (`CUDA_VISIBLE_DEVICES`
  0 and 1), as `scripts/build_gpu_job.py` does.
- Two GPU commit runs can execute at once. Commit runs of the public-lineage notebooks cost ~20–30 GPU min
  (`COMMIT_MODE='smoke'`); the scoring rerun on the hidden test runs on Kaggle's side and takes hours.

## Submissions

- **Current override (2026-11-05): submit only after the two local rounds and
  evidence of improvement over the best submitted method.** Do not spend the
  daily allowance just to fill it. See `docs/SCIENCE-2026-11-05.md`.
- Historical default, superseded by the current override: use all 5 submissions every UTC day.
- **2026-11-05 reserve:** keep at least **2 submissions available until the first
  batch has scored**, for later decisions before 00:00 UTC. P01/P02 use 2 of 5;
  3 remain. Do not send another batch before their scores just to fill the cap.
- **Submit at most 2 at a time** (user, 2026-11-02). When working a full day, spread the batches across the day so
  later picks can use earlier scores (scoring takes ~5 h); never send all 5 at once.
- Only submit kernel versions that completed (`KernelWorkerStatus.COMPLETE`) and wrote `submission.csv`.
- Log every submission in `SUBMISSIONS.md` (slot, kernel/version, change vs. its parent, ref, public score).
- Kaggle rate-limits polling: poll no faster than once per 60 s; run long waits in the background.
- Download outputs with `scripts/fetch_output.py <slot> <dest> [regex] [--log-only]`, not `kernels output`: the CLI
  lists 20 files per request and the pipeline writes ~2000, which trips HTTP 429 for an hour or more. The helper
  asks for 100 per page (the API rejects 1000) and backs off on 429.

## Run tracking

- **Add a row to `runs.csv` the moment a run is launched** (id, platform `kaggle-gpu|kaggle-tpu|kaggle-cpu|local-2070`,
  kind, ref = `<slot> v<N>` or the local log path, started_utc, status, expect = number of `DONE` lines for local
  logs, purpose) — `scripts/add_run.py <id> <platform> <kind> "<ref>" "<purpose>" [expect]` does it.
  **Fill `result` when it finishes** (the numbers, not "done"). Then run
  `.venv/Scripts/python.exe scripts/update_runs.py` to regenerate `RUNS.md`. Never edit `RUNS.md` by hand.
- During active work, keep `scripts/update_runs.py --loop` running in the background (hourly refresh). If `.local/compute-paused.json` exists, do not restart it or any training/watcher/controller. Remove the pause marker only after an explicit user request to resume project work; merely opening a session does not resume work.
- Conclusions belong in `STRATEGY.md`, scores in `SUBMISSIONS.md`; `runs.csv` holds the per-run facts.

## Layout

| Path | What |
|---|---|
| `kaggle/forks/<slot>/` | One private Kaggle notebook per experiment: `notebook.ipynb` + `kernel-metadata.json` (id `clarkkitchen/casmi26-<slot>`) |
| `RUNS.md` / `runs.csv` | **Every compute run** (Kaggle GPU/TPU/CPU, local 2070): what, where, when, status, result. Read `RUNS.md` first when resuming |
| `scripts/update_runs.py` | Refreshes statuses + quota in `runs.csv` and regenerates `RUNS.md`; runs hourly (`--loop`, background) |
| `scripts/local_seeds.py` | Scorer seeds on the local 2070 (`.venv-gpu`): holdout/refit training, ensemble diagnostics, `--split np` |
| `scripts/build_gpu_job.py` | Kaggle T4 ×2 notebook running `local_seeds.py` jobs, one list per GPU |
| `scripts/push_queue.py` | Pushes forks in order as GPU sessions free up (optional `needs` kernels), registers them in `runs.csv` |
| `kaggle/forks/tpu_train.py` | Refits on TPU v3-8 via PyTorch/XLA (probe: `x7-tpu-probe`) |
| `SUBMISSIONS.md` | Submission log and scores |
| `STRATEGY.md` | Where the score comes from, what has been tried, next ideas |
| `scripts/make_variant.py` | Builds a fork folder from a parent fork by exact-once string patches (re-pins the fragment-check SHA when patched) |
| `scripts/autopilot.py` | Background orchestrator: pushes queued forks when <2 GPU sessions run, submits completed versions under the daily cap. Queue: `kaggle/forks/queue.json`, log: `outputs/autopilot.log` |
| `cloud/workload.py`, `scripts/kaggle_cloud.py` | Our original pipeline (best 0.338). Superseded by the public lineage; kept for reference |
| `outputs/` | Downloaded kernel outputs (ignored by git) |

## Commands (PowerShell or Git Bash, from the repo root)

```bash
.venv/Scripts/python.exe -m kaggle kernels push -p kaggle/forks/<slot> --accelerator NvidiaTeslaT4 --timeout 32400
.venv/Scripts/python.exe -m kaggle kernels status clarkkitchen/casmi26-<slot>
.venv/Scripts/python.exe scripts/fetch_output.py <slot> outputs/forks/<slot> "<regex>"
.venv/Scripts/python.exe -m kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k clarkkitchen/casmi26-<slot> -v <N> -f submission.csv -m "<slot>: <change>"
.venv/Scripts/python.exe -m kaggle competitions submissions enveda-CASMI26-molecule-id-mass-spectra
```

Private datasets (locally trained weights): put the files and a `dataset-metadata.json` in a folder, `cd` into it and
run `kaggle datasets create -p .` — a multi-level relative `-p` path breaks the CLI's upload cache. New datasets are
private unless `--public` is passed.

New fork: add an entry to `VARIANTS` in `scripts/make_variant.py` (parent, patches) and run it; it keeps
`is_private: true`, `enable_internet: false`, the dataset list and the `docker_image` pin.

Variants built by `make_variant.py` get two safety patches by default:
- **Robust rerun detection**: `IS_RERUN` is also true when `test.parquet`'s content signature differs from the visible
  file's (`ab75c9ee…`, computed by `kaggle/forks/x2-test-signature`). The frozen pipeline receives it via
  `CASMI_FORCE_RERUN` (its source and SHA-256 pin are rewritten). A public author lost a submission to 0.000 when the
  rerun was not detected.
- **Fast commit** (`fast=True`): the commit run writes 25 unique dummy SMILES per row and skips every stage
  (a few GPU minutes instead of ~25); the scoring rerun runs everything. Note the official sample file repeats `CCO`
  25 times per row, so it is not used as the placeholder.

**A fast commit tests nothing.** Any new or changed code path (a new stage, a patched module, a patched `_post`)
must first pass a smoke commit (`COMMIT_MODE='smoke'`, 12 real molecules through every stage) before any variant
using it is submitted. t3/t4 of 2026-10-29 were fast commits of an untested consensus patch that overwrote the
molecule id; both scored 0.465 — a crash after the frozen pipeline stage does not error the submission, Kaggle scores
the baseline `submission.csv` already on disk. **A score of exactly 0.465 means "something crashed".**

## Credentials and sharing

### End-of-day override — 2026-11-05

The user explicitly requested: finish the current run, submit the strongest
validated candidate, pause for today, and stop all training/experiment runs
overnight. Do not start additional training after the current q75 dose trial.
One end-of-day submission is authorized even if P01/P02 remain pending; preserve
two unused daily slots. This is a one-time exception to the normal two-pending
batch rule. Stop the project watchers and hourly updater before pausing. Submitted
Kaggle scoring reruns continue so they can return scores. Do not auto-resume.

- The Kaggle token lives in `%USERPROFILE%\.kaggle\access_token` (also `kaggle key.txt`, git-ignored). Never print,
  log, commit or upload it; never put it in notebooks or metadata.
- Keep forks private. No GitHub pushes unless the user asks.
- Public-lineage assets are mostly CC BY-NC; credit the upstream authors in any notebook that is made public.

P14 v1 was NOT submitted. Its full GPU smoke passed, but the independent local/
Kaggle comparison exposed NumPy1/2 unstable quicksort selecting different tied
peaks at the top100 cutoff (4 spectra,3 queries; max score drift0.26495).
`score_frozen_channel.py` now specifies stable sorting, with exact NumPy1.26/2.0
preprocessing parity on all28 supported visible spectra. The original selected
seed/50% weight remain fixed; the robustness check still improves reserved
conditional MRR .637481→.742594, CI delta[.070929,.141013]. V2 was pushed for
another full real12 Kaggle smoke. Use the deterministic-sort audit and compare
Kaggle output with `outputs/local/frozen-p14-stable-smoke.parquet` (not v1's
local file). `submit_frozen_once.py --version 2` enforces the one remaining slot
and requires a fresh stable-sort audit. No third scoring trial is authorized.
