# Enveda CASMI 2026 — working rules

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
