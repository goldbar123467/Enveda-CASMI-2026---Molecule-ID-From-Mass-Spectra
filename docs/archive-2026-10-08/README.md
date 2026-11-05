# Enveda CASMI 2026 — Molecule ID From Mass Spectra

Workspace for [the Kaggle competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra), linked to [this GitHub repository](https://github.com/goldbar123467/Enveda-CASMI-2026---Molecule-ID-From-Mass-Spectra).

**All training runs on Kaggle cloud. Do not train locally, on CPU or GPU.** This computer edits code, packages notebooks, and controls cloud jobs. See `AGENTS.md`.

For Phase 1, the user also authorized the rented Vast instance at `24.196.244.156:42272`. Its guarded entry point is `cloud/server_run.py`; authorization binds execution to the verified container, and Windows/WSL still refuse execution. Inputs download directly to that server through short-lived storage links passed over SSH stdin; Kaggle API credentials stay on this PC. Submission notebooks still rerun privately on Kaggle with internet disabled. The project SSH key lives in ignored `.ssh/`.

`cloud/workload.py` is the submission pipeline. `prepare` embeds it into a self-contained private notebook that writes `/kaggle/working/submission.csv` and `run_report.json` (configuration, package versions, timings, validation grid). It retrieves candidates from training structures ∪ COCONUT and ranks them by spectral-library, analog-propagation, fingerprint-model and fragmentation evidence. See `EXPERIMENTS.md` for each version's validation and public leaderboard score.

## Local setup

```powershell
.\scripts\Setup.ps1
.\scripts\Kaggle.ps1 check
```

The `.venv` contains only Kaggle CLI orchestration dependencies. Python 3.11+ is required; `Setup.ps1 -PythonPath C:\path\to\python.exe` can select another interpreter. Kaggle CLI is pinned in `requirements-local.txt`, with the installed environment captured in `requirements-local.lock.txt`.

## Kaggle authentication

Kaggle credentials live outside Git in `%USERPROFILE%\.kaggle\kaggle.json` (legacy key) or `%USERPROFILE%\.kaggle\access_token` (modern token), or a directory selected by `KAGGLE_CONFIG_DIR`. Existing Kaggle environment variables also work. Keys are never embedded into notebooks or kernel metadata.

Import an existing saved credential securely, using its **file path**:

```powershell
.\scripts\Kaggle.ps1 configure-auth --credential 'C:\path\to\kaggle.json'
# For a modern token file, an explicit username is optional:
.\scripts\Kaggle.ps1 configure-auth --credential 'C:\path\to\access_token' --username YOUR_KAGGLE_USERNAME
.\scripts\Kaggle.ps1 check --online
```

`configure-auth` restricts file permissions and records only its location in ignored `.local/credential-record.json`. `check --online` makes read-only API calls to verify authentication, resolve your Kaggle username into the project config, and list competition files. It starts no jobs. If no credential exists, use [Kaggle API settings](https://www.kaggle.com/settings/api) to save a credential locally first. Accept competition rules on Kaggle and complete account/phone verification if Kaggle requires it for GPU access.

## Prepare cloud jobs without launching

```powershell
.\scripts\Kaggle.ps1 prepare --device cpu --username YOUR_KAGGLE_USERNAME
.\scripts\Kaggle.ps1 prepare --device gpu --username YOUR_KAGGLE_USERNAME
```

If a legacy credential or `configs/project.json` supplies the username, omit `--username`. Prepared bundles are written to ignored `.local/kernels/cpu` and `.local/kernels/gpu`. Inspect their `kernel-metadata.json` and `notebook.ipynb` before running. Preparation never uploads or executes code.

## Cloud runs and submissions

These commands **launch Kaggle cloud jobs** or **submit to the competition**:

```powershell
.\scripts\Kaggle.ps1 run --device cpu          # push a new kernel version and run it
.\scripts\Kaggle.ps1 status --device cpu
.\scripts\Kaggle.ps1 output --device cpu       # submission.csv, run_report.json, log -> outputs/
.\scripts\Kaggle.ps1 submit --device cpu --version 3 --message "what changed"
.\scripts\Kaggle.ps1 submissions               # scores of this account's submissions
```

`submit` sends a completed kernel version's `submission.csv`; Kaggle then reruns that version on the hidden test (about 30–60 min). The competition allows five submissions per day.

Attached public inputs (`configs/project.json`): `prvsiyan/coconut-casmi26-candidates` (COCONUT candidates and fingerprint bit selection), `prvsiyan/casmi26-fp-models-v2` (pretrained spectrum→fingerprint models), and `metric/rdkit-2026-3-3-wheel` (offline RDKit, the version the metric uses).

The CLI authenticates using your local Kaggle credential. CPU metadata disables GPU; GPU metadata selects `NvidiaTeslaT4` (T4 x2). Actual accelerator access and quota depend on your Kaggle account. No quota has been consumed or GPU allocation verified during setup. Downloaded outputs go to a separate timestamped directory under ignored `outputs/`.

Edit `cloud/workload.py` for future preprocessing, training, or inference, preserving its Kaggle-only guard. `prepare` embeds that file into a self-contained notebook, so it works with internet disabled and needs no Git clone or local package installation on Kaggle. Regenerate the tracked reference notebooks with `Kaggle.ps1 build-templates` after source/config changes. Configure additional pre-uploaded datasets, notebook outputs, or model attachments in `configs/project.json` for offline dependencies and trained checkpoints. Do not attach credential files.

## Competition settings

As verified on 2026-10-07, this is a notebook submission competition: CPU and GPU submissions have a 9-hour runtime limit, internet disabled, and an output named `submission.csv`. The launch timeout is set to 32,400 seconds. The official [competition page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) is authoritative; recheck its rules before submitting.

Competition data is attached through `competition_sources`; setup does not download it locally. Evaluation uses MRR@25. The CSV needs `molecule_id,smiles`, one row per molecule, and up to 25 SMILES guesses separated by semicolons. No prediction submission was made during setup.

Kaggle references: [authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication), [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md), [cloud kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md). The installed CLI's `kernels push` starts a run, so setup only prepares local bundles.

## Setup status — 2026-10-07

Configured Kaggle account: `clarkkitchen`. The existing token from `kaggle key.txt` was saved securely to `C:\Users\imsa\.kaggle\access_token`. Both token files have restricted Windows permissions; the project copy is ignored by Git. The token value is absent from project logs and notebooks.

Read-only checks verified authentication and access to the competition file listing. Account-specific private CPU/GPU bundles are prepared under `.local/kernels/`. No cloud run, local training, competition submission, or GitHub push was performed. GPU allocation and quota remain untested because setup did not launch a session. See ignored `.local/setup-report.json` and `.local/credential-record.json` for the setup records.
