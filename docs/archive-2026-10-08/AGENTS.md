# Enveda CASMI 2026 project instructions

## Compute policy

**Do not train locally. Use Kaggle cloud for all training, on both CPU and GPU.**

The user's later Phase 1 request authorizes CPU training and GPU fingerprint inference on the rented Vast instance `24.196.244.156:42272` (container `54749798`) as a specific exception. `cloud/server_run.py` must verify that instance's authorization and identity before numerical imports or data access. Windows and WSL remain prohibited. Competition submission reruns continue on Kaggle with internet disabled. Do not alter other projects running on the server.

- Never fit models, fine-tune, run training epochs, search hyperparameters, or benchmark training on this computer, even for a small smoke test.
- Run substantial competition preprocessing, feature generation, evaluation, and inference in Kaggle cloud too.
- Local work is limited to editing, packaging notebooks, syntax checks, unit tests with synthetic fixtures that do not train models, and Kaggle API orchestration.
- Preserve the cloud-only guard in `cloud/workload.py`. New training entry points must call it before importing training libraries, reading training data, or allocating compute.
- Use `scripts/Kaggle.ps1 prepare`, `run`, `status`, and `output` to manage private Kaggle CPU/GPU notebooks. Check the requested device and available account quota before launching a run; stop if the requested accelerator is unavailable instead of falling back to local compute.

## Credentials

- Reuse the user's Kaggle credential from the standard user-level Kaggle configuration directory, or Kaggle environment variables. The project CLI supports both legacy `kaggle.json` and modern `access_token` files.
- Never print, log, commit, or upload the API key/token. Record only its file path, presence, and authentication status in the ignored `.local/` setup report.
- Do not put Kaggle credentials in notebooks, dataset attachments, kernel metadata, command arguments, or GitHub.
- `configure-auth` imports an existing credential file to the user-level Kaggle directory and restricts its permissions. Never overwrite an existing different credential silently.

## Competition and artifacts

- Competition: `enveda-CASMI26-molecule-id-mass-spectra`.
- CPU and GPU kernel templates are private, have internet disabled, and attach the competition as an input source.
- Submission notebooks must satisfy the current Kaggle competition rules; as verified on 2026-10-07, CPU/GPU runtime is at most 9 hours, internet is disabled, and the output is named `submission.csv`.
- Keep trained models and large outputs on Kaggle or in ignored `outputs/` and `artifacts/` folders. Record seeds, configuration, package versions, kernel version, and metrics with each experiment.
- Do not download all competition data to this device as part of setup. Attach it to cloud notebooks.

## Current scope

The user requested setup only, followed by stopping. Do not launch a cloud run, start training, submit predictions, or push to GitHub as part of this setup. Future explicit work requests can authorize those actions.

Subsequent explicit requests authorize execution of `PHASE_1.md`, cloud runs and submissions within its limits, the project SSH key (ignored under `.ssh/`), and a dedicated resource-monitor agent. Record stage throughput and resource allocation when tuning server performance. GitHub pushes remain unauthorized.
