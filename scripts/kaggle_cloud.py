"""Local Kaggle orchestration. Never imports or executes the cloud workload."""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/project.json"


def read_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def credential_info():
    if os.environ.get("KAGGLE_API_TOKEN"):
        return {"present": True, "source": "KAGGLE_API_TOKEN environment variable"}
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return {"present": True, "source": "KAGGLE_USERNAME/KAGGLE_KEY environment variables"}
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    for name in ("access_token", "kaggle.json", "credentials.json"):
        path = config_dir / name
        if path.is_file():
            return {"present": True, "source": str(path)}
    return {"present": False, "source": None}


def username_from_local(config, override=None):
    username = override or config.get("username") or os.environ.get("KAGGLE_USERNAME")
    if not username:
        config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
        path = config_dir / "kaggle.json"
        if path.is_file():
            username = json.loads(path.read_text(encoding="utf-8-sig")).get("username")
    if not username:
        raise ValueError("Kaggle username is not configured. Use prepare --username YOUR_KAGGLE_USERNAME.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Invalid Kaggle username format.")
    return username


def make_notebook(device, workload):
    source = workload + f"\nmain(device={device!r})\n"
    # Avoid the module's __main__ block running once before the requested device.
    source = source.replace('if __name__ == "__main__":\n    main()\n', "")
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "project-notes",
                "metadata": {},
                "source": [
                    f"# Enveda CASMI 2026 {device.upper()}\n",
                    "Private Kaggle cloud notebook. Internet disabled.\n",
                    "Generated from cloud/workload.py; writes submission.csv and run_report.json.\n",
                    "Edit cloud/workload.py and prepare again to package changes.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "cloud-workload",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def metadata(config, device, username):
    result = {
        "id": f"{username}/enveda-casmi-2026-{device}",
        "title": f"Enveda CASMI 2026 {device.upper()}",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": device == "gpu",
        "enable_internet": False,
        "competition_sources": [config["competition"]],
        "dataset_sources": config["dataset_sources"],
        "kernel_sources": config["kernel_sources"],
        "model_sources": config["model_sources"],
    }
    if device == "gpu":
        result["machine_shape"] = config["gpu_accelerator"]
    return result


def prepare(device, username=None, template=False):
    config = read_config()
    owner = "YOUR_KAGGLE_USERNAME" if template else username_from_local(config, username)
    folder = ROOT / ("kaggle" if template else ".local/kernels") / device
    workload = (ROOT / "cloud/workload.py").read_text(encoding="utf-8")
    notebook = make_notebook(device, workload)
    ast.parse("".join(notebook["cells"][1]["source"]))
    write_json(folder / "notebook.ipynb", notebook)
    write_json(folder / "kernel-metadata.json", metadata(config, device, owner))
    return folder


def require_credentials():
    if not credential_info()["present"]:
        raise ValueError("No Kaggle credential found. Import an existing file with configure-auth first.")


def kaggle(*args):
    require_credentials()
    # Tokens remain in the user's credential store/environment, never in argv.
    return subprocess.run([sys.executable, "-m", "kaggle", *args], cwd=ROOT, check=False).returncode


def check(online=False):
    config = read_config()
    if not 0 < config["timeout_seconds"] <= 32400:
        raise ValueError("Competition cloud timeout must be between 1 second and 9 hours.")
    import nbformat

    for device in ("cpu", "gpu"):
        folder = ROOT / "kaggle" / device
        notebook = nbformat.read(folder / "notebook.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        kernel_meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        if kernel_meta != metadata(config, device, "YOUR_KAGGLE_USERNAME"):
            raise ValueError(f"{device} metadata is stale. Regenerate the templates.")

    auth = credential_info()
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_training_allowed": False,
        "templates_valid": True,
        "credential": auth,
        "authentication_verified": False,
        "competition_access_verified": False,
        "cloud_runs_launched_by_check": 0,
    }
    if online:
        result["authentication_verified"] = kaggle("kernels", "list", "--mine", "--page-size", "1") == 0
        if result["authentication_verified"]:
            # The CLI introspects modern tokens to obtain the account username.
            # Reuse that API behavior without logging the token or config dict.
            from kaggle import api

            owner = username_from_local(config, api.config_values.get("username"))
            config["username"] = owner
            write_json(CONFIG, config)
            result["username"] = owner
            result["competition_access_verified"] = kaggle(
                "competitions", "files", config["competition"], "--page-size", "200"
            ) == 0
    write_json(ROOT / ".local/setup-report.json", result)
    print("CPU/GPU notebook templates validated. Local training is disabled.")
    print("Kaggle credential: " + (auth["source"] if auth["present"] else "not found"))
    if online:
        print(f"Authentication verified: {result['authentication_verified']}")
        print(f"Competition access verified: {result['competition_access_verified']}")
        return 0 if result["competition_access_verified"] else 1
    return 0


def restrict_permissions(path):
    if os.name == "nt":
        account = os.environ["USERDOMAIN"] + "\\" + os.environ["USERNAME"]
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{account}:(F)"],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Could not restrict credential file permissions.")
    else:
        path.chmod(0o600)


def configure_auth(source, username=None):
    source = Path(source).expanduser().resolve()
    # Only parse locally; never include the credential content in a diagnostic.
    content = source.read_text(encoding="utf-8-sig").strip()
    if content.startswith("{"):
        try:
            values = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("The credential file is not valid JSON.") from None
        if not isinstance(values, dict) or not values.get("username") or not values.get("key"):
            raise ValueError("Legacy credential JSON must contain username and key.")
        username = values["username"]
        filename = "kaggle.json"
        content = json.dumps({"username": username, "key": values["key"]}) + "\n"
    else:
        if not content.startswith("KGAT_") or any(ch.isspace() for ch in content):
            raise ValueError("Expected a legacy kaggle.json or modern Kaggle API token file.")
        filename = "access_token"
        content += "\n"
    config = read_config()
    if username:
        username_from_local(config, username)
    destination = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.resolve() != source:
        if destination.read_text(encoding="utf-8-sig").strip() != content.strip():
            raise ValueError("A different credential already exists at the destination; it was preserved.")
    if destination.resolve() != source:
        destination.touch(exist_ok=True)
        restrict_permissions(destination)
        destination.write_text(content, encoding="utf-8")
    else:
        restrict_permissions(destination)
    if username:
        config["username"] = username
        write_json(CONFIG, config)
    write_json(ROOT / ".local/credential-record.json", {
        "configured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "credential_path": str(destination),
        "credential_value_logged": False,
    })
    print(f"Credential stored with restricted permissions: {destination}")
    print("Credential value was not printed or logged.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checker = commands.add_parser("check", help="Validate setup without training")
    checker.add_argument("--online", action="store_true", help="Also check authentication and competition files")
    auth = commands.add_parser("configure-auth", help="Securely import an existing local credential file")
    auth.add_argument("--credential", required=True, help="File path only; do not pass a key value")
    auth.add_argument("--username", help="Kaggle username for modern token files")
    commands.add_parser("build-templates", help="Regenerate tracked notebook templates; never runs them")
    for name in ("prepare", "run", "status", "output"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=("cpu", "gpu"), required=True)
        command.add_argument("--username")
    submit = commands.add_parser("submit", help="Submit a completed kernel version's submission.csv")
    submit.add_argument("--device", choices=("cpu", "gpu"), required=True)
    submit.add_argument("--version", required=True, help="Kernel version number to submit")
    submit.add_argument("--message", required=True)
    submit.add_argument("--username")
    commands.add_parser("submissions", help="List this account's competition submissions")
    commands.add_parser("files", help="List cloud competition file metadata")
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args.online)
    if args.command == "configure-auth":
        return configure_auth(args.credential, args.username)
    if args.command == "build-templates":
        for device in ("cpu", "gpu"):
            print(prepare(device, template=True))
        return 0
    config = read_config()
    if args.command == "files":
        return kaggle("competitions", "files", config["competition"], "--page-size", "200")
    if args.command == "submissions":
        return kaggle("competitions", "submissions", config["competition"])
    if args.command == "prepare":
        print(prepare(args.device, args.username))
        print("Prepared locally. Nothing was uploaded or executed.")
        return 0
    if args.command == "run":
        require_credentials()
        folder = prepare(args.device, args.username)
        push_args = ["kernels", "push", "--path", str(folder), "--timeout", str(config["timeout_seconds"])]
        if args.device == "gpu":
            push_args += ["--accelerator", config["gpu_accelerator"]]
        print(f"Launching private {args.device.upper()} Kaggle cloud notebook.")
        return kaggle(*push_args)
    owner = username_from_local(config, args.username)
    kernel = f"{owner}/enveda-casmi-2026-{args.device}"
    if args.command == "status":
        return kaggle("kernels", "status", kernel)
    if args.command == "output":
        output = ROOT / "outputs" / args.device / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return kaggle("kernels", "output", kernel, "--path", str(output))
    if args.command == "submit":
        if not args.version.isdigit():
            raise ValueError("Kernel version must be a number.")
        return kaggle(
            "competitions", "submit", config["competition"], "-k", kernel, "-v", args.version,
            "-f", "submission.csv", "-m", args.message,
        )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        # Avoid traceback/source excerpts from files that can contain credentials.
        print(f"Setup error: {error}", file=sys.stderr)
        sys.exit(1)
