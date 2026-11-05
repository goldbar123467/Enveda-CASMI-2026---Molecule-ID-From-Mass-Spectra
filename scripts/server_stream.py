"""Kaggle API orchestration: stream inputs to the authorized server, never to local disk."""
import argparse
import hashlib
import json
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace/enveda-casmi"
SSH = ["ssh", "-i", str(ROOT / ".ssh/enveda_casmi_2026_ed25519"),
       "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
       "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={ROOT / '.local/server_known_hosts'}",
       "-p", "42272", "root@24.196.244.156"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    if args.competition == bool(args.dataset):
        raise ValueError("Select exactly one approved input source")
    approved = set(json.loads((ROOT / "configs/project.json").read_text())["dataset_sources"])
    # PHASE_1.md explicitly names E3: inputs may be prepared before its scoring experiment.
    approved.add("prvsiyan/chebi-lipidmaps-casmi26")
    if args.dataset and args.dataset not in approved:
        raise ValueError("The dataset is not attached in the approved project configuration")
    if not args.folder.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid input folder")
    from kaggle import api

    manifest_path = ROOT / f".local/server-transfers-{args.folder}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
    for name in args.files:
        if Path(name).name != name:
            raise ValueError("Input filenames must be basenames")
        destination = f"{REMOTE}/input/{args.folder}/{name}"
        check = subprocess.run(SSH + [f"test -f {shlex.quote(destination)}"], capture_output=True)
        known = next((m for m in manifest if m.get("destination") == destination), None)
        if check.returncode == 0 and known is not None:
            print("Already transferred and verified: " + name, flush=True)
            continue
        if check.returncode not in (0, 1):
            raise RuntimeError("SSH input preflight failed")

        def stream_response(response, outfile, http_client, quiet=True, resume=False):
            command = (f"{REMOTE}/.venv/bin/python {REMOTE}/cloud/server_receive.py "
                       f"--authorization {REMOTE}/server-authorization.json "
                       f"--destination {shlex.quote(destination)}")
            if args.direct:
                if urlparse(response.url).hostname != "storage.googleapis.com":
                    raise ValueError("The source does not provide an approved direct download URL")
                request = {"url": response.url, "bytes": response.headers.get("Content-Length")}
                response.close()
                print(f"Direct cloud transfer: {name}", flush=True)
                result = subprocess.run(SSH + [command + " --direct"],
                                        input=(json.dumps(request)+"\n").encode(), capture_output=True)
                if result.returncode != 0:
                    print(result.stderr.decode(errors="replace")[-1000:], file=sys.stderr)
                    raise RuntimeError("Direct cloud input transfer failed")
                acknowledgement = json.loads(result.stdout)
                if request["bytes"] is not None and int(request["bytes"]) != acknowledgement["bytes"]:
                    raise ValueError("Direct cloud acknowledgement size mismatch")
                acknowledgement.update({"source": args.dataset or "enveda-CASMI26-molecule-id-mass-spectra", "file": name})
                manifest.append(acknowledgement)
                manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
                print(f"Verified on server: {name} ({acknowledgement['bytes']:,} bytes)", flush=True)
                return
            process = subprocess.Popen(SSH + [command], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            digest, total, started, last_print = hashlib.sha256(), 0, time.monotonic(), 0.0
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    process.stdin.write(struct.pack(">I", len(chunk)))
                    process.stdin.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                    elapsed = time.monotonic() - started
                    if elapsed - last_print >= 30:
                        print(f"Streaming {name}: {total:,} bytes in {elapsed:.0f}s", flush=True)
                        last_print = elapsed
                process.stdin.write(struct.pack(">I", 0))
                footer = {"bytes": total, "sha256": digest.hexdigest()}
                process.stdin.write((json.dumps(footer) + "\n").encode())
                process.stdin.close()
                stdout = process.stdout.read()
                stderr = process.stderr.read()
                code = process.wait()
                if code != 0:
                    # No HTTP URLs, request headers, or credentials go into error logs.
                    print(stderr.decode(errors="replace")[-2000:], file=sys.stderr)
                    raise RuntimeError("Authorized server rejected the input transfer")
                result = json.loads(stdout)
                if result.get("bytes") != total or result.get("sha256") != digest.hexdigest():
                    raise RuntimeError("Input transfer acknowledgement does not match")
                result.update({"source": args.dataset or "enveda-CASMI26-molecule-id-mass-spectra", "file": name})
                manifest.append(result)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"Verified on server: {name} ({total:,} transferred bytes)", flush=True)
            except BaseException:
                process.terminate()
                process.wait()
                raise
            finally:
                response.close()

        api.download_file = stream_response
        placeholder = str(ROOT / ".local/stream-placeholder")
        if args.competition:
            api.competition_download_file("enveda-CASMI26-molecule-id-mass-spectra", name,
                                          path=placeholder, force=True, quiet=True)
        else:
            api.dataset_download_file(args.dataset, name, path=placeholder, force=True, quiet=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Cloud transfer failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
