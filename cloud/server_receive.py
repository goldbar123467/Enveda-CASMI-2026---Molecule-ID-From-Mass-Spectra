"""Receive framed Kaggle data on the authorized cloud server; no local data cache."""
import argparse
import hashlib
import json
import struct
import sys
import zipfile
from urllib.parse import urlparse
from urllib.request import urlopen
from pathlib import Path

from workload import require_server_cloud


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise ValueError("Incomplete transfer frame")
        data.extend(chunk)
    return bytes(data)


def receive_frames(stream, output):
    digest, total = hashlib.sha256(), 0
    while True:
        length = struct.unpack(">I", read_exact(stream, 4))[0]
        if length == 0:
            break
        if length > 8 * 1024 * 1024:
            raise ValueError("Transfer frame is too large")
        data = read_exact(stream, length)
        output.write(data)
        digest.update(data)
        total += length
    trailer = json.loads(stream.readline(4096))
    if trailer.get("bytes") != total or trailer.get("sha256") != digest.hexdigest():
        raise ValueError("Transfer integrity verification failed")
    return {"bytes": total, "sha256": digest.hexdigest()}


def receive_download(stream, output):
    # A short-lived, signed Google object URL; never a Kaggle API key or request headers.
    request = json.loads(stream.readline(32768))
    parsed = urlparse(request["url"])
    if parsed.scheme != "https" or parsed.hostname != "storage.googleapis.com":
        raise ValueError("Only the Kaggle Google-storage download endpoint is allowed")
    digest, total = hashlib.sha256(), 0
    with urlopen(request["url"], timeout=120) as response:
        if urlparse(response.url).hostname != "storage.googleapis.com":
            raise ValueError("Download redirected outside the approved endpoint")
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    if request.get("bytes") is not None and int(request["bytes"]) != total:
        raise ValueError("Direct download size verification failed")
    return {"bytes": total, "sha256": digest.hexdigest()}


def publish_input(incoming, destination):
    # .pt and .whl are ZIP containers themselves: preserve their exact original bytes.
    if zipfile.is_zipfile(incoming):
        with zipfile.ZipFile(incoming) as archive:
            matches = [m for m in archive.infolist() if m.filename == destination.name and not m.is_dir()]
            if len(matches) == 1:
                with archive.open(matches[0]) as source, destination.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                extracted = True
            else:
                extracted = False
        if extracted:
            incoming.unlink()
            return
    incoming.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    proof = require_server_cloud(args.authorization, "cpu")
    root = Path(proof["project_root"]).resolve()
    destination = Path(args.destination).resolve()
    if not destination.is_relative_to(root / "input"):
        raise ValueError("The destination must stay inside the authorized input directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = root / "transfers" / (destination.name + ".incoming")
    incoming.parent.mkdir(parents=True, exist_ok=True)
    with incoming.open("wb") as output:
        result = receive_download(sys.stdin.buffer, output) if args.direct else receive_frames(sys.stdin.buffer, output)
    publish_input(incoming, destination)
    result["destination"] = str(destination)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Cloud input receive failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
