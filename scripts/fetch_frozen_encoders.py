"""Fetch official frozen encoders with source revisions and integrity receipts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/pretrained/frozen-20261107'


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch(url, destination, expected_md5=None, expected_size=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest = hashlib.md5(destination.read_bytes()).hexdigest() if expected_md5 else None
        if expected_md5 is None or digest == expected_md5:
            return dict(file=str(destination.relative_to(OUT)), url=url,
                        bytes=destination.stat().st_size,
                        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), md5=digest)
    temp = destination.with_suffix(destination.suffix + '.download')
    md5, sha, size = hashlib.md5(), hashlib.sha256(), 0
    print('DOWNLOAD', destination.name, flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as stream:
        while block := response.read(1024 * 1024):
            stream.write(block); md5.update(block); sha.update(block); size += len(block)
    if expected_size is not None:
        assert size == expected_size, destination.name
    if expected_md5:
        assert md5.hexdigest() == expected_md5, destination.name
    temp.replace(destination)
    print('VERIFIED', destination.name, size, flush=True)
    return dict(file=str(destination.relative_to(OUT)), url=url, bytes=size,
                sha256=sha.hexdigest(), md5=md5.hexdigest())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dreams = get_json('https://zenodo.org/api/records/10997887')
    dreams_mirror = get_json('https://huggingface.co/api/models/roman-bushuiev/DreaMS')
    model_id = 'ibm-research/MoLFormer-XL-both-10pct'
    molecule = get_json(f'https://huggingface.co/api/models/{model_id}/revision/compat-v4')
    revision = molecule['sha']
    jobs = []
    for item in dreams['files']:
        if item['key'] in ['embedding_model.ckpt', 'ssl_model.ckpt']:
            url = f"https://huggingface.co/roman-bushuiev/DreaMS/resolve/{dreams_mirror['sha']}/{item['key']}?download=true"
            jobs.append((url, OUT/'dreams'/item['key'], item['checksum'].split(':')[1], item['size']))
    for item in molecule['siblings']:
        name = item['rfilename']
        if name.endswith('.py') or name in ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'README.md']:
            jobs.append((f'https://huggingface.co/{model_id}/resolve/{revision}/{name}', OUT/'molformer'/name, None, None))
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda args: fetch(*args), jobs))
    manifest = dict(dreams_record=10997887, dreams_mirror_revision=dreams_mirror['sha'], dreams_code_revision='dbec3a0b514a99e5056cfccde4559fda8cfe8129',
                    molecule_model=model_id, molecule_revision=revision, files=receipts)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('DONE official encoder files verified', revision, flush=True)


if __name__ == '__main__':
    main()
