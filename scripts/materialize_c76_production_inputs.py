"""Repair only owned input discovery: real directories, links to original files.

Path.rglob does not descend through the initial directory links. Preserve that
unused layout and expose exactly the same files with directory structure intact.
No scientific source, asset, query, permission or active namespace is changed.
"""
import os
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112/local-production-preview'
    pin = read(base / 'preview-freeze.json')
    native = Path('/home/imsa/.local/share/enveda-c76/production-preview')
    source = native / 'input'; archive = native / 'input-layout-attempt01'
    assert source == Path(pin['input']) and source.parent.resolve() == native.resolve()
    assert source.is_dir() and not source.is_symlink() and not archive.exists()
    assert Path('/kaggle/input').resolve() == Path(pin['original_input_target'])
    assert not (base / 'run-intent.json').exists() and not (base / 'input-layout-amendment.json').exists()
    missing = dict(public_checkpoints=len(list(source.rglob('final_model.pt'))),
                   pair_manifests=len(list(source.rglob('pair-manifest.json'))))
    assert missing == {'public_checkpoints': 0, 'pair_manifests': 0}
    write(base / 'initial-input-discovery-guard.json', dict(checked_utc=stamp(),
        reason='Directory links are not traversed by Path.rglob; zero checkpoints/manifests discovered before inference',
        actual_counts=missing, inference_started=False, all_original_files_preserved=True))
    os.rename(source, archive)
    source.mkdir()
    targets = {}
    def expose(original, target, ancestors=()):
        real = original.resolve(strict=True)
        if real.is_dir():
            assert real not in ancestors, ('Unexpected input directory cycle', real)
            target.mkdir()
            for child in sorted(real.iterdir()): expose(child, target / child.name, ancestors + (real,))
        else:
            assert real.is_file()
            target.symlink_to(real)
            targets[target.relative_to(source).as_posix()] = str(real)
    for item in sorted(archive.iterdir()): expose(item, source / item.name)
    assert list(source.rglob('pair-manifest.json')) and list(source.rglob('final_model.pt'))
    assert not list(source.rglob('validation-contract.json'))
    write(base / 'input-file-links.json', targets)
    write(base / 'input-layout-amendment.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        original_preview_freeze_sha256=sha(base / 'preview-freeze.json'),
        original_layout_preserved=str(archive), new_owned_layout=str(source),
        linked_files=len(targets), file_links_sha256=sha(base / 'input-file-links.json'),
        source_and_asset_bytes_changed=False, only_directory_discoverability_repaired=True,
        active_namespace_changed=False, inference_started=False, quality_read=False,
        C13_read=False, scoring_entries=0))
    print('C76 OWNED INPUT TREE MATERIALIZED; ALL FILES POINT TO ORIGINAL BYTES', len(targets), flush=True)


if __name__ == '__main__': main()
