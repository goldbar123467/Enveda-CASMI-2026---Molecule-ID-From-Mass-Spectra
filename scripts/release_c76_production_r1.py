"""Additional R1 dataset preflight around the immutable production launcher.

The original bound launcher retains its old R0 readiness check. Verify the
ACTUAL current mounted datasets as well without rewriting the frozen binding.
"""
import argparse
import json
import sys
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import BASE, BINDING, OPS_PLAN, verify_files


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['preflight', 'push-cpu', 'push-gpu'])
    args = parser.parse_args()
    import release_c76_production_smokes as original
    binding = original.verified()  # All source/engineering gates before API access.
    plan = read(OPS_PLAN); verify_files(plan['files_sha256'])
    assert plan['complete_pipeline_binding_sha256'] == sha(BINDING)
    bundle = ROOT / 'outputs/publish/c76-complete-direct-pair-r1-20261112'
    expected_dataset = read(bundle / 'dataset-metadata.json')['id']
    assert binding['repaired_pair_manifest_sha256'] == sha(bundle / 'pair-manifest.json')
    datasets = set()
    for trial in ['cpu', 'gpu']:
        folder = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
        metadata = read(folder / 'kernel-metadata.json')
        assert metadata['id'] == binding['candidates'][trial]['kernel']
        assert expected_dataset in metadata['dataset_sources']
        assert not any(name.endswith('/casmi26-c76-complete-direct-pair-20261112') for name in metadata['dataset_sources'])
        datasets.update(metadata['dataset_sources'])
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    states = {name: str(api.dataset_status(name)) for name in sorted(datasets)}
    assert all(value == 'ready' for value in states.values()), states
    write(BASE / 'r1-current-mounted-datasets-preflight.json', dict(checked_utc=stamp(),
        source_sha256=sha(__file__), complete_pipeline_binding_sha256=sha(BINDING),
        actual_R1_bundle_dataset=expected_dataset, actual_mounted_datasets=states,
        prior_source_gates_passed=True, C13_read=False, scoring_entries=0))
    if args.phase == 'preflight': print('C76 CURRENT R1 DATASETS READY', json.dumps(states)); return
    previous = sys.argv
    try:
        sys.argv = [str(ROOT / 'scripts/release_c76_production_smokes.py'), args.phase]
        original.main()
    finally: sys.argv = previous


if __name__ == '__main__': main()
