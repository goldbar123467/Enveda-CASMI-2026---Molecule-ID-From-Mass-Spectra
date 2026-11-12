"""Require actual local ICE/GL completion, with no prediction-budget truncation.

Read operational metadata only. Unsupported adduct/element coverage is reported,
never dropped from whole-query evaluation or treated as a quality filter.
"""
import argparse
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['fixture', 'full'])
    mode = parser.parse_args().mode
    base = ROOT / 'outputs/local/c75-local-core-port-20261112'
    port = base / (mode+'-port'); done = read(port / 'run-complete.json')
    dest = port / 'forward-coverage-audit.json'; assert not dest.exists()
    working = Path(done['working'])
    assert str(working) == '/home/imsa/.local/share/enveda-c75/'+mode+'/working'
    results = {}
    for name, rel in [('ICEBERG', 'ice_work/ice_out.json.meta.json'),
                      ('GLACIER', 'gl_work/gl_out.json.meta.json')]:
        path = working / rel; value = read(path)
        assert value['status'] == 'ok', (name, value['status'])
        assert value['n_mols_scored'] == value['n_mols_covered'], (name, 'Incomplete supported molecules')
        assert value['n_mols_scored'] > 0 and value['n_mols'] > 0
        assert value['n_pred_fail'] == 0 and value.get('n_feat_fail', 0) == 0, name
        assert value['device'] == 'cuda' and value['rdkit'] == '2025.03.6'
        results[name] = dict(metadata_sha256=sha(path),
            **{key: value[key] for key in ['n_mols', 'n_mols_scored', 'n_mols_covered',
                'n_cands', 'n_scored', 'n_pred', 'n_pred_fail', 'n_bad_smiles', 'seconds']},
            feature_failures=value.get('n_feat_fail', 0), supported_molecules_complete=True,
            time_budget_truncation=False)
    assert results['ICEBERG']['n_mols'] == done['queries']
    assert results['GLACIER']['n_mols'] <= done['queries']
    write(dest, dict(completed_utc=stamp(), source_sha256=sha(__file__),
        mode=mode, run_complete_sha256=sha(port / 'run-complete.json'),
        queries=done['queries'], acquisitions=done['acquisitions'], channels=results,
        all_supported_forward_molecules_complete=True, no_prediction_or_feature_failures=True,
        unsupported_coverage_changes_evaluation_denominator=False,
        query_truth_or_quality_read=False, C13_read=False, scoring_entries=0, passed=True))
    print('C75 ACTUAL FORWARD CHANNEL COMPLETION AUDIT PASS', mode, flush=True)


if __name__ == '__main__': main()
