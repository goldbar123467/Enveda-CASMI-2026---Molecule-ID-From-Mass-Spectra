"""Evaluate the one predeclared C12 recipe on40 used dev identities; never submit."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase

from precision_cache import ROOT, sha, write, features
from prepare_c12_pool import OUT, CONFIG, CLOUD
from prefix_expansion import expand
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary


def main():
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3'
    out = OUT / 'development'
    assert not (OUT / 'development.json').exists(), 'Development result is sealed'
    assert not (OUT / 'development-protocol.json').exists(), 'Evaluation already started; inspect existing run before recovery'
    contract = json.loads((OUT / 'source-contract.json').read_text())
    assert sha(CONFIG) == contract['config_sha256']
    for name, pin in contract['source_sha256'].items(): assert sha(ROOT / 'scripts' / name) == pin, name
    receipt = json.loads((out / 'scoring-receipt.json').read_text())
    assert receipt['scope'] == 'development' and receipt['queries'] == 40 and receipt['input_errors'] == 0
    assert not receipt['prediction_metrics_computed'] and not receipt['query_labels_read']
    assert receipt['source_contract_sha256'] == sha(OUT / 'source-contract.json')
    assert receipt['scores_sha256'] == sha(out / 'scores.parquet')
    config = json.loads(CONFIG.read_text())
    for value in receipt['local_cloud_parity'].values(): assert value['max_raw_score_abs_difference'] <= config['local_raw_score_parity_tolerance']
    local = ROOT / 'outputs/local/c10-casmi2022-20261110'
    write(OUT / 'development-protocol.json', dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        evaluator_sha256=sha(__file__), source_contract_sha256=sha(OUT / 'source-contract.json'),
        scoring_receipt_sha256=sha(out / 'scoring-receipt.json'), labels_sha256=sha(local / 'labels.csv'),
        selection='One fixed candidate; retain only if positive development increment overP16 with at least one absent truth recovered. No parameter search.',
        reserved_prediction_metrics_read=False, automatic_submission=False))
    labels = pd.read_csv(local / 'labels.csv', dtype=str, nrows=40).set_index('molecule_id')
    assert len(labels) == 40 and labels.partition.eq('development').all()
    assert labels.index.tolist() == json.loads((local / 'partitions.json').read_text())['development']
    parent = pd.read_csv(out / 'parent.csv', dtype=str, keep_default_na=False)
    assert parent.molecule_id.tolist() == labels.index.tolist()
    scores = pd.read_parquet(out / 'scores.parquet')
    assert scores.neural_ok.astype(bool).all() and scores.molecule_id.nunique() == 40
    identity_cache = dict(zip(scores.smiles.astype(str), scores.metric_key.astype(str)))
    def identity(smiles):
        if smiles not in identity_cache: identity_cache[smiles] = features.structure(smiles)[0]
        return identity_cache[smiles]
    def subset(frame): return frame.set_index('molecule_id').loc[labels.index].reset_index()
    frames = dict(P16=parent)
    frames['P01'] = subset(pd.read_csv(CLOUD / 'c10_p01.csv', dtype=str, keep_default_na=False))
    frames['P14'] = subset(pd.read_csv(CLOUD / 'c10_p14.csv', dtype=str, keep_default_na=False))
    frames['P15'] = subset(pd.read_csv(ROOT / 'outputs/local/c10-candidates-20261110/p15-control.csv', dtype=str, keep_default_na=False))
    for frame in frames.values():
        for row in frame.itertuples(index=False):
            for smiles in row.smiles.split(';'): identity(smiles)
    final, actions = expand(parent, scores, config)
    frames['C12'] = final
    values = {name: metric_vector(frame, labels, identity_cache) for name, frame in frames.items()}
    assert np.array_equal(values['P16'], values['P01'])
    assert abs(values['P01'].mean() - .2722727272727273) < 1e-14
    comparisons = {name: paired_summary(values[name], values['C12']) for name in ['P01', 'P16', 'P15']}
    new_truths = int(((values['P16'] == 0) & (values['C12'] > 0)).sum())
    lost_truths = int(((values['P16'] > 0) & (values['C12'] == 0)).sum())
    selected = comparisons['P16']['delta'] > 1e-12 and new_truths > 0
    final.to_csv(OUT / 'development.csv', index=False); actions.to_csv(OUT / 'development-actions.csv', index=False)
    pd.DataFrame(dict(molecule_id=labels.index, **values)).to_csv(OUT / 'development-predictions.csv', index=False)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), queries=40, candidate='P16_plus_deep_prefix',
        fixed_config_sha256=sha(CONFIG), source_contract_sha256=sha(OUT / 'source-contract.json'),
        development_protocol_sha256=sha(OUT / 'development-protocol.json'),
        scoring_receipt_sha256=sha(out / 'scoring-receipt.json'), comparisons=comparisons,
        baseline_truths=int((values['P16'] > 0).sum()), candidate_truths=int((values['C12'] > 0).sum()),
        absent_truths_recovered=new_truths, truths_lost=lost_truths,
        actions={str(k): int(v) for k, v in actions.reason.value_counts().items()},
        total_insertions=int(actions.inserted.sum()), maximum_row_insertions=int(actions.inserted.max()),
        new_candidate_selected=bool(selected), controls={name: paired_summary(values['P01'], values[name]) for name in ['P14', 'P15']},
        negative_control_interpretation='P14 has the wrong development sign relative to its actual public regression; P15 direction agrees but uncertainty reaches zero. Source/CE shift limits public-score calibration.',
        truth_absent_mrr=0, reserved_prediction_metrics_computed=False, automatic_submission=False, submission_qualified=False,
        limitation='Already-used positive-only CASMI2022 development; a positive point estimate alone cannot qualify a trial. Freeze before120 metrics, then require supported incremental benefit versusP01/P16 plus real Kaggle smoke and independent audits.')
    write(OUT / 'development.json', report)
    write(OUT / 'selection-freeze.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        selected_candidate='P16_plus_deep_prefix' if selected else 'P16', new_candidate_selected=bool(selected),
        config_sha256=sha(CONFIG), source_contract_sha256=sha(OUT / 'source-contract.json'),
        development_report_sha256=sha(OUT / 'development.json'), evaluator_sha256=sha(__file__),
        reserved_prediction_metrics_computed=False, automatic_submission=False,
        next_action='If selected, validate the frozen complete recipe on120 reserved plus real Kaggle smoke/independent audits. Otherwise preserve the unused reserve; never submit an unqualified variant.'))
    print('DONE C12 DEVELOPMENT', json.dumps(report), flush=True)


if __name__ == '__main__': main()
