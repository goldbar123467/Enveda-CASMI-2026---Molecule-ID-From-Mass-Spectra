"""Describe first-choice errors on the same already-used240 panel; CPU only."""
import functools
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from research_score_plateau import ROOT, BASE, OUT, sha


def main():
    assert (ROOT / '.local/compute-paused.json').exists()
    assert not (OUT / 'used240-isomer-errors.json').exists()
    prior = json.loads((OUT / 'used240-postmortem.json').read_text())
    assert prior['all240_labels_previously_consumed'] and not prior['reserved_labels_read']
    for path, pin in prior['inputs_sha256'].items():
        assert sha(ROOT / path) == pin, path
    parent = pd.read_csv(BASE / 'parent.csv', dtype=str)
    ranks = pd.read_csv(OUT / 'used240-rank-breakdown.csv', dtype={'molecule_id':str}).set_index('molecule_id')
    queries = pd.read_parquet(BASE / 'queries.parquet').drop_duplicates('molecule_id').set_index('molecule_id')

    @functools.lru_cache(maxsize=1200)
    def formula(smiles):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        return rdMolDescriptors.CalcMolFormula(mol)

    errors = []
    for row in parent.itertuples(index=False):
        rank = int(ranks.loc[row.molecule_id, 'baseline_rank'])
        if rank <= 1:
            continue
        candidates = row.smiles.split(';')
        errors.append(dict(mode=str(queries.loc[row.molecule_id, 'ionization_mode']),
            same_formula=formula(candidates[0]) == formula(candidates[rank-1])))
    groups = {}
    for name, subset in [('all', errors), ('positive', [r for r in errors if r['mode']=='positive']),
                         ('negative', [r for r in errors if r['mode']=='negative'])]:
        groups[name] = dict(present_truth_below_first=len(subset), same_formula_wrong_first=sum(r['same_formula'] for r in subset))
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        postmortem_sha256=sha(OUT / 'used240-postmortem.json'), rank_breakdown_sha256=sha(OUT / 'used240-rank-breakdown.csv'),
        rdkit_version=rdBase.rdkitVersion, groups=groups, reserved_labels_read=False, training_performed=False,
        limitation='Formula equality of submitted SMILES on used development, not a hidden-test error distribution. Excludes missing truths and already-correct first choices; this does not establish that every wrong first choice can be fixed by one fragmentation model.')
    (OUT / 'used240-isomer-errors.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('DONE USED240 FORMULA ERROR AUDIT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
