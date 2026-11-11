"""CPU-only, receipt-bound replay and boundary attribution of consumed panels.

Never executes notebook top-level code, model calls, or reserved C13 data.
Only the executed notebook's pure identity/fusion/overlay functions are extracted.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import ast
import base64
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from prepare_c36_validation import ROOT, OUT as C36, sha, write, verify

OUT = ROOT / 'outputs/local/c37-selection-trace-20261111'
PANELS = {'c10': 'c10-casmi2022-validation', 'c20': 'c20-generated-recovery'}


def extract_executed(prefix):
    path = ROOT / 'kaggle/forks' / PANELS[prefix] / 'notebook.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    wrapper = ''.join(notebook['cells'][12]['source'])
    values = [n.value for n in ast.walk(ast.parse(wrapper)) if isinstance(n, ast.Constant)
              and isinstance(n.value, str) and 'FROZEN_B64 =' in n.value]
    source = max(values, key=len)
    constants = {}
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in ['FROZEN_B64', 'FROZEN_SHA256']:
                constants[n.targets[0].id] = ast.literal_eval(n.value)
    payload = base64.b64decode(constants['FROZEN_B64'])
    receipt = json.loads((OUT / (prefix+'-stage-artifacts/execution_receipt.json')).read_text())
    assert hashlib.sha256(payload).hexdigest() == constants['FROZEN_SHA256'] == receipt['source_sha256']
    frozen = json.loads(payload)
    code = '\n'.join(''.join(c['source']) for c in frozen['cells'] if c['cell_type'] == 'code')
    return code, dict(notebook_path=str(path.relative_to(ROOT)), notebook_sha256=sha(path),
                      executed_frozen_sha256=constants['FROZEN_SHA256'], receipt_sha256=sha(OUT/(prefix+'-stage-artifacts/execution_receipt.json')))


class Identity:
    def __init__(self, contract):
        known = pd.read_parquet(C36/'identity-cache.parquet')
        self.keys = dict(zip(known.smiles, known.metric_key))
        for name in ['deep_index', 'positive_features', 'negative_features', 'stress_features']:
            f = pd.read_parquet(ROOT/contract['inputs'][name]['path'], columns=['smiles', 'metric_key'])
            self.keys.update(zip(f.smiles, f.metric_key))
        self.path = OUT/'extended-identity-cache.json'
        self.extra = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.keys.update(self.extra)
        self.memo = {}; self.enumerator = rdMolStandardize.TautomerEnumerator()

    def __call__(self, s):
        s = str(s).strip()
        if s in self.memo: return self.memo[s]
        mol = Chem.MolFromSmiles(s)
        if mol is None or not mol.GetNumAtoms(): self.memo[s] = None; return None
        heavy = tuple(sorted(Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() != 1).items()))
        if s not in self.keys:
            Chem.RemoveStereochemistry(mol)
            canonical = self.enumerator.Canonicalize(mol)
            k = Chem.MolToInchiKey(canonical)[:14] or None
            self.keys[s] = k; self.extra[s] = k
            if len(self.extra) % 100 == 0:
                write(self.path, self.extra)
                print('C37 IDENTITIES', len(self.extra), flush=True)
        k = self.keys[s]
        self.memo[s] = (k, heavy) if k else None
        return self.memo[s]

    def save(self): write(self.path, self.extra)


def load_csv(path):
    f = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert f.molecule_id.is_unique and list(f.columns) == ['molecule_id', 'smiles']
    return {mid: s.split(';') for mid, s in f.itertuples(index=False)}


def truth_rank(items, target, identity):
    ids = [identity(s) for s in items]
    return ids.index(target)+1 if target in ids else 0


def main():
    pa.set_cpu_count(2); RDLogger.DisableLog('rdApp.*')
    assert rdBase.rdkitVersion == '2026.03.3'
    assert (ROOT/'.local/compute-paused.json').is_file()
    assert not (OUT/'selection-trace.json').exists(), 'Preserve prior trace'
    contract, _ = verify()
    identity = Identity(contract)
    metadata = pd.read_csv(C36/'queries.csv', dtype={'group': str})
    metadata = metadata[metadata.cohort.eq('primary')].set_index('molecule_id')
    old = pd.read_csv(C36/'candidate-loss-witnesses.csv').set_index('molecule_id')
    parent = load_csv(ROOT/contract['inputs']['parent']['path'])
    frozen_sources = {}; pin = {}; artifacts = {}
    for prefix in PANELS:
        code, pin[prefix] = extract_executed(prefix); frozen_sources[prefix] = code
        d = OUT/(prefix+'-stage-artifacts')
        names = ['e004_candidate_inputs.json', 'e004_verification.json', 'experiment_audit_factorial.csv',
                 'experiment_audit_factorial.json', 'rank2_overlay.json', 'v43_execution_receipt.json',
                 'submission_reference.csv', 'submission_e004_control.csv', 'submission_e006_control.csv',
                 'submission_baseline0465.csv', 'variants/submission_CUSTOM.csv']
        artifacts[prefix] = {n: sha(d/n) for n in names}
        receipt = json.loads((d/'v43_execution_receipt.json').read_text())
        assert artifacts[prefix]['submission_baseline0465.csv'] == receipt['submission']['sha256']
        e004 = json.loads((d/'e004_verification.json').read_text())
        assert artifacts[prefix]['e004_candidate_inputs.json'] == e004['hashes']['e004_candidate_inputs.json']
    write(OUT/'trace-input-contract.json', dict(scripts={Path(__file__).name: sha(__file__)},
          c36_contract_sha256=sha(C36/'input-contract.json'), executed_sources=pin, artifacts=artifacts,
          reserved_labels_read=False, inference=False, gpu_work=False))
    records = []; replays = {}; membership = {}; examples = {}
    for prefix, code in frozen_sources.items():
        d = OUT/(prefix+'-stage-artifacts'); data = json.loads((d/'e004_candidate_inputs.json').read_text())
        tree = ast.parse(code)
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ['metric_fusion', '_rank2_overlay']]
        assert len(nodes) == 2
        scope = {}; exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pure_executed_selectors', 'exec'), scope)
        stages = {n: load_csv(d/file) for n, file in [('reference', 'submission_reference.csv'),
                  ('e004', 'submission_e004_control.csv'), ('e006', 'submission_e006_control.csv'),
                  ('frozen_rank2', 'submission_baseline0465.csv'), ('custom', 'variants/submission_CUSTOM.csv')]}
        assert set(stages['e006']) == set(data['prefusion'])
        # Execute the exact saved rank-2 function in a private local directory.
        replay_dir = OUT/('overlay-replay-'+prefix); replay_dir.mkdir(exist_ok=True)
        (replay_dir/'submission.csv').write_bytes((d/'submission_e006_control.csv').read_bytes())
        previous = Path.cwd()
        try:
            os.chdir(replay_dir)
            replay = scope['_rank2_overlay']({'metric_identity': identity, 'ENG': data['engine2']})
        finally: os.chdir(previous)
        replayed = load_csv(replay_dir/'submission.csv')
        assert replayed == stages['frozen_rank2'], 'Actual executed overlay replay differs'
        original_report = json.loads((d/'rank2_overlay.json').read_text())
        assert replay['inserted'] == original_report['inserted']
        replays[prefix] = dict(rows=replay['rows'], inserted=replay['inserted'], exact_smiles_list_parity=True,
                               function_sha256=hashlib.sha256(ast.get_source_segment(code, nodes[1]).encode()).hexdigest())
        audit = pd.read_csv(d/'experiment_audit_factorial.csv').set_index('molecule_id')
        diffs = Counter()
        for count, mid in enumerate(stages['e006'], 1):
            r = metadata.loc[mid]; target = identity(r.truth_smiles)
            assert target[0] == r.truth
            first = [s for s in data['prefusion'][mid] if s and s != 'CCO']
            fused, scores, collapsed = scope['metric_fusion'](first, data['engine2'][mid].get('smiles', []), identity)
            assert len(fused) == int(audit.loc[mid, 'candidate_count']), (mid, len(fused), audit.loc[mid, 'candidate_count'])
            assert collapsed == int(audit.loc[mid, 'collapsed_source_votes']), mid
            lists = {**{n: rows[mid] for n, rows in stages.items()}, 'parent': parent[mid],
                     'prefusion': data['prefusion'][mid], 'engine2': data['engine2'][mid].get('smiles', []),
                     'pubchem': data['pubchem'].get(mid, {}).get('pc', []), 'base_tail': data['base_tail'].get(mid, []),
                     'fusion_overflow': data['fusion_overflow'].get(mid, []), 'metric_fusion40': fused}
            ranks = {name: truth_rank(items, target, identity) for name, items in lists.items()}
            keysets = {name: {identity(s) for s in items} for name, items in lists.items()}
            for name in ['e004', 'e006', 'frozen_rank2', 'custom', 'parent']:
                assert None not in keysets[name] and len(keysets[name]) == len(lists[name]) <= 25
            diffs['frozen_to_custom_membership_rows'] += keysets['frozen_rank2'] != keysets['custom']
            diffs['custom_to_parent_membership_rows'] += keysets['custom'] != keysets['parent']
            diffs['overlay_membership_rows'] += keysets['e006'] != keysets['frozen_rank2']
            diffs['overlay_truths_added'] += bool(ranks['frozen_rank2'] and not ranks['e006'])
            diffs['overlay_truths_lost'] += bool(ranks['e006'] and not ranks['frozen_rank2'])
            expanded = set().union(*(keysets[n] for n in ['base_tail', 'prefusion', 'engine2', 'pubchem', 'fusion_overflow', 'parent']))
            dropped = old.loc[mid, 'loss'] == 'retrieved_but_final_dropped'
            cause = 'retained_final' if ranks['parent'] else 'outside_observed_routes'
            if not ranks['parent']:
                if ranks['frozen_rank2']: cause = 'postfrozen_membership_change'
                elif ranks['e006']: cause = 'forced_engine2_rank2_evicted_rank25'
                elif ranks['metric_fusion40']: cause = 'same_formula_'+str(audit.loc[mid, 'branch'])+'_final25_boundary'
                elif ranks['prefusion'] or ranks['engine2']: cause = 'reciprocal_fusion_40_boundary'
                elif ranks['base_tail']: cause = 'main_prefusion_truncation_or_merge'
                elif ranks['pubchem']: cause = 'pubchem_merge_gate_or_budget'
                elif ranks['fusion_overflow'] or bool(old.loc[mid, 'retained_union_truth']): cause = 'earlier_scoring_input_outside_final_source_lists'
            record = dict(molecule_id=mid, mode=r['mode'], group=r.group, historical_dropped26=bool(dropped),
                          historical_loss=old.loc[mid, 'loss'], expanded_truth_observed=target in expanded,
                          e006_branch=str(audit.loc[mid, 'branch']), last_observed_boundary=cause)
            record.update({name+'_truth_rank': rank for name, rank in ranks.items()})
            if ranks['metric_fusion40']:
                j = ranks['metric_fusion40']-1
                from rdkit.Chem import rdMolDescriptors
                form = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(fused[j]))
                record['fusion40_truth_formula_slots'] = [i+1 for i,s in enumerate(fused) if rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s)) == form]
            records.append(record)
            if count % 40 == 0: print('C37 TRACED', prefix, count, flush=True); identity.save()
        membership[prefix] = dict(diffs)
    identity.save()
    f = pd.DataFrame(records); assert len(f) == 240 and f.historical_dropped26.sum() == 26
    f.to_csv(OUT/'selection-boundary-witnesses.csv', index=False)
    selected = f[f.historical_dropped26]
    f[f.historical_dropped26].to_csv(OUT/'dropped26-boundary-witnesses.csv', index=False)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), input_contract_sha256=sha(OUT/'trace-input-contract.json'),
                  witnesses_sha256=sha(OUT/'selection-boundary-witnesses.csv'), overlay_replay=replays,
                  postfrozen_membership_audit=membership,
                  historical_dropped26_boundary_counts=selected.last_observed_boundary.value_counts().to_dict(),
                  historical_dropped26_by_mode={mode: g.last_observed_boundary.value_counts().to_dict() for mode,g in selected.groupby('mode')},
                  expanded_truth_coverage=int(f.expanded_truth_observed.sum()), final_truth_coverage=int(f.parent_truth_rank.gt(0).sum()),
                  newly_observed_from_former_absent64=int((f.historical_loss.eq('absent_from_retained_routes') & f.expanded_truth_observed).sum()),
                  all_queries_boundary_counts=f.last_observed_boundary.value_counts().to_dict(), reserved_labels_read=False,
                  inference=False, gpu_work=False,
                  limitations=['Final25 boundary is exact observed membership, not an invented internal physical-score rank.',
                              'Historical 26 denominator retained; expanded intermediate routes are reported separately.',
                              'A selection policy causing a loss is not by itself an implementation defect.'])
    write(OUT/'selection-trace.json', report)
    print('DONE C37 TRACE', json.dumps(report), flush=True)


if __name__ == '__main__': main()
