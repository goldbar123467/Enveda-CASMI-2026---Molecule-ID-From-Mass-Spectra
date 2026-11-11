"""Independent membership arithmetic and scalar identity spot checks for C37."""
import ast
import functools
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from trace_c37_selection import ROOT, OUT, C36, Identity, sha, write, verify, extract_executed, load_csv


def main():
    RDLogger.DisableLog('rdApp.*'); contract, _ = verify(); identity = Identity(contract)
    assert not (OUT/'independent-review.json').exists()
    tc = json.loads((OUT/'trace-input-contract.json').read_text())
    for name,pin in tc['scripts'].items(): assert sha(ROOT/'scripts'/name)==pin
    for prefix,files in tc['artifacts'].items():
        for name,pin in files.items(): assert sha(OUT/(prefix+'-stage-artifacts')/name)==pin
    code,_=extract_executed('c10')
    names={'canonicalize_smiles_to_ik14','heavy_atom_composition','metric_identity'}
    nodes=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    ns=dict(Chem=Chem,lru_cache=functools.lru_cache,_tautomer_canonicalizer=rdMolStandardize.TautomerEnumerator())
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_official_identity_helpers','exec'),ns)
    sample=sorted(identity.extra,key=lambda s:hashlib.sha256(('C37-independent-identity:'+s).encode()).hexdigest())[:64]
    assert len(sample)==64
    for s in sample: assert ns['metric_identity'](s)==identity(s)
    # Membership arithmetic independent of the rank-2 implementation.
    counts=Counter();metadata=pd.read_csv(C36/'queries.csv').set_index('molecule_id')
    parent=load_csv(ROOT/contract['inputs']['parent']['path'])
    for prefix in ['c10','c20']:
        d=OUT/(prefix+'-stage-artifacts');data=json.loads((d/'e004_candidate_inputs.json').read_text())
        before=load_csv(d/'submission_e006_control.csv');after=load_csv(d/'submission_baseline0465.csv')
        replay=load_csv(OUT/('overlay-replay-'+prefix)/'submission.csv');assert replay==after
        for mid in before:
            a=[identity(s) for s in before[mid]];b=[identity(s) for s in after[mid]];p=[identity(s) for s in parent[mid]]
            assert len(a)==len(set(a)) and len(b)==len(set(b)) and set(b)==set(p)
            lost=set(a)-set(b);added=set(b)-set(a)
            if lost:
                assert len(lost)==len(added)==1 and lost=={a[24]}
                assert added=={identity(data['engine2'][mid]['smiles'][0])}
                counts['overlay_displaced_candidates']+=1
            target=identity(metadata.loc[mid,'truth_smiles'])
            counts['overlay_true_answers_displaced']+=target in lost
            counts['overlay_true_answers_added']+=target in added
            counts['final_truths']+=target in set(p);counts['queries']+=1
    assert counts['queries']==240 and counts['final_truths']==150
    assert counts['overlay_displaced_candidates']==3 and counts['overlay_true_answers_displaced']==counts['overlay_true_answers_added']==0
    refinement=json.loads((OUT/'boundary-refinement.json').read_text())
    assert sum(refinement['counts'].values())==26
    assert sha(OUT/'refined26-boundary-witnesses.csv')==refinement['witness_sha256']
    formate=json.loads((OUT/'formate-compat-audit.json').read_text())
    assert formate['adapter_sha256']==sha(ROOT/'scripts/legacy_formate_compat.py')
    for name,pin in formate['patched_source_sha256'].items():assert sha(OUT/'prospective-formate-compat'/name)==pin
    pop=json.loads((OUT/'population-audit.json').read_text())
    target=pd.read_parquet(OUT/'unlabelled-target/test.parquet')
    modes=target.groupby('molecule_id').ionization_mode.agg(set)
    assert len(modes)==400 and int(modes.map(len).gt(1).sum())==97
    assert len(target)==1213 and set(target.instrument_type)=={'timsTOF'}
    assert not target.adduct.eq('[M+FA-H]-').any()
    assert pop['summaries']['target']['mode']['mixed']==97
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),reviewer_sha256=sha(__file__),
                reviewed_artifacts={n:sha(OUT/n) for n in ['selection-trace.json','trace-input-contract.json','boundary-refinement.json',
                    'formate-compat-audit.json','population-audit.json','prospective-validation-protocol.json']},
                actual_executed_identity_helper_sample_size=64,official_identity_cache_sample_exact=True,
                independent_membership_counts=dict(counts),all240_overlay_replay_exact=True,
                final_parent_membership_preserved=True,refined26_counts_reconciled=True,
                public_target_metadata_independently_recounted=True,
                formate_adapter_current_source_hashes_pass=True,
                old_c36_input_and_feature_seals_pass=True,
                quality_metrics_computed=False,reserved_holdout_opened=False,new_gpu_work=False,new_cloud_run=False,new_submission=False,
                decision='NO-GO for GPU work, retention-policy promotion or scoring submission. GO for the documented prospective validation/source-support design; formate scalar compatibility patch is prepared for future controlled integration.',
                limitations=['Independent arithmetic and source extraction in one CPU process, not a separate human or agent review.',
                             'Identity check is a fixed64-case spot check plus receipt-bound prior caches; it is not a new full-database identity audit.',
                             'No beneficial complete-pipeline replacement was frozen; fresh C13 was therefore not consumed.'])
    write(OUT/'independent-review.json',report);print('DONE C37 REVIEW',json.dumps(report),flush=True)


if __name__=='__main__':main()
