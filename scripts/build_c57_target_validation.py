"""Build the SAME R3 masked inference for all35auxiliary queries; no labels."""
import copy,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import build_c41_validation as base
from build_general_anchor import guarded_source
from build_c09_controls import unwrap
from c45_complete_parent import ROOT,read,write,sha
HEAD=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
PANEL=ROOT/'outputs/local/c53-target-holdout-20261112'
QUERY=ROOT/'outputs/publish/c53-target-holdout-20261112/validation-queries.parquet'
OUT=ROOT/'outputs/local/c57-target-validation-20261112'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'validation-build.json').exists()
    boundary=read(HEAD/'fit-boundary-freeze.json');family=read(ROOT/'outputs/local/c56-bounded-semantic-heads-20261112/complete-candidate-freeze.json')
    development=read(ROOT/'outputs/local/c56-bounded-semantic-heads-20261112/used-auxiliary-results.json');assert development['combined_used_development_pair_passed']
    assert not family['C53_quality_read'] and not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    for name,pin in boundary['files_sha256'].items():assert sha(HEAD/name)==pin
    q=pd.read_parquet(QUERY);assert len(q)==2125 and q.molecule_id.nunique()==482
    assert not {'key','smiles','normalized_smiles','formula','inchikey14'}&set(q.columns)
    panel=read(PANEL/'validation-freeze.json');assert panel['files_sha256'][str(QUERY.relative_to(ROOT))]==sha(QUERY)
    witnesses=pd.read_parquet(HEAD/'original-structure-identity-witnesses.parquet');raw=set(read(HEAD/'engine2-raw-key-exclusion.json')['keys'])
    official=set(witnesses.loc[witnesses.exclude,'official_key'].dropna());mapping=dict(zip(witnesses.smiles,witnesses.official_key))
    reference=ROOT/'outputs/research/c08-pipeline/reference/train_structs.parquet';st=pd.read_parquet(reference,columns=['sid','inchikey14','smiles'])
    assert len(st)==275810 and np.array_equal(st.sid.to_numpy(),np.arange(len(st)))
    mask=st.inchikey14.astype(str).isin(raw).to_numpy()|np.asarray([mapping.get(str(s)) is None or mapping.get(str(s)) in official for s in st.smiles])
    np.save(OUT/'main-reference-sid-exclusion.npy',mask);(OUT/'engine2-raw-key-exclusion.json').write_bytes((HEAD/'engine2-raw-key-exclusion.json').read_bytes())
    write(OUT/'reference-exclusion-freeze.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        fit_boundary_sha256=sha(HEAD/'fit-boundary-freeze.json'),panel_freeze_sha256=sha(PANEL/'validation-freeze.json'),reference_structure_sha256=sha(reference),
        reference_SIDs=len(st),excluded_main_SIDs=int(mask.sum()),excluded_raw_keys=len(raw),
        global_C39_and_auxiliary_chemistry_and_alias_masks=True,query_truth_map_mounted=False,
        files_sha256={name:sha(OUT/name) for name in ['main-reference-sid-exclusion.npy','engine2-raw-key-exclusion.json']}))
    base.OUT=OUT;base.C39=PANEL;base.QUERIES=QUERY;base.PREFIX='c57'
    helper=(ROOT/'scripts/c41_reference_guard.py').read_text(encoding='utf8');compat=(ROOT/'scripts/legacy_formate_compat.py').read_text(encoding='utf8')
    pin=base.build('full',list(dict.fromkeys(q.molecule_id)),helper,compat)
    folder=ROOT/'kaggle/forks/c57-full-masked-validation';package=ROOT/'outputs/publish/c57-full-masked-validation-20261111'
    oldcontract=sha(package/'validation-contract.json')
    (package/'ATTRIBUTION.txt').write_text('Private query-only controlled target-domain holdout:482molecules/2125usable original acquisitions;384Enveda180+98EnvedaNPexamples. All usable selected source acquisitions and actual mixed modes retained. No structure labels or per-query answer mapping mounted.\n'
        'Primary Enveda180 source: https://github.com/enveda/enveda-180 and official competition training data https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra . NP example source attribution is retained in the local original-row ledger.\n'
        'This is controlled in-corpus confirmation with identity/formula/scaffold/near-chemistry/raw-alias quarantine for newly fitted components. Public parent/encoder historical exposure persists. No independent-laboratory or universal spectral novelty claim.\n',encoding='utf8')
    contract=read(package/'validation-contract.json');contract['files']['ATTRIBUTION.txt']=sha(package/'ATTRIBUTION.txt')
    contract.update(auxiliary_only=False,positive_only=False,whole_raw_acquisitions_claimed=True,complete_candidate_freeze_sha256=sha(ROOT/'outputs/local/c56-bounded-semantic-heads-20261112/complete-candidate-freeze.json'))
    write(package/'validation-contract.json',contract);newcontract=sha(package/'validation-contract.json')
    nb=read(folder/'notebook.ipynb')
    for i in [0,6,27]:
        source=unwrap(''.join(nb['cells'][i]['source']))
        if i==0:
            assert oldcontract not in source  # Dependency setup has no data pin.
            continue
        assert source.count(oldcontract)==1
        source=source.replace(oldcontract,newcontract);nb['cells'][i]['source']=guarded_source(source,'c57_target_controls_'+str(i)).splitlines(keepends=True)
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),'c57_'+str(i),'exec')
    (folder/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')
    prior=read(ROOT/'kaggle/forks/c41r3-full-masked-validation/validation-pin.json')
    assert pin['frozen_scientific_payload_sha256']==prior['frozen_scientific_payload_sha256']
    assert pin['guard_source_sha256']==prior['guard_source_sha256'] and pin['compat_source_sha256']==prior['compat_source_sha256']
    pin.update(notebook_sha256=sha(folder/'notebook.ipynb'),query_contract_sha256=newcontract,wrapper_source_sha256=sha(__file__),
        same_scientific_payload_as_completed_C41R3=True,original_C39_study_replaced=False,auxiliary_only=False)
    write(folder/'validation-pin.json',pin)
    write(OUT/'validation-build.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        dataset='clarkkitchen/casmi26-c57-full-masked-validation-20261111',kernel='clarkkitchen/casmi26-c57-full-masked-validation',
        mode='all482controlled_target',queries=482,acquisitions=2125,notebook_sha256=pin['notebook_sha256'],query_contract_sha256=newcontract,
        same_original_R3_scientific_payload=True,reference_mask_required=True,labels_mounted=False,source_outcomes_read=False,C13_read=False,
        scored_entry=False,models_fitted_in_kernel=False))
    print('C57 FULL TARGET MASKED INFERENCE FROZEN',json.dumps(pin))


if __name__=='__main__':main()
