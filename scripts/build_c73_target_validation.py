"""Package the unchanged frozen dual-T4 core for all512fresh target queries."""
import json
import pandas as pd
import build_c41_validation as base
from build_general_anchor import guarded_source
from build_c09_controls import unwrap
from c45_complete_parent import ROOT,read,write,sha,stamp
OUT=ROOT/'outputs/local/c73-clean-target-validation-20261112'
PANEL=ROOT/'outputs/local/c72-clean-target-holdout-20261112'
QUERY=ROOT/'outputs/publish/c72-clean-target-holdout-20261112/validation-queries.parquet'
FAMILY=ROOT/'outputs/local/c71-final-clean-direct-pair-20261112'


def main():
    assert not (OUT/'validation-build.json').exists()
    assert not (PANEL/'evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    family=read(FAMILY/'complete-pipeline-binding.json')
    for name,pin in family['files_sha256'].items():assert sha(ROOT/name)==pin
    panel=read(PANEL/'validation-freeze.json');native=read(PANEL/'independent-acquisition-audit.json')
    assert panel['complete_pipeline_binding_sha256']==sha(FAMILY/'complete-pipeline-binding.json')
    assert native['queries']==512 and native['acquisitions']==2215 and native['all_usable_acquisitions_retained']
    assert native['all512official_structure_identities_verified'] and native['all_original_metadata_and_untrimmed_peak_arrays_exact']
    assert native['all_USED_and_global231reserved_identity_formula_scaffold_and_near_chemistry_disjoint']
    fit_alias=read(PANEL/'fitted-query-alias-audit.json')
    assert fit_alias['actual_fitting_query_acquisitions']==3220
    assert fit_alias['full_peak_or_top96_alias_overlap_with_actual_fitting_acquisitions']==[0,0]
    ref=read(OUT/'reference-exclusion-freeze.json')
    for name,pin in ref['files_sha256'].items():assert sha(OUT/name)==pin
    q=pd.read_parquet(QUERY);assert len(q)==2215 and q.molecule_id.nunique()==512
    assert not {'key','smiles','normalized_smiles','formula','inchikey14'}&set(q.columns)
    base.OUT=OUT;base.C39=PANEL;base.QUERIES=QUERY;base.PREFIX='c73'
    helper=(ROOT/'scripts/c41_reference_guard.py').read_text(encoding='utf8');compat=(ROOT/'scripts/legacy_formate_compat.py').read_text(encoding='utf8')
    pin=base.build('full',list(dict.fromkeys(q.molecule_id)),helper,compat)
    folder=ROOT/'kaggle/forks/c73-full-masked-validation';package=ROOT/'outputs/publish/c73-full-masked-validation-20261111'
    oldcontract=sha(package/'validation-contract.json')
    (package/'ATTRIBUTION.txt').write_text('Private query-only controlled target validation:512newEnveda180molecules/2215usable original acquisitions. All usable source acquisitions and actual353positive/124mixed/35negativequerymodes retained. Source: https://github.com/enveda/enveda-180 and official competition training data https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra .\n'
        'New C71CPU/GPU complete scientific recipes were frozen before selection and any outcomes. Global reference masks preserve every prior exclusion and add ALL USED/fresh/global231reserved identity/formula/scaffold/near-chemistry and acquisition aliases. No structure labels or per-query answer mappings mounted.\n'
        'Controlled in-corpus target experiment only. Public parent and encoder historical exposure persists. This is not independent-laboratory or public-score calibration. Original source attribution and raw-row provenance remain in the private local ledger.\n',encoding='utf8')
    contract=read(package/'validation-contract.json');contract['files']['ATTRIBUTION.txt']=sha(package/'ATTRIBUTION.txt')
    contract.update(whole_raw_acquisitions_claimed=True,complete_pipeline_binding_sha256=sha(FAMILY/'complete-pipeline-binding.json'),new_source_quality_read=False)
    write(package/'validation-contract.json',contract);newcontract=sha(package/'validation-contract.json')
    metadata=read(package/'dataset-metadata.json');metadata['title']='CASMI26 C73 new clean target validation';write(package/'dataset-metadata.json',metadata)
    nb=read(folder/'notebook.ipynb')
    for i in [6,27]:
        source=unwrap(''.join(nb['cells'][i]['source']));assert source.count(oldcontract)==1
        nb['cells'][i]['source']=guarded_source(source.replace(oldcontract,newcontract),'c73_clean_target_controls_'+str(i)).splitlines(keepends=True)
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),'c73_'+str(i),'exec')
    (folder/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')
    prior=read(ROOT/'kaggle/forks/c57-full-masked-validation/validation-pin.json')
    assert pin['frozen_scientific_payload_sha256']==prior['frozen_scientific_payload_sha256']==family['P01_frozen_scientific_payload_sha256']
    assert pin['guard_source_sha256']==prior['guard_source_sha256'] and pin['compat_source_sha256']==prior['compat_source_sha256']
    pin.update(notebook_sha256=sha(folder/'notebook.ipynb'),query_contract_sha256=newcontract,wrapper_source_sha256=sha(__file__),
        same_scientific_payload_as_completed_C57=True,complete_pipeline_binding_sha256=sha(FAMILY/'complete-pipeline-binding.json'))
    write(folder/'validation-pin.json',pin)
    write(OUT/'validation-build.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),
        dataset=metadata['id'],kernel=read(folder/'kernel-metadata.json')['id'],queries=512,acquisitions=2215,
        notebook_sha256=pin['notebook_sha256'],query_contract_sha256=newcontract,complete_pipeline_binding_sha256=sha(FAMILY/'complete-pipeline-binding.json'),
        same_original_R3_scientific_payload=True,reference_masks_required=True,labels_mounted=False,new_source_outcomes_read=False,C13_read=False,
        scored_entry=False,models_fitted_in_kernel=False))
    print('C73 FRESH512QUERY/2215ACQUISITION NOTEBOOK FROZEN; independent reference audit required before release',flush=True)


if __name__=='__main__':main()
