"""Independent retained-reference chemistry, alias closure and exact SID audit."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,write,sha,stamp,module
OUT=ROOT/'outputs/local/c73-clean-target-validation-20261112'


def main():
    target=OUT/'independent-reference-mask-audit.json';assert not target.exists()
    RDLogger.DisableLog('rdApp.*');pin=read(OUT/'reference-exclusion-freeze.json')
    assert pin['source_sha256']==sha(ROOT/'scripts/prepare_c73_reference_masks.py')
    for name,value in pin['files_sha256'].items():assert sha(OUT/name)==value
    classification=read(OUT/'chemistry-classification-freeze.json')
    for name,value in classification['files_sha256'].items():assert sha(ROOT/name)==value
    witness=pd.read_parquet(OUT/'original-structure-identity-witnesses.parquet')
    raw=set(read(OUT/'engine2-raw-key-exclusion.json')['keys']);mask=np.load(OUT/'main-reference-sid-exclusion.npy')
    assert raw==set(witness.loc[witness.exclude,'raw_key'])
    assert not witness.groupby('raw_key').exclude.nunique().gt(1).any()
    assert not witness[witness.official_key.notna()].groupby('official_key').exclude.nunique().gt(1).any()
    old=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
    assert set(read(old/'engine2-raw-key-exclusion.json')['keys'])<=raw
    assert set(read(ROOT/'outputs/local/c72-clean-target-holdout-20261112/additional-raw-alias-quarantine.json')['keys'])<=raw
    official=set(witness.loc[witness.exclude,'official_key'].dropna())
    reference=ROOT/'outputs/research/c08-pipeline/reference/train_structs.parquet';assert sha(reference)==pin['reference_structure_sha256']
    st=pd.read_parquet(reference);lookup=dict(zip(witness.smiles,witness.official_key))
    expected=np.array([str(r.inchikey14) in raw or pd.isna(lookup.get(str(r.smiles))) or lookup.get(str(r.smiles)) in official for r in st.itertuples(index=False)])
    np.testing.assert_array_equal(mask,expected)
    previous=np.load(ROOT/'outputs/local/c57-target-validation-20261112/main-reference-sid-exclusion.npy');assert mask[previous].all()
    held=pd.read_parquet(OUT/'global-exclusion-chemistry.parquet');keys=set(held.key);forms=set(held.formula)
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);mols=[Chem.MolFromSmiles(s) for s in held.smiles]
    fps=[generator.GetFingerprint(m) for m in mols];scaffolds={MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in mols};scaffolds.discard('')
    canonical=pd.read_parquet(ROOT/'outputs/local/long-frozen-20261107-r1/molecules.parquet').drop_duplicates('key').set_index('key')[['smiles','formula']].to_dict('index')
    allowed=witness[~witness.exclude].drop_duplicates('official_key');metric=module('c73_independent_extra_identity','c09e_features.py').structure;maximum=0.
    for i,row in enumerate(allowed.itertuples(index=False),1):
        assert row.official_key not in keys
        d=canonical.get(row.official_key)
        if d is None:
            key,text,formula,*_=metric(row.smiles);assert key==row.official_key;d=dict(smiles=text,formula=formula)
        assert d['formula'] not in forms
        m=Chem.MolFromSmiles(d['smiles']);sc=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
        assert not(sc and sc in scaffolds)
        similarity=max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(m),fps));assert similarity<.7;maximum=max(maximum,similarity)
        if i%50000==0:print('C73 INDEPENDENT ALLOWED CHEMISTRY',i,'/',len(allowed),flush=True)
    retained=[lookup[str(s)] for s in st.loc[~mask,'smiles']]
    assert not set(retained)&(keys|official)
    write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),reference_exclusion_freeze_sha256=sha(OUT/'reference-exclusion-freeze.json'),
        main_SIDs=len(mask),excluded_main_SIDs=int(mask.sum()),retained_main_SIDs=int((~mask).sum()),excluded_raw_keys=len(raw),
        independently_checked_allowed_official_identities=len(allowed),maximum_retained_to_global_exclusion_Tanimoto=maximum,
        every_raw_and_official_alias_component_uniformly_quarantined=True,exact_main_SID_mask_reconstructed=True,
        all_prior_masks_and_fresh_acquisition_aliases_preserved=True,retained_global_excluded_identity_formula_scaffold_or_near_chemistry=0,
        query_or_candidate_removal=0,per_query_answer_map_mounted=False,quality_outcomes_read=False,C13_read=False,passed=True))
    print('C73 INDEPENDENT REFERENCE MASK AUDIT PASS',len(allowed),'allowed identities',flush=True)


if __name__=='__main__':main()
