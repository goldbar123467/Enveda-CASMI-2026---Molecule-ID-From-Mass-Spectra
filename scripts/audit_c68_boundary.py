"""Independent all-row check of reserved chemistry quarantine and fit masks."""
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,sha


def audit_boundary(q,rows):
    folder=ROOT/'outputs/local/c68-reserved-fit-boundary-20261112'
    freeze=read(folder/'boundary-freeze.json')
    assert freeze['source_sha256']==sha(ROOT/'scripts/prepare_c68_reserved_boundary.py')
    for name,pin in freeze['files_sha256'].items():assert sha(ROOT/name)==pin
    reserved=set(pd.read_parquet(ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet',columns=['key']).key)
    catalogue=pd.read_parquet(folder/'global-exclusion-chemistry.parquet');assert set(catalogue.key)==reserved and len(reserved)==231
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    molecules=[Chem.MolFromSmiles(s) for s in catalogue.smiles];assert all(m is not None for m in molecules)
    fps=[generator.GetFingerprint(m) for m in molecules];forms=set(catalogue.formula)
    scaffolds={MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in molecules};scaffolds.discard('')
    unique=pd.concat([q[['key','smiles','formula']],rows[['key','smiles','formula']]]).drop_duplicates(['key','smiles','formula'])
    bad={}
    for r in unique.itertuples(index=False):
        m=Chem.MolFromSmiles(r.smiles);assert m is not None
        sc=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
        blocked=r.key in reserved or r.formula in forms or bool(sc and sc in scaffolds) or max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(m),fps))>=.7
        bad[(r.key,r.smiles,r.formula)]=blocked
    qbad=np.array([bad[(r.key,r.smiles,r.formula)] for r in q.itertuples(index=False)])
    groups=set(q.loc[qbad,'chemistry_group']);qeligible=~q.chemistry_group.isin(groups)
    np.testing.assert_array_equal(qeligible,q.fitting_eligible)
    rbad=np.array([bad[(r.key,r.smiles,r.formula)] for r in rows.itertuples(index=False)])
    expected=~rbad&rows.molecule_id.map(pd.Series(qeligible.to_numpy(),index=q.molecule_id)).to_numpy(bool)
    np.testing.assert_array_equal(expected,rows.fitting_eligible)
    assert not set(rows.loc[expected,'key'])&reserved
    assert not rows.loc[expected,'reserved_connection'].notna().any()
    assert qeligible.sum()==freeze['fitting_queries'] and expected.sum()==freeze['fitting_candidate_rows']
    return dict(global_reserved_identities=231,resolved_global_chemistry=231,independent_all_row_identity_formula_scaffold_similarity_masks_exact=True,
        fitting_queries=int(qeligible.sum()),fitting_candidate_rows=int(expected.sum()),fitting_reserved_identity_or_chemistry_overlap=0,
        evaluated_queries=len(q),evaluated_candidate_rows=len(rows),C13_query_mapping_or_outcome_read=False)
