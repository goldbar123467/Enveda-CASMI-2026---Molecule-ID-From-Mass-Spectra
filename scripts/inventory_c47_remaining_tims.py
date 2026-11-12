"""Outcome-blind census of all remaining primary timsTOF chemistry.

C39 is now used development data. Never redefine it as fresh source evidence.
This inventory does not fit a model, select by performance or open C13 labels.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
from collections import Counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,sha,read,write


def main():
    RDLogger.DisableLog('rdApp.*');out=ROOT/'outputs/local/c47-remaining-tims-inventory-20261112';assert not out.exists();out.mkdir()
    inventory=ROOT/'outputs/local/c38-source-support-20261111/massbank-latest-tims-inventory.parquet'
    c39=ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv'
    c13=ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet'
    used_paths=[ROOT/p for p in ['outputs/publish/c08-validation-20261108/labels.csv','outputs/publish/c09-validation-20261109/labels.csv',
        'outputs/local/c10-casmi2022-20261110/labels.csv','outputs/local/c20-negative-development-20261110/labels.csv']]
    files=[inventory,c39,c13]+used_paths
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        rule='Census ALL valid primary timsTOF identities not in C39, prior used panels or the entire global C13 identity inventory. Also exclude same formula, same nonempty Murcko scaffold or Morgan2 Tanimoto>=.70 to any now-used C39 query. No sampling or performance-based selection.',
        purpose='Determine whether an unmatched chemistry holdout exists before committing another CPU/GPU fit.',
        C39_is_now_used=True,C13_labels_read=False,predictions_read=False,new_model_training=False,
        limitation='Remaining chemistry from the SAME MassBank2026.11 archive, not a new laboratory, independent source domain or universally unseen-by-pretraining sample.')
    write(out/'protocol.json',protocol)
    labels=pd.read_csv(c39,usecols=['key','smiles','formula']);used=set(labels.key)|set(pd.read_parquet(c13,columns=['key']).key)
    for path in used_paths:used.update(pd.read_csv(path,usecols=['key']).key)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    query_mol=[Chem.MolFromSmiles(s) for s in labels.smiles];assert all(m is not None for m in query_mol)
    query_fp=[gen.GetFingerprint(m) for m in query_mol];forms=set(labels.formula)
    scaffolds={MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in query_mol};scaffolds.discard('')
    raw=pd.read_parquet(inventory);counts=Counter();records=[]
    for key,group in raw.groupby('metric_key',sort=True):
        if not group.valid_identity.all():counts['invalid_identity']+=1;continue
        if key in used or group.in_consumed_panel.any():counts['used_or_reserved_identity']+=1;continue
        molecule=Chem.MolFromSmiles(group.smiles.iloc[0])
        if molecule is None:counts['unparseable_structure']+=1;continue
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        formula=CalcMolFormula(molecule);scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=molecule,includeChirality=False)
        maximum=max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(molecule),query_fp))
        if formula in forms:counts['C39_formula']+=1;continue
        if scaffold and scaffold in scaffolds:counts['C39_nonempty_scaffold']+=1;continue
        if maximum>=.7:counts['C39_similarity']+=1;continue
        counts['inventory_survivor']+=1
        records.append(dict(key=key,smiles=group.smiles.iloc[0],formula=formula,scaffold=scaffold,
            maximum_C39_tanimoto=float(maximum),available_primary_records=len(group)))
    pd.DataFrame(records).to_parquet(out/'remaining-chemistry.parquet',index=False)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/'protocol.json'),
        primary_records=len(raw),primary_identities=raw.metric_key.nunique(),outcomes=dict(counts),
        remaining_identities=len(records),remaining_primary_records=sum(r['available_primary_records'] for r in records),
        remaining_chemistry_sha256=sha(out/'remaining-chemistry.parquet'),C13_labels_read=False,predictions_read=False,
        qualified_new_panel=False,acquisition_export_and_fit_reference_masks_required=True,source_domain_independent=False)
    write(out/'inventory.json',result);print('C47 ALL REMAINING TIMS INVENTORY',result)


if __name__=='__main__':main()
