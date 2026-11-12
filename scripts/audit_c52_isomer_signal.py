"""Describe the failed USED ranking signal before authorizing another fit."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from c45_complete_parent import ROOT,read,write,sha,stamp
OUT=ROOT/'outputs/local/c52-unused-target-inventory-20261112'


def main():
    RDLogger.DisableLog('rdApp.*');target=OUT/'used-isomer-signal-audit.json';assert not target.exists()
    fit=ROOT/'outputs/local/c48-clean-residual-heads-20261112';cache=ROOT/'outputs/local/long-frozen-20261107-r1'
    ids=np.load(fit/'fitting-identities.npy');catalogue=pd.read_parquet(cache/'molecules.parquet').iloc[ids]
    counts=catalogue.groupby('formula').size();eligible=catalogue.formula.map(counts).ge(2)
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    records=[]
    for study,origin,parent_path,answer_path,labels_path in [
        ('C39_USED',fit/'C39-predictions',ROOT/'outputs/local/c45-complete-parent-validation-20261112/P16.csv',fit/'C39-predictions/cpu.csv',ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv'),
        ('C51_NOW_USED',ROOT/'outputs/local/c51-complete-auxiliary-20261112/cpu-predictions',ROOT/'outputs/local/c51-complete-auxiliary-20261112/P16.csv',ROOT/'outputs/local/c51-complete-auxiliary-20261112/cpu-predictions/cpu.csv',ROOT/'outputs/local/c48-auxiliary-holdout-20261112/labels.csv')]:
        identity=pd.read_parquet(origin/'candidate-identities.parquet').set_index('smiles');scores=pd.read_parquet(origin/'candidate-scores.parquet')
        scores=scores[scores.trial.eq('cpu')].set_index(['molecule_id','smiles'])
        names=pd.read_parquet(origin/'encoded-molecule-names.parquet').canonical.tolist();vectors=np.load(origin/'encoded-molecule-vectors.npy',mmap_mode='r')
        center=np.load(fit/'fit-scaling.npz')['molecule_center'];lookup={s:i for i,s in enumerate(names)}
        before=pd.read_csv(parent_path,dtype=str).set_index('molecule_id');after=pd.read_csv(answer_path,dtype=str).set_index('molecule_id');labels=pd.read_csv(labels_path,dtype=str).set_index('molecule_id')
        for mid,text in before.smiles.items():
            old=text.split(';');new=after.loc[mid,'smiles'].split(';');truth=labels.loc[mid,'key']
            true=[s for s in old if identity.loc[s,'key']==truth]
            if not true:continue
            true=true[0];i,j=old.index(true),new.index(true)
            if i==j:continue
            for other in old:
                if other==true or (old.index(other)<i)==(new.index(other)<j):continue
                ct=identity.loc[true,'canonical'];co=identity.loc[other,'canonical']
                a=np.asarray(vectors[lookup[ct]],np.float64)-center;b=np.asarray(vectors[lookup[co]],np.float64)-center
                cosine=float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
                fpa=generator.GetFingerprint(Chem.MolFromSmiles(ct));fpb=generator.GetFingerprint(Chem.MolFromSmiles(co))
                records.append(dict(study=study,molecule_id=mid,true_parent_rank=i+1,true_new_rank=j+1,harmed=j>i,
                    other_parent_rank=old.index(other)+1,same_formula=identity.loc[true,'formula']==identity.loc[other,'formula'],
                    truth_key=truth,other_key=identity.loc[other,'key'],centered_MoLFormer_target_cosine=cosine,
                    Morgan2_2048_Tanimoto=float(DataStructs.TanimotoSimilarity(fpa,fpb)),fingerprints_differ=fpa!=fpb,
                    CPU_score_margin_truth_minus_other=float(scores.loc[(mid,true),'score']-scores.loc[(mid,other),'score'])))
    pd.DataFrame(records).to_parquet(OUT/'used-isomer-pairs.parquet',index=False)
    assert all(r['same_formula'] for r in records)
    write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),used_pair_records=len(records),
        harmed_pairs=sum(r['harmed'] for r in records),pairs_distinguishable_by_Morgan2=sum(r['fingerprints_differ'] for r in records),
        same_formula_fitting_identities=int(eligible.sum()),total_fitting_identities=len(ids),same_formula_fitting_groups=int(counts.ge(2).sum()),
        pair_evidence_sha256=sha(OUT/'used-isomer-pairs.parquet'),records=records,
        conclusion='Both directions of used-panel changes involve same-formula pairs. Audit whether structural fingerprints distinguish these pairs, and whether enough training formula groups exist for explicit isomer-negative supervision.',
        statistical_claim=False,new_validation_quality_read=False,C13_read=False,
        limitation='Descriptive adaptive-development diagnostics. Target cosine or fingerprint distinguishability alone does not establish generalization or authorize promotion.'))
    print('C52 ISOMER SIGNAL AUDIT',len(records),'pairs; Morgan distinguishable',sum(r['fingerprints_differ'] for r in records),'same-formula fit IDs',int(eligible.sum()),flush=True)


if __name__=='__main__':main()
