"""Two-parent paired summaries and a scaffold-cluster bootstrap sensitivity check.

This does not create a fresh or scaffold-disjoint test set. It resamples existing
target families and keeps the two fitted parents together for each molecule.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'

import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from tail_fusion import paired_summary

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/professional'


def rr(rank):
    rank=np.asarray(rank,dtype=float)
    return np.where(rank<=25,1/rank,0)


def clustered_interval(delta,labels):
    labels=np.asarray(labels)
    unique,inverse=np.unique(labels,return_inverse=True)
    counts=np.bincount(inverse)
    sums=np.bincount(inverse,weights=np.asarray(delta,dtype=float))
    rng=np.random.default_rng(20261105)
    draws=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=5000)
    values=(draws@sums)/(draws@counts)
    return dict(clusters=len(unique),largest_cluster=int(counts.max()),
                ci95=np.quantile(values,[.025,.975]).tolist(),bootstrap_replicates=5000,
                method='resample Bemis-Murcko target clusters; weight members equally within the resulting sample')


def main():
    mol=pd.read_parquet(ROOT/'outputs/local/n0/cache/molecules.parquet',columns=['key','smiles']).set_index('key')
    def scaffold(key):
        graph=Chem.MolFromSmiles(str(mol.loc[key,'smiles']))
        assert graph is not None
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(graph),isomericSmiles=False) or 'ACYCLIC'
    native=[];panels=[]
    for seed in [20,21]:
        frame=pd.read_parquet(OUT/f'native_cont{seed}.parquet')
        general=frame[frame.model.eq(f'general_cont{seed}')].set_index('key').sort_index()
        enriched=frame[frame.model.eq(f'np_cont{seed}')].set_index('key').reindex(general.index)
        assert len(general)==len(enriched)==234 and enriched.familiar_rank.notna().all()
        native.append((general,enriched))
        panels.append(pd.read_parquet(OUT/f'physical_cont{seed}.parquet').set_index('molecule_id').sort_index())
    assert native[0][0].index.equals(native[1][0].index)
    assert panels[0].index.equals(panels[1].index)
    keys=native[0][0].index.to_numpy();labels=np.array([scaffold(k) for k in keys])
    report=dict(parents=2,independent_targets_not_468=True,fresh_test=False,competition_mrr=False,
                caveat='Conditional on two fitted parent pairs and reused target cohorts. Scaffold resampling is a sensitivity analysis; training/evaluation scaffolds may overlap.',native={})
    for name,col,mask in [('familiar','familiar_rank',np.ones(len(keys),bool)),('unseen','unseen_rank',np.ones(len(keys),bool)),
                          ('hard185','unseen_rank',native[0][0].unseen_candidates.ge(8).to_numpy())]:
        base=np.mean([rr(g[col].to_numpy()) for g,n in native],axis=0)[mask]
        candidate=np.mean([rr(n[col].to_numpy()) for g,n in native],axis=0)[mask]
        report['native'][name]=dict(paired=paired_summary(base,candidate),scaffold_sensitivity=clustered_interval(candidate-base,labels[mask]))
    panel_keys=np.array([str(k).removeprefix('holdout_') for k in panels[0].index])
    panel_labels=np.array([scaffold(k) for k in panel_keys])
    general=np.mean([f[f'general_cont{s}'].to_numpy() for s,f in zip([20,21],panels)],axis=0)
    candidate=np.mean([f[f'np_cont{s}'].to_numpy() for s,f in zip([20,21],panels)],axis=0)
    baseline=panels[0].baseline.to_numpy()
    report['panel_vs_ordinary']=dict(paired=paired_summary(general,candidate),scaffold_sensitivity=clustered_interval(candidate-general,panel_labels))
    report['panel_vs_s3']=dict(paired=paired_summary(baseline,candidate),scaffold_sensitivity=clustered_interval(candidate-baseline,panel_labels))
    (OUT/'continuation_two_parent_sensitivity.json').write_text(json.dumps(report,indent=2))
    for name,item in list(report['native'].items())+[(k,report[k]) for k in ['panel_vs_ordinary','panel_vs_s3']]:
        print(name,'delta',round(item['paired']['delta'],6),'moleculeCI',item['paired']['ci95'],'scaffoldCI',item['scaffold_sensitivity']['ci95'],'clusters',item['scaffold_sensitivity']['clusters'])


if __name__=='__main__':
    main()
