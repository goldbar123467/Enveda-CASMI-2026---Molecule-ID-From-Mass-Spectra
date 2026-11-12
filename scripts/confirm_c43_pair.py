"""One frozen paired C13 confirmation, only after both C39 admissions pass.

Preparation and scoring read query-only inputs/predictions. Only the explicit
evaluate command consumes labels; a durable global marker prevents another test.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,json,subprocess
from fractions import Fraction
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from precision_cache import ROOT,sha,write,features
from validation_repair import clustered_summary

OUT=ROOT/'outputs/local/c43-paired-confirmation-20261111'
C13=ROOT/'outputs/local/c13-confirmation-20261110'
CLOUD=ROOT/'outputs/forks/c13-mixed-casmi2022-validation-v2'
HEAD=ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111'
POLICY=ROOT/'configs/c43-paired-confirmation-20261111.json'
C39=ROOT/'outputs/local/c42-fixed-c39-validation-20261111'
C40=ROOT/'outputs/local/c40r1-crossmodal-heads-20261111'
CACHE=ROOT/'outputs/local/long-frozen-20261107-r1'
CONSUMPTION=C13/'evaluation-consumption.json'


def stamp():return datetime.now(timezone.utc).isoformat()


def authorize():
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']


def prior():
    authorize();quality=json.loads((C39/'quality-results.json').read_text())
    assert all(quality['results'][t]['admission']['passed'] for t in ['cpu','gpu']),'Both fixed recipes must qualify before C13 access'
    independent=json.loads((C39/'independent-quality-audit.json').read_text())
    assert independent['passed_pair'] and independent['quality_results_sha256']==sha(C39/'quality-results.json')
    assert independent['source_sha256']==sha(ROOT/'scripts/audit_c42_quality.py')
    assert not quality['C13_read'] and quality['all_queries_retained'] and quality['missing_truths_score_zero']
    freeze=json.loads((C39/'evaluation-freeze.json').read_text())
    assert freeze['source_sha256']==sha(ROOT/'scripts/evaluate_c42_fixed.py')
    assert quality['evaluation_freeze_sha256']==sha(C39/'evaluation-freeze.json')
    return quality


def prepare():
    prior();assert not CONSUMPTION.exists();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'preparation.json').exists()
    audited=json.loads((C13/'execution-v2-audit.json').read_text())
    assert audited['queries']==120 and audited['baseline_replay_exact'] and not audited['fresh_prediction_label_metrics_read']
    for n,h in audited['artifact_sha256'].items():assert sha(CLOUD/n)==h,n
    assert sha(CLOUD/'c13_p01.csv')==audited['baseline_sha256']
    import shutil
    for dest,source in [('P01.csv',CLOUD/'c13_p01.csv'),('P14.csv',CLOUD/'c13_p14.csv'),('queries.parquet',CLOUD/'c13_queries.parquet')]:shutil.copy2(source,OUT/dest)
    q=pd.read_parquet(OUT/'queries.parquet');original=pd.read_parquet(ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet')
    pd.testing.assert_frame_equal(q[original.columns],original,check_exact=True)
    assert len(q)==q.molecule_id.nunique()==120 and q.ionization_mode.value_counts().to_dict()=={'positive':60,'negative':60}
    assert not {'smiles','key','formula','inchikey14','mass'}&set(q.columns)
    production={}
    for trial,slot in [('cpu','p20-c42-cpu-alignment'),('gpu','p21-c42-gpu-alignment')]:
        folder=ROOT/'kaggle/forks'/slot;pin=json.loads((folder/'production-pin.json').read_text())
        assert sha(folder/'notebook.ipynb')==pin['notebook_sha256'] and pin['head_manifest_sha256']==sha(HEAD/'head-manifest.json')
        production[trial]=dict(path=str(folder.relative_to(ROOT)),notebook_sha256=pin['notebook_sha256'],pin_sha256=sha(folder/'production-pin.json'))
    preparation=dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),
        C39_quality_sha256=sha(C39/'quality-results.json'),head_manifest_sha256=sha(HEAD/'head-manifest.json'),
        C13_input_audit_sha256=sha(C13/'execution-v2-audit.json'),C13_preparation_sha256=sha(C13/'preparation-report.json'),
        production=production,files_sha256={n:sha(OUT/n) for n in ['P01.csv','P14.csv','queries.parquet']},
        quality_labels_read=False,query_count=120,paired_frozen_recipes=True)
    write(OUT/'preparation.json',preparation);print('C43 PREPARED WITHOUT LABELS',flush=True)


def verify():
    prior();p=json.loads((OUT/'preparation.json').read_text())
    assert p['source_sha256']==sha(__file__) and p['policy_sha256']==sha(POLICY)
    assert p['C39_quality_sha256']==sha(C39/'quality-results.json') and p['head_manifest_sha256']==sha(HEAD/'head-manifest.json')
    for n,h in p['files_sha256'].items():assert sha(OUT/n)==h,n
    for row in p['production'].values():
        assert sha(ROOT/row['path']/'notebook.ipynb')==row['notebook_sha256']
        assert sha(ROOT/row['path']/'production-pin.json')==row['pin_sha256']
    return p


def score():
    verify();assert not CONSUMPTION.exists() and not (OUT/'predictions').exists()
    subprocess.run([str(ROOT/'.venv-encoders/Scripts/python.exe'),str(HEAD/'score_c42_crossmodal.py'),str(HEAD),
        str(ROOT/'outputs/publish/frozen-p14-20261107'),str(OUT/'P01.csv'),str(OUT/'queries.parquet'),str(OUT/'predictions')],cwd=ROOT,check=True)
    subprocess.run([str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/audit_c42_ranking.py',str(OUT/'predictions'),str(OUT/'P01.csv')],cwd=ROOT,check=True)
    write(OUT/'prediction-freeze.json',dict(frozen_utc=stamp(),preparation_sha256=sha(OUT/'preparation.json'),
        policy_sha256=sha(POLICY),source_sha256=sha(__file__),score_receipt_sha256=sha(OUT/'predictions/score-receipt.json'),
        ranking_audit_sha256=sha(OUT/'predictions/independent-ranking-audit.json'),quality_labels_read=False))
    print('C43 BOTH PREDICTIONS FROZEN; LABELS UNREAD',flush=True)


def chemistry_and_fit_exposure(labels):
    """Report every query's neighbor exposure; never filter a favorable cohort."""
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    querymol=[Chem.MolFromSmiles(s) for s in labels.smiles];assert all(m is not None for m in querymol)
    fp=[gen.GetFingerprint(m) for m in querymol]
    scaffold=[MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in querymol]
    roots=list(range(len(labels)))
    def find(i):
        while roots[i]!=i:roots[i]=roots[roots[i]];i=roots[i]
        return i
    for i in range(len(labels)):
        for j in range(i):
            if labels.key.iloc[i]==labels.key.iloc[j] or labels.formula.iloc[i]==labels.formula.iloc[j] or (scaffold[i] and scaffold[i]==scaffold[j]) or DataStructs.TanimotoSimilarity(fp[i],fp[j])>=.7:roots[find(i)]=find(j)
    groups=np.asarray([find(i) for i in range(len(labels))])
    ids=np.load(C40/'fitting-identities.npy',allow_pickle=False);rows=np.load(C40/'fitting-spectrum-rows.npy',allow_pickle=False)
    boundary=json.loads((C40/'fit-boundary-freeze.json').read_text())
    for n in ['fitting-identities.npy','fitting-spectrum-rows.npy']:assert sha(C40/n)==boundary['files_sha256'][n]
    mol=pd.read_parquet(CACHE/'molecules.parquet').iloc[ids]
    selected=pd.read_parquet(CACHE/'selected.parquet',columns=['row']);actual_rows=selected.row.to_numpy()[rows]
    raw=pd.read_parquet(ROOT/'outputs/competition-raw/train.parquet',columns=['inchikey14'])
    actual_raw_keys=set(raw.inchikey14.iloc[actual_rows].astype(str));del raw,selected
    witnesses=pd.read_parquet(ROOT/'outputs/local/c41r1-masked-pipeline-validation-20261111/original-structure-identity-witnesses.parquet',columns=['raw_key','official_key'])
    fitted_original_official=set(witnesses.loc[witnesses.raw_key.isin(actual_raw_keys),'official_key'].dropna())
    overlaps=labels.key.isin(set(mol.key)|fitted_original_official).to_numpy()
    prior_labels=pd.read_csv(ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv')
    prior_identity=labels.key.isin(set(prior_labels.key)).to_numpy();prior_maximum=np.zeros(len(labels))
    prior_formula=np.zeros(len(labels),np.int64);prior_scaffold=np.zeros(len(labels),np.int64)
    for row in prior_labels.itertuples(index=False):
        m=Chem.MolFromSmiles(row.smiles);assert m is not None
        prior_maximum=np.maximum(prior_maximum,DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),fp))
        prior_formula+=np.asarray([row.formula==v for v in labels.formula])
        sc=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
        if sc:prior_scaffold+=np.asarray([sc==v for v in scaffold])
    maximum=np.zeros(len(labels));sameformula=np.zeros(len(labels),np.int64);samescaffold=np.zeros(len(labels),np.int64)
    for count,row in enumerate(mol.itertuples(index=False),1):
        m=Chem.MolFromSmiles(row.smiles);assert m is not None
        maximum=np.maximum(maximum,DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),fp))
        sameformula+=np.asarray([row.formula==v for v in labels.formula])
        sc=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
        if sc:samescaffold+=np.asarray([sc==v for v in scaffold])
        if count%25000==0:authorize();print('C43 FIT EXPOSURE CHECK',count,'/',len(mol),flush=True)
    exposure=pd.DataFrame(dict(molecule_id=labels.index,fit_identity_overlap=overlaps,maximum_fit_tanimoto=maximum,
        fitting_same_formula=sameformula,fitting_same_nonempty_scaffold=samescaffold,prior_C39_identity_overlap=prior_identity,
        prior_C39_maximum_tanimoto=prior_maximum,prior_C39_same_formula=prior_formula,prior_C39_same_nonempty_scaffold=prior_scaffold,group=groups))
    exposure.to_csv(OUT/'fitting-exposure.csv',index=False)
    return groups,dict(new_head_identity_overlaps=int(overlaps.sum()),queries_with_fitting_tanimoto_at_least_point7=int((maximum>=.7).sum()),
        prior_C39_identity_overlaps=int(prior_identity.sum()),queries_with_prior_C39_tanimoto_at_least_point7=int((prior_maximum>=.7).sum()),
        queries_with_prior_C39_same_formula=int((prior_formula>0).sum()),queries_with_prior_C39_same_nonempty_scaffold=int((prior_scaffold>0).sum()),
        queries_with_fitting_same_formula=int((sameformula>0).sum()),queries_with_fitting_same_nonempty_scaffold=int((samescaffold>0).sum()),
        original_raw_identity_groups_checked=len(actual_raw_keys),query_groups=len(set(groups)),
        near_chemistry_excluded_from_all_training=False,universal_pretraining_exclusion=False)


def evaluate(consume,resume=False):
    assert consume,'Explicit --consume is required; no background worker opens C13'
    import msvcrt
    lock=(C13/'paired-consumption.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    verify();assert not (OUT/'confirmation-results.json').exists()
    frozen=json.loads((OUT/'prediction-freeze.json').read_text())
    assert frozen['source_sha256']==sha(__file__) and frozen['preparation_sha256']==sha(OUT/'preparation.json') and frozen['policy_sha256']==sha(POLICY)
    assert frozen['score_receipt_sha256']==sha(OUT/'predictions/score-receipt.json')
    receipt=json.loads((OUT/'predictions/score-receipt.json').read_text())
    for n,h in receipt['files_sha256'].items():assert sha(OUT/'predictions'/n)==h,n
    assert frozen['ranking_audit_sha256']==sha(OUT/'predictions/independent-ranking-audit.json')
    prep=json.loads((C13/'preparation-report.json').read_text());assert sha(C13/'labels.csv')==prep['labels_sha256']
    commitment=dict(cycle='C43_C42R1_PAIRED',source_sha256=sha(__file__),policy_sha256=sha(POLICY),prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'))
    if CONSUMPTION.exists():
        old=json.loads(CONSUMPTION.read_text());assert resume and all(old.get(k)==v for k,v in commitment.items()),'Reserved test already consumed or different evaluation'
    else:
        assert not resume;write(CONSUMPTION,dict(started_utc=stamp(),state='one_joint_evaluation_started',**commitment))
    # This is the sole hypothesis evaluation. No fitting/selection/cohort
    # revisions follow these labels; exact crash recovery requires the same seal.
    RDLogger.DisableLog('rdApp.*');baseline=pd.read_csv(OUT/'P01.csv',dtype=str,keep_default_na=False);mids=baseline.molecule_id.tolist()
    labels=pd.read_csv(C13/'labels.csv',dtype=str).set_index('molecule_id').loc[mids]
    assert len(labels)==120 and labels['mode'].value_counts().to_dict()=={'positive':60,'negative':60}
    queries=pd.read_parquet(OUT/'queries.parquet')
    actual_modes=queries.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids]
    assert actual_modes.tolist()==labels['mode'].tolist() and actual_modes.value_counts().to_dict()=={'positive':60,'negative':60}
    frames={'P01':baseline,'P14':pd.read_csv(OUT/'P14.csv',dtype=str,keep_default_na=False)}
    for t in ['cpu','gpu']:frames[t]=pd.read_csv(OUT/'predictions'/(t+'.csv'),dtype=str,keep_default_na=False)
    identities=pd.read_parquet(OUT/'predictions/candidate-identities.parquet');keys=dict(zip(identities.smiles,identities.key))
    for f in frames.values():
        assert f.molecule_id.tolist()==mids
        for text in f.smiles:
            for s in text.split(';'):
                if s not in keys:keys[s]=features.structure(s)[0]
    rr={};coverage={};ranks={}
    for name,f in frames.items():
        values=[];present=[];positions=[]
        for text,truth in zip(f.smiles,labels.key):
            kk=[keys[s] for s in text.split(';')];assert 1<=len(kk)<=25 and len(set(kk))==len(kk)
            rank=kk.index(truth)+1 if truth in kk else 0
            positions.append(rank);values.append(1./rank if rank else 0.);present.append(truth in kk)
        rr[name]=np.asarray(values);coverage[name]=np.asarray(present);ranks[name]=positions
    np.savez(OUT/'confirmation-vectors.npz',**rr)
    groups,exposure=chemistry_and_fit_exposure(labels);results={}
    for name in ['cpu','gpu','P14']:
        summarize=lambda mask:clustered_summary(rr['P01'][mask],rr[name][mask],groups[mask],10000,20261143)
        item=dict(all120=summarize(np.ones(120,bool)),mode={mode:summarize(labels['mode'].eq(mode).to_numpy()) for mode in ['positive','negative']},
            recovered_truths=int((~coverage['P01']&coverage[name]).sum()),displaced_truths=int((coverage['P01']&~coverage[name]).sum()),truth_absent_queries=int((~coverage[name]).sum()))
        if name in ['cpu','gpu']:
            assert np.array_equal(coverage['P01'],coverage[name])
            negative=labels['mode'].eq('negative').to_numpy();assert frames[name].loc[negative].equals(baseline.loc[negative])
            reasons=[]
            def exact_delta(mask):
                return sum(((Fraction(1,ranks[name][i]) if ranks[name][i] else Fraction(0))-(Fraction(1,ranks['P01'][i]) if ranks['P01'][i] else Fraction(0)) for i in np.flatnonzero(mask)),Fraction(0))/int(mask.sum())
            exact={k:exact_delta(mask) for k,mask in [('all120',np.ones(120,bool)),('positive60',labels['mode'].eq('positive').to_numpy())]}
            item['exact_point_deltas']={k:dict(numerator=v.numerator,denominator=v.denominator,value=float(v)) for k,v in exact.items()}
            if any(v<0 for v in exact.values()):reasons.append('predeclared_confirmation_point_regression')
            if exposure['new_head_identity_overlaps']>0:reasons.append('new_head_fitted_identity_exposure')
            if exposure['prior_C39_identity_overlaps']>0:reasons.append('prior_C39_identity_exposure')
            item['admission']=dict(passed=not reasons,reasons=reasons,statistical_promotion=False)
        results[name]=item
    record=dict(completed_utc=stamp(),commitment=commitment,queries=120,actual_query_modes=actual_modes.value_counts().to_dict(),results=results,fitting_exposure=exposure,
        one_joint_reserved_evaluation=True,no_recipe_or_cohort_selection=True,all_absent_truths_zero=True,
        leaderboard_calibration_established=False,statistical_promotion=False,automatic_submission=False,
        limitation='C13 is a single-acquisition FT/HCD source stress panel, not a target timsTOF population proxy. Fitting chemistry neighbors and public pretraining exposure remain disclosed. Uncertain intervals do not prove public transfer.')
    write(OUT/'confirmation-results.json',record)
    consumption=json.loads(CONSUMPTION.read_text());consumption.update(state='completed',completed_utc=stamp(),results_sha256=sha(OUT/'confirmation-results.json'));write(CONSUMPTION,consumption)
    print('C43 PAIRED CONFIRMATION',json.dumps({t:results[t] for t in results}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score','evaluate']);p.add_argument('--consume',action='store_true');p.add_argument('--resume-same-evaluation',action='store_true');a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='score':score()
    else:evaluate(a.consume,a.resume_same_evaluation)
