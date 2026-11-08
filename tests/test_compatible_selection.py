"""Reserved performance must not drive selection or trigger coefficient reselection."""
import contextlib,hashlib,io,json,shutil,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
try:
    import torch
    import pandas as pd
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import select_compatible_candidate as selector
except ImportError:
    selector=None


@unittest.skipUnless(selector is not None,'Numerical environment required')
class CompatibleSelectionTests(unittest.TestCase):
    def test_development_wins_despite_reserved_regression_and_freezes_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'outputs/local/long-compatible-20261108';run.mkdir(parents=True)
            package=root/'outputs/publish/c08-validation-20261108';package.mkdir(parents=True)
            validation=root/'outputs/forks/c08-external-validation';validation.mkdir(parents=True)
            bundle=root/'outputs/publish/c08-models-20261108';bundle.mkdir(parents=True)
            (bundle/'c08-manifest.json').write_text('{"test_metadata_only":true}')
            config=root/'configs/compatible-campaign-20261108.json';config.parent.mkdir()
            config.write_text(json.dumps(dict(candidate=dict(arms=['A','B'],weights=[.05,.1,.2],integration='tail only'))))
            ids=[f'm{i:03}' for i in range(128)];candidates=[f'S{i:02}' for i in range(1,26)]
            keys={s:f'K{i:013}' for i,s in enumerate(candidates,1)}
            labels=pd.DataFrame(dict(molecule_id=ids,partition=['development' if i%2==0 else 'confirmation' for i in range(128)],
                key=[keys['S20'] if i%2==0 else 'K9999999999999' if i%16==1 else keys['S10'] for i in range(128)]))
            labels.to_csv(package/'labels.csv',index=False)
            pd.DataFrame(dict(molecule_id=ids)).to_parquet(package/'validation-queries.parquet',index=False)
            contract=package/'validation-contract.json'
            contract.write_text(json.dumps(dict(files={name:selector.digest(package/name) for name in ['labels.csv','validation-queries.parquet']})))
            shutil.copy2(package/'validation-queries.parquet',validation/'c08_validation_queries.parquet')
            (validation/'c08_validation_report.json').write_text(json.dumps(dict(queries=128,truth_absent_mrr=0,truth_injection=False,validation_contract_sha256=selector.digest(contract))))
            baseline=pd.DataFrame(dict(molecule_id=ids,smiles=';'.join(candidates)));baseline.to_csv(validation/'c08_validation_p01.csv',index=False)
            pd.DataFrame(dict(smiles=candidates,metric_key=[keys[s] for s in candidates])).to_parquet(validation/'neural_candidate_features.parquet',index=False)
            pd.DataFrame(columns=['smiles','metric_key']).to_parquet(validation/'outsider_neural_features.parquet',index=False)
            def predict(bundle_path,arm,base,queries,path,**_):
                target='S20' if arm=='A' else 'S10';rows=[]
                for mid in ids:
                    for rank,s in enumerate(candidates,1):rows.append(dict(molecule_id=mid,smiles=s,baseline_rank=rank,c08_ok=True,c08_score=100 if s==target else -rank))
                frame=pd.DataFrame(rows);frame.to_parquet(path,index=False)
                receipt=dict(manifest_sha256=selector.digest(bundle/'c08-manifest.json'),arm=arm,steps=40000,smoke=False,input_errors=0,
                    queries_sha256=selector.digest(queries),baseline_content_sha256=hashlib.sha256(base.to_csv(index=False,lineterminator='\n').encode()).hexdigest())
                Path(str(path)+'.json').write_text(json.dumps(receipt));return frame,receipt
            arguments=['select','--validation',str(validation),'--bundle',str(bundle)]
            with patch.object(selector,'ROOT',root),patch.object(selector,'load_modules',return_value=(None,SimpleNamespace())),patch.object(selector,'score',side_effect=predict),patch.object(sys,'argv',arguments),contextlib.redirect_stdout(io.StringIO()):
                selector.main()
                frozen=(run/'qualification/selection.json').read_bytes();decision=json.loads(frozen)
                report=json.loads((run/'qualification/report.json').read_text())
                self.assertEqual(decision['winner']['arm'],'A');self.assertEqual(decision['winner']['weight'],.2)
                self.assertLess(report['confirmation']['delta'],0);self.assertEqual(report['truth_absent_queries'],8)
                selector.main()
                self.assertEqual(frozen,(run/'qualification/selection.json').read_bytes())


if __name__=='__main__':unittest.main()
