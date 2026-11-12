"""Synthetic-only exercise of the exposure report before reserved consumption."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import c61_exposure as exposure


class ExposureTests(unittest.TestCase):
    def test_original_raw_alias_and_prior_identity_are_reported_without_filtering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=root/'model';model.mkdir();out=root/'report';out.mkdir();cache=root/'cache'
            np.save(model/'fitting-identities.npy',np.array([0]));np.save(model/'fitting-spectrum-rows.npy',np.array([0]))
            (model/'fit-boundary-freeze.json').write_text(json.dumps(dict(files_sha256={n:exposure.sha(model/n) for n in ['fitting-identities.npy','fitting-spectrum-rows.npy']})),encoding='utf8')
            labels=pd.DataFrame(dict(molecule_id=['a','b','c'],key=['ethanol','pyridine','benzene'],smiles=['CCO','c1ccncc1','c1ccccc1'],formula=['C2H6O','C5H5N','C6H6'])).set_index('molecule_id')
            def parquet(path,*args,**kwargs):
                name=Path(path).name
                if name=='molecules.parquet':return pd.DataFrame(dict(key=['ethanol'],smiles=['CCO'],formula=['C2H6O']))
                if name=='selected.parquet':return pd.DataFrame(dict(row=[0]))
                if name=='train.parquet':return pd.DataFrame(dict(inchikey14=['raw-alias']))
                if name=='original-structure-identity-witnesses.parquet':return pd.DataFrame(dict(raw_key=['raw-alias'],official_key=['pyridine']))
                raise AssertionError(str(path))
            prior=pd.DataFrame(dict(key=['pyridine'],smiles=['c1ccncc1'],formula=['C5H5N']))
            with patch.object(exposure,'ROOT',root),patch.object(exposure,'C40',model),patch.object(exposure,'CACHE',cache),patch.object(exposure,'OUT',out),patch.object(exposure.pd,'read_parquet',side_effect=parquet),patch.object(exposure.pd,'read_csv',return_value=prior):
                groups,result=exposure.chemistry_and_fit_exposure(labels)
            self.assertEqual(result['new_head_identity_overlaps'],2)
            self.assertEqual(result['prior_used_identity_overlaps'],1)
            self.assertEqual(result['original_raw_identity_groups_checked'],1)
            self.assertEqual(len(groups),3)
            report=pd.read_csv(out/'fitting-exposure.csv')
            self.assertEqual(report.molecule_id.tolist(),['a','b','c'])
            self.assertEqual(report.fit_identity_overlap.tolist(),[True,True,False])
            self.assertEqual(report.prior_used_identity_overlap.tolist(),[False,True,False])


if __name__=='__main__':unittest.main()
