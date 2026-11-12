"""Synthetic collection/privacy/release tests and fixed-model arithmetic checks.

All account calls are fake. Real R1 model/source bytes may be read, but no
actual reserved query mapping, answers, GPU inference or scoring API is used.
"""
import copy
import hashlib
import json
import tempfile
import unittest
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from c75_port_utils import ROOT, read, write, sha
import c76_confirmation_inference as inference
import release_c76_confirmation as release
import collect_c76_confirmation as collection
import complete_c76_confirmation as complete
import audit_c76_confirmation_execution as execution


def fake_api_modules(api):
    module = types.ModuleType('kaggle.api.kaggle_api_extended'); module.KaggleApi = lambda: api
    return {'kaggle': types.ModuleType('kaggle'), 'kaggle.api': types.ModuleType('kaggle.api'),
        'kaggle.api.kaggle_api_extended': module,
        'msvcrt': SimpleNamespace(LK_NBLCK=1, locking=lambda *args: None)}


class ExecutionTests(unittest.TestCase):
    def test_private_push_intent_precedes_API_and_never_blindly_retries(self):
        for lost_response in [False, True]:
            with self.subTest(lost_response=lost_response), tempfile.TemporaryDirectory() as name, ExitStack() as stack:
                root=Path(name);out=root/'release';plan=root/'plan.json';write(plan,{'synthetic':True})
                folder=root/'kaggle/forks'/release.SLOTS['legacy'];folder.mkdir(parents=True)
                kernel='clarkkitchen/casmi26-'+release.SLOTS['legacy']
                write(folder/'notebook.ipynb',{'synthetic':True})
                write(folder/'kernel-metadata.json',dict(id=kernel,is_private=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaTeslaT4',dataset_sources=[release.DATASET]))
                write(folder/'validation-pin.json',dict(notebook_sha256=sha(folder/'notebook.ipynb'),metadata_sha256=sha(folder/'kernel-metadata.json')))
                class API:
                    calls=0
                    def authenticate(self):pass
                    def dataset_status(self,dataset):return 'ready'
                    def kernels_push(self,*args,**kwargs):
                        self.calls+=1
                        intent=read(out/'legacy-push-intent.json')
                        assert intent['notebook_sha256']==sha(folder/'notebook.ipynb') and intent['inference_plan_sha256']==sha(plan)
                        assert kwargs['acc']=='NvidiaTeslaT4'
                        if lost_response:raise ConnectionError('Synthetic lost private-run response')
                        return SimpleNamespace(version_number=1,error=None,url='https://www.kaggle.com/code/'+kernel,to_json=lambda:json.dumps({'synthetic':True}))
                api=API();stack.enter_context(patch.dict(sys.modules,fake_api_modules(api)))
                for attr,value in [('ROOT',root),('RELEASE',out),('PLAN',plan)]:stack.enter_context(patch.object(release,attr,value))
                stack.enter_context(patch.object(release,'verify_plan',return_value={}))
                if lost_response:
                    with self.assertRaises(ConnectionError):release.main('push-legacy')
                else:release.main('push-legacy')
                self.assertTrue((out/'legacy-push-intent.json').exists())
                with self.assertRaises(AssertionError):release.main('push-legacy')
                self.assertEqual(api.calls,1)

    def test_current_core_cannot_start_before_legacy_collection(self):
        class PriorHold(Exception):pass
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            class API:
                def authenticate(self):pass
            stack.enter_context(patch.dict(sys.modules,fake_api_modules(API())))
            stack.enter_context(patch.object(release,'RELEASE',Path(name)/'release'))
            stack.enter_context(patch.object(release,'verify_plan',return_value={}))
            stack.enter_context(patch.object(release,'collected',side_effect=PriorHold('Missing actual legacy core')))
            with self.assertRaises(PriorHold):release.main('push-c76')
            self.assertFalse((Path(name)/'release/c76-push-intent.json').exists())

    def test_dataset_publication_private_and_rejects_extra_label_file(self):
        for extra_label in [False,True]:
            with self.subTest(extra_label=extra_label),tempfile.TemporaryDirectory() as name,ExitStack() as stack:
                root=Path(name);package=root/'package';package.mkdir();out=root/'release';plan=root/'plan.json';write(plan,{'synthetic':True})
                names=['validation-queries.parquet','validation-sample.csv','main-reference-sid-exclusion.npy','engine2-raw-key-exclusion.json','ATTRIBUTION.txt']
                for filename in names:(package/filename).write_text('SYNTHETIC ONLY',encoding='utf8')
                write(package/'validation-contract.json',dict(files={filename:sha(package/filename) for filename in names}))
                write(package/'dataset-metadata.json',dict(id=release.DATASET))
                if extra_label:(package/'labels.csv').write_text('SYNTHETIC FORBIDDEN EXTRA',encoding='utf8')
                class API:
                    calls=0
                    def authenticate(self):pass
                    def dataset_create_new(self,*args,**kwargs):
                        self.calls+=1;assert kwargs['public'] is False and kwargs['convert_to_csv'] is False and kwargs['dir_mode']=='skip'
                        assert (out/'dataset-publish-intent.json').exists()
                        return SimpleNamespace(error=None,to_json=lambda:json.dumps({'synthetic':True}))
                api=API();stack.enter_context(patch.dict(sys.modules,fake_api_modules(api)))
                for attr,value in [('RELEASE',out),('PLAN',plan),('PACKAGE',package)]:stack.enter_context(patch.object(release,attr,value))
                stack.enter_context(patch.object(release,'verify_plan',return_value={}))
                if extra_label:
                    with self.assertRaises(AssertionError):release.main('publish')
                    self.assertEqual(api.calls,0);self.assertFalse((out/'dataset-publish-intent.json').exists())
                else:
                    release.main('publish');self.assertEqual(api.calls,1)

    def test_fixed_CPU_and_GPU_models_independent_scores_and_formula_slots(self):
        import numpy as np
        import pandas as pd
        ranker = complete.load('c76_execution_probe_ranker', inference.BUNDLE / 'c67_ranker.py')
        rows = pd.DataFrame(dict(molecule_id=['SYNTHETIC_A']*3 + ['SYNTHETIC_B']*3,
            formula=['isomers', 'other', 'isomers']*2, smiles=['CCO', 'c1ccccc1', 'COC']*2,
            parent_rank=[1, 2, 3]*2, available=[True, True, True, True, True, False]))
        x = np.random.default_rng(20261112).normal(size=(6, 101)).astype(np.float32)
        for trial in ['cpu', 'gpu']:
            with np.load(inference.BUNDLE / (trial+'-model.npz'), allow_pickle=False) as model:
                expected_lists, expected = ranker.predict(rows, x, model, gpu=trial == 'gpu')
                lists, scores = execution.independent_head(rows, x, model, trial, np)
            self.assertEqual(lists, expected_lists); np.testing.assert_allclose(scores, expected, atol=1e-10, rtol=1e-10)
            self.assertEqual(lists['SYNTHETIC_B'], ['CCO', 'c1ccccc1', 'COC'])
            self.assertEqual(lists['SYNTHETIC_A'][1], 'c1ccccc1')

    def test_same_formula_leader_can_change_and_missing_group_keeps_parent(self):
        import numpy as np
        import pandas as pd
        rows = pd.DataFrame(dict(molecule_id=['SYNTHETIC']*3, formula=['same', 'other', 'same'],
            smiles=['CCO', 'c1ccccc1', 'COC'], parent_rank=[1, 2, 3], available=True))
        x = np.zeros((3, 101)); x[2, 0] = 100
        model = dict(mean=np.zeros(101), std=np.ones(101), weights=np.eye(1, 101).reshape(-1))
        lists, _ = execution.independent_head(rows, x, model, 'cpu', np)
        self.assertEqual(lists['SYNTHETIC'], ['COC', 'c1ccccc1', 'CCO'])
        rows.loc[0, 'available'] = False
        lists, _ = execution.independent_head(rows, x, model, 'cpu', np)
        self.assertEqual(lists['SYNTHETIC'], ['CCO', 'c1ccccc1', 'COC'])
        rows.available = True; x[:] = 0; x[:, 0] = np.log(rows.parent_rank.to_numpy(float))
        lists, scores = execution.independent_head(rows, x, model, 'cpu', np)
        self.assertEqual(lists['SYNTHETIC'], ['CCO', 'c1ccccc1', 'COC']); np.testing.assert_array_equal(scores, np.zeros(3))

    def test_scalar_mass_retains_valid_and_reverts_unknown_or_inconsistent(self):
        import numpy as np
        import pandas as pd
        import types
        ranker = complete.load('c76_mass_probe_ranker', inference.BUNDLE / 'c67_ranker.py')
        compat = complete.load('c76_mass_probe_compat', inference.BUNDLE / 'legacy_formate_compat.py')
        chemistry = types.ModuleType('synthetic_mass_table')
        source = inference.BUNDLE / 'legacy_chem_source.py'
        exec(compile(compat.patch_chem_source(source.read_text(encoding='utf8')), str(source), 'exec'), chemistry.__dict__)
        def ion(mass, token):
            number, charge, delta, _ = chemistry.ADDUCTS[compat.normalize_legacy_adduct(token)]
            return (mass*number+delta)/charge
        ids = ['SYNTHETIC_VALID', 'SYNTHETIC_BAD', 'SYNTHETIC_UNKNOWN', 'SYNTHETIC_FORMATE']
        p01 = pd.DataFrame(dict(molecule_id=ids, smiles=['CCO;COC']*4)); p16 = p01.copy(); p16.smiles = 'COC;CCO'
        q = pd.DataFrame(dict(molecule_id=ids, precursor_mz=[ion(300., '[M+H]+'), ion(300., '[M+H]+'), 300., ion(300., '[M+FA-H]-')],
            adduct=['[M+H]+', '[M+H]+', '[M+FA-H]- ', '[M+FA-H]-']))
        refs = pd.DataFrame(dict(precursor_mz=[ion(m, '[M+H]+') for m in [300., 310., 300., 300.]], adduct=['[M+H]+']*4))
        hits = pd.DataFrame(dict(molecule_id=ids, view='all_reference', reference_row=range(4)))
        expected, _ = ranker.physical_parent(p01, p16, q, hits, refs, chemistry.neutral_mass)
        actual, counts = execution.independent_physical(p01, p16, q, hits, refs, chemistry.ADDUCTS, compat.normalize_legacy_adduct)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        self.assertEqual(actual.smiles.tolist(), ['COC;CCO', 'CCO;COC', 'CCO;COC', 'COC;CCO'])
        self.assertEqual(counts, dict(kept=2, reverted=2))

    def test_collector_accepts_exact_supported_IDs_and_rejects_data_or_health_drift(self):
        import numpy as np
        import pandas as pd
        ids = ['SYNTHETIC_' + str(i) for i in range(120)]
        q = pd.DataFrame(dict(molecule_id=ids, spectrum_id=['SYNTHETIC_SPEC_'+str(i) for i in range(120)],
            ionization_mode=['positive']*60 + ['negative']*60, precursor_mz=300.,
            adduct=['[M+H]+']*60 + ['[M-H]-']*59 + ['[M+FA-H]- '], collision_energy=float('nan'),
            ms2_mzs=[np.array([50., 100., 150.])]*120, ms2_normalized_intensities=[np.array([1., .5, .2])]*120))
        for corruption in [None, 'missing_PC', 'wrong_PC', 'trimmed_token', 'missing_acquisition', 'neural_fallback', 'GL_failure', 'wrong_device']:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory(prefix='c76-synthetic-core-') as name, ExitStack() as stack:
                root = Path(name); cloud = root/'cloud'; cloud.mkdir(); plan = root/'plan.json'; write(plan, {'synthetic': True})
                folder = root/'kaggle/forks'/collection.SLOTS['c76']; folder.mkdir(parents=True)
                write(folder/'notebook.ipynb', {'synthetic': True}); write(folder/'validation-pin.json', {'query_contract_sha256': 'contract'})
                source = root/'outputs/local/c37-selection-trace-20261111/pinned-public-source'; source.mkdir(parents=True)
                for filename in ['chem.py', 'frag.py']:
                    (source/filename).write_bytes((ROOT/'outputs/local/c37-selection-trace-20261111/pinned-public-source'/filename).read_bytes())
                from legacy_formate_compat import patch_chem_source, patch_frag_source
                before = {n: sha(source/n) for n in ['chem.py', 'frag.py']}
                after = {n: hashlib.sha256(fn((source/n).read_text(encoding='utf8')).encode()).hexdigest()
                    for n, fn in [('chem.py', patch_chem_source), ('frag.py', patch_frag_source)]}
                write(cloud/'c41_legacy_compatibility.json', {tag: dict(before_sha256=before, after_sha256=after) for tag in ['v3','v4']})
                support = root/'outputs/local/c59-c13-compatibility-20261112/channel-support-freeze.json'; support.parent.mkdir(parents=True)
                write(support, dict(all_query_ids=ids, pubchem_supported_ids=ids[:-1]))
                actual = q.copy(deep=True)
                if corruption == 'trimmed_token': actual.loc[119,'adduct'] = '[M+FA-H]-'
                if corruption == 'missing_acquisition': actual = actual.iloc[:-1]
                actual.to_parquet(cloud/'c41_queries.parquet', index=False)
                frame = pd.DataFrame(dict(molecule_id=ids, smiles=['CCO;COC;c1ccccc1']*120))
                for filename in ['c41_p01.csv','c41_p14.csv']: frame.to_csv(cloud/filename,index=False)
                neural = pd.DataFrame(dict(molecule_id=ids,metric_key=['synthetic_key']*120,neural_ok=True,neural_score=1.))
                if corruption == 'neural_fallback': neural.loc[0,'neural_ok'] = False
                for filename in ['c41_neural_p01.parquet','neural_public.parquet','neural_ours.parquet']: neural.to_parquet(cloud/filename,index=False)
                engine = {mid:[] for mid in ids}; pc_ids = ids[:-1]
                if corruption == 'missing_PC': pc_ids = pc_ids[:-1]
                if corruption == 'wrong_PC': pc_ids = ids[1:]
                pc = {mid:{} for mid in pc_ids}
                write(cloud/'eng_lists.json',engine);write(cloud/'pc_lists.json',pc)
                write(cloud/'c76_c13_channel_ids.json',dict(input_IDs=ids,Engine2_IDs=ids,PubChem_IDs=pc_ids,query_selection_or_token_rewriting=False,quality_labels_read=False))
                write(cloud/'c41_report.json',dict(queries=120,baseline_replay_exact=True,metrics_computed=False,query_labels_mounted=False,
                    recipe_selection=False,scoring_submission=False,validation_contract_sha256='contract',baseline_sha256=sha(cloud/'c41_p01.csv'),p14_receipt=dict(input_errors=0)))
                write(cloud/'execution_receipt.json',dict(is_full_run=True,processed_molecules=120,source_sha256=collection.CORE_PINS['c76']))
                devices = [{'name':'Tesla T4'}, {'name':'Tesla T4' if corruption!='wrong_device' else 'CPU'}]
                write(cloud/'c41_t4_allocation.json',dict(devices=devices));write(cloud/'c41_gpu1_preflight.json',dict(name='Tesla T4',dreams_finite=True,molformer_finite=True))
                for filename in ['c41_main_reference_mask_audit.json','c41_engine2_reference_mask_receipt.json']:
                    write(cloud/filename,dict(excluded_reference_rows=10,forbidden_rows_retained=0,window_calls=1,engine_representative_builds=1,engine_representatives_removed=1))
                for filename in ['gl_work/gl_out.json.meta.json','ice_work/ice_out.json.meta.json']:
                    path = cloud/filename;path.parent.mkdir(exist_ok=True)
                    write(path,dict(status='ok',n_mols_scored=1,n_mols_covered=1,n_pred_fail=int(corruption=='GL_failure' and filename.startswith('gl')),n_feat_fail=0))
                write(cloud/'gl_work/gl_out.json.c76_adapter.json',dict(weights_exact=True,decision_device='cpu',cpu_threads=2,intensity_device='cuda:0'))
                log = 'engine lists 120\nengine lists 120\npubchem lists 119\nerrors 0 | popularity re-ranked lists\nPubChem channel\nEngine 1\ncandidate lists\nGLACIER + ICEBERG forward models\nTwo-engine fusion\nMetric-identity fusion\nWindow-40 overlay\nExecution receipt\n'
                (cloud/'submission_pipeline.log').write_text(log,encoding='utf8');write(cloud/'synthetic.log',[dict(data=log)])
                paths = [p for p in cloud.rglob('*') if p.is_file()]
                write(cloud/'fetch-complete.json',dict(source_sha256=sha(collection.__file__),arm='c76',inference_plan_sha256=sha(plan),
                    notebook_sha256=sha(folder/'notebook.ipynb'),remote_source_exact=True,actual_original_image_exact=True,actual_GPU_enabled=True,private=True,
                    kernel='synthetic/synthetic',files_sha256={p.relative_to(cloud).as_posix():sha(p) for p in paths}))
                stack.enter_context(patch.object(collection,'ROOT',root));stack.enter_context(patch.object(collection,'CLOUDS',{'c76':cloud}))
                stack.enter_context(patch.object(collection,'PLAN',plan));stack.enter_context(patch.object(collection,'verify_plan',return_value=None))
                stack.enter_context(patch.object(collection,'read_queries',return_value=q))
                if corruption is None:
                    collection.audit('c76');self.assertEqual(read(cloud/'collection-audit.json')['exact_PC_supported_IDs'],119)
                else:
                    with self.assertRaises(AssertionError): collection.audit('c76')
                    self.assertFalse((cloud/'collection-audit.json').exists())

    def test_full_postprocessing_replay_both_real_models_audit_and_seal_synthetic(self):
        # Core executions are simulated here; P16 and the two unchanged model
        # runtimes execute for real, on synthetic inputs only. All generated
        # receipts remain inside the temporary fixture and are never promoted.
        import numpy as np
        import pandas as pd
        import c74_complete_target as c74
        import c76_submission_contract as contract
        ranker=complete.load('c76_postprocess_fixture_ranker',inference.BUNDLE/'c67_ranker.py')
        identity=c74.module('c76_postprocess_fixture_identity','c09e_features.py').structure
        ids=['SYNTHETIC_POSTPROCESS_'+str(i) for i in range(120)]
        strings=['CCO','COC','c1ccccc1'];keys=[identity(s)[0] for s in strings]
        q=pd.DataFrame(dict(molecule_id=ids,spectrum_id=['SYNTHETIC_SPECTRUM_'+str(i) for i in range(120)],
            precursor_mz=300.,ionization_mode=['positive']*60+['negative']*60,
            adduct=['[M+H]+']*60+['[M-H]-']*59+['[M+FA-H]- '],collision_energy=float('nan'),
            ms2_mzs=[np.array([50.,100.,150.])]*120,ms2_normalized_intensities=[np.array([1.,.5,.2])]*120))
        records=[]
        for mid in ids:
            for rank,(text,key) in enumerate(zip(strings,keys),1):
                row=dict(molecule_id=mid,smiles=text,metric_key=key,baseline_rank=rank,neural_ok=True)
                row.update({field:1. for field in ranker.RAW_FIELDS})
                row.update(neural_cos=.9/rank,neural_fp=.8/rank,neural_score=.9/rank,neural_std=.01,library_max=.99)
                records.append(row)
        raw=pd.DataFrame(records)
        with tempfile.TemporaryDirectory(prefix='c76-synthetic-postprocessing-') as name,ExitStack() as stack:
            root=Path(name);confirm=root/'synthetic-confirmation';confirm.mkdir();prep=confirm/'preparation';prep.mkdir()
            original=root/'synthetic-original';original.mkdir();q.to_parquet(original/'validation-queries.parquet',index=False)
            plan=confirm/'inference-plan.json';write(plan,{'synthetic_only':True})
            reference=prep/'masked-P16-reference';reference.mkdir()
            pd.DataFrame(dict(precursor_mz=[900.],ionization_mode=['positive'],adduct=['[M+H]+'],normalized_smiles=['CCO'])).to_parquet(reference/'ext_meta.parquet',index=False)
            for filename,values in [('ext_mz.npy',[60.,120.,180.]),('ext_in.npy',[1.,.7,.1])]:
                array=np.zeros((1,128),np.float32);array[0,:3]=values;np.save(reference/filename,array)
            reference_files={filename:sha(reference/filename) for filename in ['ext_meta.parquet','ext_mz.npy','ext_in.npy']}
            write(reference/'mask-manifest.json',dict(reference_files=reference_files,audit_files={}))
            audit_files={(reference/filename).relative_to(root).as_posix():pin for filename,pin in reference_files.items()}
            write(prep/'independent-reference-audit.json',dict(passed=True,forbidden_reference_rows=0,
                all_prior_and_global231_chemistry_and_alias_masks_preserved=True,artifact_sha256=audit_files,synthetic_only=True))
            dummy=root/'synthetic-source-notebook.json';write(dummy,{'synthetic':True})
            write(prep/'independent-notebook-audit.json',dict(passed=True,artifact_sha256={dummy.relative_to(root).as_posix():sha(dummy)},synthetic_only=True))
            script_dir=root/'scripts';script_dir.mkdir()
            (script_dir/'complete_c76_confirmation.py').write_bytes(Path(complete.__file__).read_bytes())
            (script_dir/'audit_c76_confirmation_execution.py').write_bytes(Path(execution.__file__).read_bytes())
            clouds={arm:root/('synthetic-cloud-'+arm) for arm in ['legacy','c76']}
            for arm,cloud in clouds.items():
                cloud.mkdir();pd.DataFrame(dict(molecule_id=ids,smiles=[';'.join(strings)]*120)).to_csv(cloud/'c41_p01.csv',index=False)
                pd.DataFrame(dict(molecule_id=ids,smiles=[';'.join(strings)]*120)).to_csv(cloud/'c41_p14.csv',index=False)
                q.to_parquet(cloud/'c41_queries.parquet',index=False)
                for filename in ['c41_neural_p01.parquet','neural_public.parquet','neural_ours.parquet']:raw.to_parquet(cloud/filename,index=False)
                raw[['molecule_id','library_max']].to_parquet(cloud/'frozen_candidate_evidence.parquet',index=False)
                pd.DataFrame(dict(smiles=pd.Series(dtype=str),metric_key=pd.Series(dtype=str))).to_parquet(cloud/'outsider_neural_features.parquet',index=False)
                write(cloud/'execution_receipt.json',dict(processed_molecules=120,is_full_run=True))
                write(cloud/'collection-audit.json',dict(complete_pipeline_binding_sha256=sha(complete.BINDING),files_sha256={
                    path.name:sha(path) for path in cloud.iterdir() if path.is_file()},synthetic_core=True))
            def collected(arm):return clouds[arm],read(clouds[arm]/'collection-audit.json')
            def verify(pins):return contract.verify_files(pins,root)
            parents=confirm/'parents';predictions=confirm/'predictions';prefreeze=confirm/'pre-evaluation-artifact-freeze.json'
            for module in [complete,execution]:
                for attr,value in [('ROOT',root),('CONFIRM',confirm),('PLAN',plan),('PREPARATION',prep),('PARENTS',parents),('PREDICTIONS',predictions),('PRE_FREEZE',prefreeze)]:
                    stack.enter_context(patch.object(module,attr,value))
                stack.enter_context(patch.object(module,'verify_plan',return_value={}))
                stack.enter_context(patch.object(module,'verify_files',side_effect=verify))
                stack.enter_context(patch.object(module,'collected',side_effect=collected))
            stack.enter_context(patch.object(complete,'ORIGINAL',original));stack.enter_context(patch.object(execution,'CLOUDS',clouds))
            complete.replay('legacy');complete.replay('c76');complete.score();complete.freeze();execution.main();complete.seal()
            result=read(confirm/'execution-qualification.json');self.assertTrue(result['both_frozen_heads_independently_rebuilt'])
            self.assertEqual(result['queries'],120)
            frozen=read(confirm/'joint-prediction-freeze.json');self.assertEqual(set(frozen['candidates']),{'cpu','gpu'})
            self.assertEqual(set(frozen['controls']),{'legacy','c76'})
            for trial in ['cpu','gpu']:
                final=pd.read_csv(predictions/(trial+'.csv'));self.assertEqual(final.molecule_id.tolist(),ids)
            # A changed final prediction must fail the immutable pre-seal hashes.
            (predictions/'cpu.csv').write_text('SYNTHETIC CORRUPTION',encoding='utf8')
            with self.assertRaises(AssertionError):verify(read(prefreeze)['files_sha256'])


if __name__ == '__main__':
    import os
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='2'
    import torch
    unittest.main(verbosity=2)
