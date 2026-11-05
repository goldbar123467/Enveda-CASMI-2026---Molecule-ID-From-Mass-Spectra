"""Fixed molecule-mixture sampling, restricted to the caller's training identities."""
import numpy as np


def mixture_probabilities(training, molecule_keys, natural_product_keys, fraction=.5):
    training=np.asarray(training,dtype=np.int64)
    assert training.ndim==1 and len(training)>1 and len(np.unique(training))==len(training)
    assert 0<=fraction<=1 and training.min()>=0 and training.max()<len(molecule_keys)
    flags=np.array([str(molecule_keys[i]) in natural_product_keys for i in training],dtype=bool)
    count=int(flags.sum())
    assert 0<count<len(training),'Need both annotated and unannotated training molecules'
    probability=np.full(len(training),(1-fraction)/len(training),dtype=np.float64)
    probability[flags]+=fraction/count
    assert np.isfinite(probability).all() and np.isclose(probability.sum(),1)
    return probability,dict(training_molecules=len(training),annotated_np_molecules=count,
                            baseline_np_fraction=count/len(training),sampled_np_fraction=float(probability[flags].sum()),
                            mixture_fraction=fraction)


def patch_training_loop(source):
    """Change initialization and positive-molecule sampling only; keep the original loss."""
    changes={
        '    model=SpectrumCandidateModel(cfg).to(device)':
        '    model=SpectrumCandidateModel(cfg).to(device)\n    model.load_state_dict(_INITIAL_STATE,strict=True)',
        "batch_ids=rng.choice(ids,min(cfg['batch_size'],len(ids)),replace=False)":
        "batch_ids=rng.choice(ids,min(cfg['batch_size'],len(ids)),replace=False,p=_SAMPLING_PROBABILITIES)"}
    for old,new in changes.items():
        assert source.count(old)==1,'Pinned original training loop changed'
        source=source.replace(old,new)
    compile(source,'fixed_mixture_train_stage','exec')
    return source
