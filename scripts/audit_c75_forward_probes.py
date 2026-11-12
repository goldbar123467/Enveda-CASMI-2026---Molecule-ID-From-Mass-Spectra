"""Compare raw USED forward scores across controlled runtime probes, no truth."""
import math
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    expected_path=base/'original-cloud-raw-forward/gl_work/gl_out.json'
    expected=read(expected_path)
    input_path=base/'gl-engineering-probe-v1/original-cloud-nine-input.json'
    cases={'Torch211_base_CPU':'gl-engineering-probe-v1',
           'Torch210_base_CPU':'gl-torch210-probe-v1',
           'Torch210_original_numeric':'gl-exact-numeric-probe-v1',
           'Torch210_original_graph':'gl-graph-matched-probe-v1'}
    dest=base/'controlled-forward-probe-comparison.json';assert not dest.exists()
    results={}
    for name,folder in cases.items():
        path=base/folder/'cloud-nine.json';metadata=read(str(path)+'.meta.json')
        assert metadata['status']=='ok' and metadata['n_mols_scored']==metadata['n_mols_covered']==9
        assert metadata['n_pred_fail']==metadata['n_feat_fail']==0
        values=read(path);pairs=[];missing=0
        for mid,entries in values.items():
            for smiles,value in entries.items():
                prior=expected[mid].get(smiles)
                a=value is not None and math.isfinite(value)
                b=prior is not None and math.isfinite(prior)
                missing+=a!=b
                if a and b:pairs.append((value,prior))
        results[name]=dict(molecules=len(values),compared_scores=len(pairs),missing_pattern_changes=missing,
            exact_scores=sum(x==y for x,y in pairs),maximum_absolute_difference=max(abs(x-y) for x,y in pairs),
            outside_frozen_tolerance=sum(abs(x-y)>1e-5+1e-5*abs(y) for x,y in pairs),
            output_sha256=sha(path),metadata_sha256=sha(str(path)+'.meta.json'),torch=metadata['torch'],
            all_supported_molecules_complete=True)
    write(dest,dict(completed_utc=stamp(),source_sha256=sha(__file__),
        original_cloud_raw_output_sha256=sha(expected_path),same_original_nine_input_sha256=sha(input_path),
        comparisons=results,full_context_early_rejection_sha256=sha(base/'gl-full-context-probe-v1/early-rejection.json'),
        USED_only=True,truth_or_quality_read=False,new_source_quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 CONTROLLED RAW FORWARD COMPARISON',results,flush=True)


if __name__=='__main__':main()
