"""Prospective decision gate for one frozen recipe on fresh balanced120."""
import math


def gate(comparisons,polarity_comparisons):
    failures=[]
    for name in ['candidate_vs_P01','candidate_vs_P16','alias_excluded_vs_P01','alias_excluded_vs_P16']:
        summary=comparisons[name]
        assert summary['n']==120,'Every query, including absent truths, is required'
        delta=float(summary['delta']);lo,hi=map(float,summary['ci95'])
        assert all(math.isfinite(v) for v in [delta,lo,hi]) and lo<=hi
        alias=name.startswith('alias_')
        passed=(delta>=0 and lo>=0) if alias else (delta>0 and lo>0)
        if not passed:failures.append(name)
    for view in ['all_reference','input_alias_excluded']:
        for mode in ['positive','negative']:
            for parent in ['P01','P16']:
                name=view+':'+mode+':'+parent;summary=polarity_comparisons[name]
                assert summary['n']==60,'Both fixed source strata must remain complete'
                assert math.isfinite(float(summary['delta']))
                if summary['delta']<0:failures.append(name)
    return dict(passed=not failures,failed_comparisons=failures)
