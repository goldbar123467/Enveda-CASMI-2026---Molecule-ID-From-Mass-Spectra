"""Query-only population support audit; no scoring, labels, or predictions read."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import json
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from trace_c37_selection import ROOT, OUT, C36, sha, write
from legacy_formate_compat import normalize_legacy_adduct


def query_table(frame):
    assert not any(c in frame for c in ['smiles', 'truth', 'key', 'inchikey14']), 'Query metadata only'
    records = []
    for mid, g in frame.groupby('molecule_id', sort=False):
        modes = set(g.ionization_mode.astype(str))
        assert modes and modes <= {'positive', 'negative'}
        adducts = Counter(normalize_legacy_adduct(a) for a in g.adduct)
        # Deterministic plurality; mixed queries separately identified below.
        dominant = min(adducts, key=lambda a: (-adducts[a], a))
        ce = []
        if 'collision_energy_ev' in g:
            for value in g.collision_energy_ev:
                if value is not None:
                    v = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
                    ce.extend(v[np.isfinite(v)].tolist())
        records.append(dict(molecule_id=str(mid), mode='mixed' if len(modes)>1 else next(iter(modes)), dominant_adduct=dominant,
                            adduct_count=len(adducts), adducts='|'.join(sorted(adducts)),
                            instruments='|'.join(sorted(set(g.instrument_type.astype(str)))),
                            spectra=len(g), precursor_median=float(np.median(g.precursor_mz)),
                            collision_energy_available=bool(ce), collision_energy_median=float(np.median(ce)) if ce else None,
                            peaks_median=float(np.median([len(x) for x in g.ms2_mzs]))))
    result = pd.DataFrame(records); assert result.molecule_id.is_unique
    return result


def summarize(f):
    def counts(column): return {str(k): int(v) for k,v in f[column].value_counts().items()}
    def quantiles(column): return {str(k): float(v) for k,v in f[column].quantile([0,.1,.5,.9,1]).items()}
    return dict(queries=len(f), spectra=int(f.spectra.sum()), mode=counts('mode'), dominant_adduct=counts('dominant_adduct'),
                instruments=counts('instruments'), spectra_per_query=counts('spectra'), mixed_adduct_queries=int(f.adduct_count.gt(1).sum()),
                precursor_median_quantiles=quantiles('precursor_median'), peaks_median_quantiles=quantiles('peaks_median'),
                collision_energy_available_queries=int(f.collision_energy_available.sum()))


def support(target, panel):
    columns = ['mode', 'dominant_adduct']
    t = target.groupby(columns).size(); p = panel.groupby(columns).size()
    rows = []
    for key, n in t.items():
        count = int(p.get(key, 0))
        rows.append(dict(mode=key[0], adduct=key[1], target_queries=int(n), panel_queries=count,
                         target_weight=float(n/len(target)), importance_weight=float((n/len(target))/(count/len(panel))) if count else None))
    supported = sum(r['target_queries'] for r in rows if r['panel_queries'])/len(target)
    weights = []
    lookup = {(r['mode'],r['adduct']):r['importance_weight'] or 0. for r in rows}
    for r in panel.itertuples(index=False): weights.append(lookup.get((r.mode,r.dominant_adduct),0.))
    weights = np.asarray(weights)
    effective = float(weights.sum()**2/(weights@weights)) if weights@weights else 0.
    return dict(strata=rows, target_mode_adduct_support_fraction=supported, importance_effective_queries=effective,
                exact_target_instrument_query_support=float(target.instruments.isin(set(panel.instruments)).mean()),
                warning='Importance weights are a covariate-support diagnostic only. No weighted MRR was computed; unmeasured chemistry and source exposure remain unresolved.')


def main():
    pa.set_cpu_count(2)
    assert (ROOT/'.local/compute-paused.json').is_file()
    assert not (OUT/'population-audit.json').exists()
    paths = dict(target=OUT/'unlabelled-target/test.parquet',
                 primary=ROOT/'outputs/local/c21-negative-tail-20261110/queries.parquet',
                 historical_source=ROOT/'outputs/forks/c09-clean-validation-v2/c09_clean_queries.parquet')
    # Protocol is sealed before query summaries and uses no old local/LB outcomes.
    protocol = dict(version=1, frozen_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=sha(__file__),
        inputs={n:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for n,p in paths.items()},
        scope='Query metadata and source support only. No fresh C13 data, target truth, predictions, local MRR or historical leaderboard outcomes read.',
        prospective_population=dict(
            unit='One molecule query with its original complete usable acquisition set; never select one scan for convenience.',
            sources='At least two documented external acquisition sources, including a timsTOF source; source held out from candidate-model selection.',
            sampling='Deterministic seeded draw within ionization-mode/canonical-adduct/instrument/acquisition-count strata, independent of predictions and candidate truth coverage.',
            target_mix='Freeze strata from the full unlabelled public test metadata; never optimize strata to reproduce prior leaderboard signs.',
            chemistry='Freeze transitive groups from official identity, scaffold, formula and Morgan Tanimoto>=0.70; no group crosses train/selection/source-confirmation splits.',
            exposure='Audit exact and near acquisition matches and documented training/reference identities for every adopted model. Unknown pretraining exposure must be reported, not assumed absent.',
            missing_support='An absent target stratum or source cannot be repaired with infinite weights, repeated molecules, or resampling; report the gap and obtain new qualifying acquisitions.',
            endpoints='All-query MRR@25 with absent truth zero; candidate recall@25, true-top1 gains/losses, recovered/displaced answers; paired chemistry-cluster uncertainty by source and polarity.',
            decision='Freeze one complete pipeline candidate before source outcomes. Require material supported development gain and no adverse source/polarity point result; separately disclose uncertainty. Use a never-consumed independent confirmation only after those checks.',
            interim='Current panels remain diagnostic. No claim of a representative or leaderboard-calibrated panel until source and covariate support plus exposure audits pass.',
            c13='Existing untouched C13 stays sealed; this audit does not establish that it matches target acquisition support.'),
        gpu_work=False, quality_metrics_computed=False)
    write(OUT/'prospective-validation-protocol.json', protocol)
    queries = {n: query_table(pd.read_parquet(p)) for n,p in paths.items()}
    assert len(queries['target']) == 400 and len(queries['primary']) == 240 and len(queries['historical_source']) == 80
    for n,f in queries.items(): f.to_csv(OUT/(n+'-population-features.csv'),index=False)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(OUT/'prospective-validation-protocol.json'),
                  summaries={n:summarize(f) for n,f in queries.items()},
                  support={n:support(queries['target'],queries[n]) for n in ['primary','historical_source']},
                  target_formate_alias_count=0, target_alias_lookup_repair_changes_queries=0,
                  conclusion='Neither current validation population supports the target instrument. Acquisition count, polarity/adduct mix and source provenance require explicit prospective support; resampling existing rows does not fix the missing source.',
                  labels_read=False, predictions_read=False, quality_metrics_computed=False, reserved_holdout_opened=False,
                  new_inference=False, production_changed=False, validation_population_qualified=False)
    target_raw = pd.read_parquet(paths['target'],columns=['adduct'])
    assert not target_raw.adduct.eq('[M+FA-H]-').any()
    write(OUT/'population-audit.json', report)
    print('DONE C37 POPULATION',json.dumps(report),flush=True)


if __name__ == '__main__': main()
