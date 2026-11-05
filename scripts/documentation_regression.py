"""Fixed documentation-prior regression on the three cached natural-product scenarios."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tail_fusion import paired_summary

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.time()
    candidates = pd.read_parquet(ROOT / 'outputs/local/nppanel/main_candidates.parquet')
    annotation = pd.read_parquet(ROOT / 'outputs/research/pop-prior/np_candidate_structures.parquet').set_index('inchikey14')
    rows = []
    for (scenario, key), group in candidates.groupby(['scenario', 'molecule'], observed=True, sort=True):
        group = group.sort_values('rank', kind='stable').head(25).reset_index(drop=True)
        prior = annotation.reindex(group.candidate_key.astype(str)).fillna(0)
        npflag = prior.np_coconut.astype(bool) | prior.np_lotus.astype(bool) | prior.src_natural_product.astype(bool)
        score = (1 / (3 + group['rank'].to_numpy()) + .10 * prior['pop'].to_numpy() / 20
                 + (.05 if float(group.lib_max.max()) >= .87 else 0) * prior.src_spectral_library.to_numpy()
                 + .02 * npflag.to_numpy())
        order = [0] + [int(i) for i in np.argsort(-score, kind='stable') if i != 0]

        def rr(index):
            hit = np.flatnonzero(group.is_truth.to_numpy()[index])
            return 1 / (1 + hit[0]) if len(hit) else 0.

        rows.append(dict(scenario=str(scenario), key=key, before=rr(np.arange(len(group))), after=rr(order)))
    frame = pd.DataFrame(rows)
    assert frame.groupby('scenario').size().eq(250).all()
    report = {scenario: paired_summary(group.before, group.after) for scenario, group in frame.groupby('scenario')}
    out = ROOT / 'outputs/local/professional/np_documentation_regression.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f'[{time.time() - start:.0f}s] DONE fixed documentation regression; 250 identities per scenario, missing truths retained as zero', flush=True)


if __name__ == '__main__':
    main()
