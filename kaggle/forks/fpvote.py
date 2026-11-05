"""Fold-safe FPNet vote: score each listed candidate with the five fold-safe FPNets (f.z, as in the v4 engine).

usage: python fpvote.py <evidence.parquet> <queries.parquet> <out.parquet> [device] [max_molecules]
Adds `fp_vote` (f.z under the five-model mean logits, single-spectrum and merged-polarity views averaged, as the
engine does) and `fp_vote_z` (z-score within the molecule; 0 where unavailable) to the evidence frame.
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd


def find(pattern):
    return sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True), key=len)


def main(evidence, queries, out, device='cuda', max_molecules=0):
    t0 = time.time()
    code = os.path.dirname(os.path.dirname(find('casmi26-v3-models/code/casmi/fpnet.py')[0]))
    sys.path.insert(0, code)
    from casmi import chem, fpnet
    from casmi.spectra import merge_spectra
    paths = [find(f'fpnet_fold{k}.pt')[0] for k in range(5)]
    bits = np.load(os.path.join(os.path.dirname(paths[0]), 'fp_bits.npy'))
    bank = fpnet.ModelBank(paths, device=device)
    assert bank.nbits == len(bits), (bank.nbits, len(bits))
    frame = pd.read_parquet(evidence)
    q = pd.read_parquet(queries)
    q['molecule_id'] = q.molecule_id.astype(str)
    query = {m: g for m, g in q.groupby('molecule_id', sort=False)}
    frame['fp_vote'] = np.nan
    cache, done = {}, 0
    for mid, g in frame.groupby('molecule_id', sort=False):
        mid = str(mid)
        if not bool(g.processed.iloc[0]) or mid not in query:
            continue
        if max_molecules and done >= max_molecules:
            break
        spectra = []
        for r in query[mid].itertuples(index=False):
            ce = np.atleast_1d(np.asarray(r.collision_energy_ev, dtype=float)) if r.collision_energy_ev is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                prec=float(r.precursor_mz), adduct=r.adduct,
                                mode=1 if str(r.ionization_mode) == 'positive' else -1,
                                ce=float(np.nanmean(ce)) if len(ce) else np.nan, ce_n=int(len(ce)) if len(ce) else 1))
        items_s = []
        for s in spectra:
            pm, pi = fpnet.prep_peaks(s['mz'], s['it'], s['prec'])
            ok = not np.isnan(s['ce'])
            items_s.append(dict(mz=pm, it=pi, prec=s['prec'], adduct_ix=chem.adduct_index(s['adduct']),
                                ce=s['ce'] if ok else 0.0, ce_known=1.0 if ok else 0.0,
                                n_merged=min(s['ce_n'], 8), mode=float(s['mode'])))
        items_m = []
        for md in (1, -1):
            grp = [s for s in spectra if s['mode'] == md]
            if not grp:
                continue
            mm, ii = merge_spectra([(s['mz'], s['it']) for s in grp])
            prec = float(np.median([s['prec'] for s in grp]))
            pm, pi = fpnet.prep_peaks(mm, ii, prec)
            ces = [s['ce'] for s in grp if not np.isnan(s['ce'])]
            adducts = [s['adduct'] for s in grp]
            items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(max(set(adducts), key=adducts.count)),
                                ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                                n_merged=min(sum(max(1, s['ce_n']) for s in grp), 8), mode=float(md)))
        if not items_s:
            continue
        z = 0.5 * (bank.logits(items_s).mean(0) + bank.logits(items_m).mean(0))
        scores = []
        for s in g.smiles:
            if s not in cache:
                raw = chem.raw_fingerprint(s)
                cache[s] = None if raw is None else raw[bits].astype(np.float32)
            scores.append(np.nan if cache[s] is None else float(cache[s] @ z))
        frame.loc[g.index, 'fp_vote'] = scores
        done += 1
        if done % 50 == 0:
            print('FPVOTE', done, 'molecules', round(time.time() - t0), 's', flush=True)
    v = frame.fp_vote
    grp = v.groupby(frame.molecule_id, sort=False)
    frame['fp_vote_z'] = ((v - grp.transform('mean')) / grp.transform(lambda s: s.std(ddof=0)).clip(lower=1e-6)).fillna(0).clip(-5, 5)
    tmp = str(out) + '.tmp'
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    print('FPVOTE DONE molecules', done, 'scored rows', int(v.notna().sum()), 'seconds', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3] if len(a) > 3 else 'cuda', int(a[4]) if len(a) > 4 else 0)
