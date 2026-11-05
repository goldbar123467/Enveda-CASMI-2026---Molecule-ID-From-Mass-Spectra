"""Paired acquisition-sampling experiments using the original Saner training loop.

General: choose any spectrum of a sampled molecule. Native-preferred: when that
training molecule has a matched Enveda spectrum, choose only its native spectra.
Molecule frequency, loss, negatives, architecture and epoch budget stay identical.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import local_seeds as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['general', 'native'], required=True)
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--epochs', type=int, default=128)
    ap.add_argument('--native-index', type=Path)
    args = ap.parse_args()
    L.OVERRIDES.update(width=256, layers=4, heads=8)
    original_setup = L.setup

    def setup(split):
        cfg, data, device = original_setup(split)
        audit = dict(mode=args.mode, seed=L.BASE_SEED + 1000 * args.seed, epochs=args.epochs,
                     architecture={k: cfg[k] for k in ['width', 'heads', 'layers']},
                     sampling='uniform by training molecule; native spectra preferred only within that molecule' if args.mode == 'native' else 'original all-spectrum sampling',
                     training_identities=len(data.train), heldout_identities=len(data.val),
                     training_indices_sha256=hashlib.sha256(np.asarray(data.train, dtype='<i8').tobytes()).hexdigest(),
                     heldout_identity_sha256=hashlib.sha256('\n'.join(sorted(data.mol.key.iloc[data.val].astype(str))).encode()).hexdigest(),
                     fixed_final_epoch=True)
        if args.mode == 'native':
            assert args.native_index and args.native_index.is_file()
            native = np.load(args.native_index, allow_pickle=False)
            assert native.ndim == 1 and len(native) and native.min() >= 0 and native.max() < len(data.spec)
            assert len(np.unique(native)) == len(native)
            training = set(map(int, data.train))
            heldout = set(map(int, data.val))
            by_mol = {}
            for index, mol in zip(native, data.spec.mol_idx.to_numpy()[native]):
                if int(mol) in training:
                    by_mol.setdefault(int(mol), []).append(int(index))
            assert not set(by_mol) & heldout
            for mol, indices in by_mol.items():
                data.by_mol[mol] = np.asarray(indices, dtype=np.int64)
            audit.update(native_training_molecules=len(by_mol), native_training_spectra=sum(map(len, by_mol.values())),
                         native_index_sha256=hashlib.sha256(args.native_index.read_bytes()).hexdigest())
            assert len(by_mol) > 1000
        L.OUT.mkdir(parents=True, exist_ok=True)
        (L.OUT / 'training_manifest.json').write_text(json.dumps(audit, indent=2))
        print('PAIRED_TRAIN_MANIFEST', json.dumps(audit), flush=True)
        return cfg, data, device

    L.setup = setup
    L.train('holdout', [args.seed], args.epochs, 'np')


if __name__ == '__main__':
    main()
