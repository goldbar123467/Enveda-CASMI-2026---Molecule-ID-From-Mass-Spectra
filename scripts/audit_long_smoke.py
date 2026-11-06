"""Independent schema/metric-dedup/public blend and fivefold arithmetic audit."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import audit_smoke_output


def main():
    folder = Path(sys.argv[1])
    audit_smoke_output.main()
    receipt = json.loads((folder / 'p12_qualification_receipt.json').read_text())
    assert receipt['model']['models'] == 5 and receipt['model']['family'] == 'control'
    assert receipt['ensemble']['models'] == 5 and receipt['ensemble']['input_errors'] == 0
    ours = pd.read_parquet(folder / 'neural_ours.parquet').set_index(['molecule_id', 'smiles'])
    parts = []
    for fold in range(5):
        path = folder / f'neural_candidate_features-fold-{fold}.parquet'
        part = pd.read_parquet(path).set_index(['molecule_id', 'smiles'])
        assert not part.index.has_duplicates and set(part.index) == set(ours.index)
        assert json.loads(Path(str(path) + '.errors.json').read_text()) == []
        parts.append(part.reindex(ours.index))
    error = float(np.abs(ours.neural_score.to_numpy() - np.mean([part.neural_score.to_numpy() for part in parts], axis=0)).max())
    assert error < 1e-6
    report_path = folder / 'independent_smoke_audit.json'
    report = json.loads(report_path.read_text())
    report.update(control_models=5, independent_fivefold_mean_error=error,
                  manifest_sha256=receipt['model']['manifest_sha256'])
    report_path.write_text(json.dumps(report, indent=2))
    print('FIVEFOLD INDEPENDENT AUDIT PASSED', error)


if __name__ == '__main__':
    main()
