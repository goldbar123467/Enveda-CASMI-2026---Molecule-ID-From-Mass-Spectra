"""Package kaggle/forks/x3-rule-replay as a CPU notebook from the s1 fork's pinned sources. Local packaging only."""
import ast
import json
from pathlib import Path

FORKS = Path(__file__).resolve().parents[1] / "kaggle/forks"
nb = json.loads((FORKS / "s1-p477/notebook.ipynb").read_text(encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
modules_cell = "\n".join(l for l in code[4].split("\n") if not l.startswith(("FROZEN_B64 = ", "print(")))
physical_cell = code[6]
head = (
    "import os, sys, subprocess, glob\nfrom pathlib import Path\n"
    "whl = [w for w in glob.glob('/kaggle/input/**/rdkit-2026.3.3-*.whl', recursive=True)"
    " if f'cp{sys.version_info.major}{sys.version_info.minor}' in w]\n"
    "print('wheel', whl)\n"
    "if whl:\n    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], check=True)\n"
    "ROOT = Path('/kaggle/working'); CODE = ROOT / 'p2s_code'; CODE.mkdir(parents=True, exist_ok=True)\n"
)
tail = (FORKS / "x3-rule-replay/replay_tail.py").read_text(encoding="utf-8")
cells = [head, modules_cell, physical_cell, tail]
for c in cells:
    ast.parse(c)
out = {
    "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
               "source": c.splitlines(keepends=True)} for c in cells],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(FORKS / "x3-rule-replay/notebook.ipynb").write_text(json.dumps(out, indent=1), encoding="utf-8")
meta = json.loads((FORKS / "x1-panel-probe/kernel-metadata.json").read_text(encoding="utf-8"))
meta.update(id="clarkkitchen/casmi26-x3-rule-replay", title="casmi26 x3 rule replay",
            dataset_sources=["sanpier/catalogue-experiment-saner", "metric/rdkit-2026-3-3-wheel"])
(FORKS / "x3-rule-replay/kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print("packaged x3-rule-replay")
