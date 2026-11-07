# Attribution and licensing

This repository records an Enveda CASMI competition workspace. It contains
third-party code and notebook derivatives; it does not grant a blanket license
over those works. Retain their original notices and observe the license of each
source before reuse. Training data, pretrained weights, caches and downloaded
research notebooks are excluded from Git.

F07 (7 November) uses the frozen-feature alignment design from
[MSAlign](https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026), official
[DreaMS](https://github.com/pluskal-lab/DreaMS) (MIT) and
[IBM MoLFormer](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct)
(Apache-2.0). Exact revisions and checkpoint provenance are recorded in
`docs/FROZEN-TRAINING-2026-11-07.md`. P13's consistent outsider blend is inspired
by [zz3t3rn41's public peaks-v2 notebook](https://www.kaggle.com/code/zz3t3rn41/casmi26-peaks-to-structures-v2).
It retains the previously attributed P01 pipeline and checkpoint lineage.

The 2026-11-06 common-fold/model-folder/OOF/L2 ensemble workflow is inspired by
[Chris Deotte's Kaggle Playground solutions](https://github.com/cdeotte/kaggle-playground-solutions)
(Apache-2.0, commit e5a03fffe08a8904feb40c66b85142abb1ec6cc7). New trainer/stack
code is written for molecular retrieval; no binary-classification or GPT recipe
is transplanted. The Saner model/features remain credited below. New external
spectra come from MassBank 2026.03 and directly contributed GNPS libraries;
record-level source, accession and licenses are retained in the external cache.
See docs/LONG-TRAINING-2026-11-06.md for upstream links, mirror hashes, exclusions
and reproducibility constraints. The 0.480 lineage has now been reproduced on
our account (submission 58032057, verified 2026-11-06); older statements below
describe the earlier research snapshot.

The public pipeline lineage is credited in the notebooks and `STRATEGY.md`:

- Dmitrii Gluzdov: [data and attributed v4g reproduction assets](https://www.kaggle.com/code/dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures),
  candidate generation, library and forward-model pipeline, fold-safe FPNet and
  natural-product benchmark assets. Consult their `ATTRIBUTION` and
  `SOURCE-LICENSES` files; many public-lineage assets are CC BY-NC.
- Saner Turhaner: [spectrum-to-structure training kit](https://www.kaggle.com/datasets/sanpier/casmi-spectrum-model-v1),
  neural scoring, reciprocal-rank fusion and labelled catalogue experiment.
  `scripts/local_seeds.py` uses this downloaded kit; the training loop in
  `scripts/scientific_rounds.py` is adapted from its `train.py` with the
  candidate-only objective described in the experiment report.
- Roman Rozen: [From Peaks to Structures](https://www.kaggle.com/code/romanrozen/0-468-casmi-from-peaks-to-structures),
  leader-safe consensus and candidate recovery.
- Roman Tamrazov: [Spectral Parallax](https://www.kaggle.com/code/romantamrazov/casmi-2026-spectral-parallax),
  fragment checks and provenance reranking.
- busyaprime: [Two 0.472 Fixes, Merged the Right Way](https://www.kaggle.com/code/busyaprime/lb-0-477-two-0-472-fixes-merged-the-right-way),
  the 0.477 merge recipe.
- alex chilton, bgunshui, Baidalin Adilzhan and Kunal Desale: the
  [reported 0.480 ensemble lineage](https://www.kaggle.com/code/alexchilton/casmi26-lb-0-480-neural-ensemble)
  examined for the 2026-11-05 experiment. Its standardized ensemble motivates our
  tail-only fusion; we have not reproduced that notebook's Kaggle score.
- Heliosli: [training-structure and extended-library overlays](https://www.kaggle.com/code/heliosli/casmi26-0-482),
  examined but not adopted because required assets were unavailable.
- Huseyin Emre Aksoy: the upstream Apache-2.0 MetFrag-lite fragment scorer,
  reached through Ahmed Berat Özer's v7c notebook in the embedded lineage.

Scientific motivation: Paul Krzakala et al.,
[MSAlign, arXiv:2605.19752v2](https://arxiv.org/abs/2605.19752v2).
Our candidate-loss/fusion experiment uses the existing Saner model; it is not
the MSAlign architecture or a reproduction of its published results.
