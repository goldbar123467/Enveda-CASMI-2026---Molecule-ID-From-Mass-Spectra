# Attribution and licensing

This repository records an Enveda CASMI competition workspace. It contains
third-party code and notebook derivatives; it does not grant a blanket license
over those works. Retain their original notices and observe the license of each
source before reuse. Training data, pretrained weights, caches and downloaded
research notebooks are excluded from Git.

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
