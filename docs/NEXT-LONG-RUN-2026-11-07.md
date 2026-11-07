# Public research and proposed next long run — 7 November 2026

Status: the proposal below was written while paused. The user subsequently
authorized resumption, F07 training and two submissions on 7 November. See
[F07 execution protocol](FROZEN-TRAINING-2026-11-07.md). P12 failed its hidden
rerun with no score; P01 remains the production anchor.

## Recommendation

The best-supported next GPU experiment is an MSAlign-style retrieval model with
frozen pretrained representations and learned projection heads. Mine difficult
mass-compatible molecular candidates from permitted public structure sources,
then train against those candidates. This tests new representations rather than
repeating the candidate-loss-only experiment already attempted on5 November.
It is an experiment recommendation, not an expected Kaggle-score guarantee.

L06 used ten128-epoch fits and finished all1,280 epochs with no restarts. Control
beat external training on reserved retrieval confirmation0.680621 vs0.646321.
The external cache added only996 new identities despite hundreds of thousands
of spectra. More time or bulk spectrum downloads alone are not established gains.

## Primary sources checked

- [MSAlign v3, revised2 November](https://arxiv.org/html/2605.19752v3): frozen
  DreaMS and MolDeBERTa encoders, candidate-based contrastive training, metadata,
  and complementary late score fusion. Its ablation also tests IBM MoLFormer.
  These are benchmark results, not this competition's scores.
- [Official MSAlign implementation](https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026):
  representation precomputation, separate preprocessing environments, candidate
  masks, training-only checkpoint selection and later evaluation. The default
  molecular model is gated; the repository explicitly requires accepting terms.
- [Official DreaMS](https://github.com/pluskal-lab/DreaMS): public pretrained
  spectrum representations. The API uses1024-dimensional vectors and defaults
  to batches of32, but actual RTX2070 memory/throughput must be measured.
- [IBM MoLFormer model card](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct):
  Apache-2.0; frozen features are an intended use; PubChem/ZINC pretraining.
  This is the practical first molecular-encoder candidate, subject to documenting
  the exact checkpoint/source/data provenance required by the competition.
- [MolDeBERTa model card](https://huggingface.co/SaeedLab/MolDeBERTa-base-123M-mtr):
  CC-BY-NC-ND-4.0 metadata. Access and reuse conditions must be resolved before
  choosing that exact model. Availability is not blanket competition approval.
- Current public source snapshots, downloaded using Kaggle's API:
  [peaks-to-structures-v2](https://www.kaggle.com/code/zz3t3rn41/casmi26-peaks-to-structures-v2)
  and [Roman Rozen's notebook](https://www.kaggle.com/code/romanrozen/0-482-casmi-from-peaks-to-structures).
  Snapshots are in ignored `outputs/research/2026-11-07-long-run/`. Titles and
  historical best scores do not establish the latest version's reproducibility.

The official NVIDIA catalog was also checked using its list-only CLI. No skill
was installed; specialized NVIDIA runtime skills are not necessary for the
proposed frozen-representation experiment.

## Public mining: useful targets

1. **Hard negative structures:** known training identities and permitted public
   PubChem/COCONUT/LOTUS structures close in mass and fingerprint/embedding space.
   Keep held-out identity groups out of learned head training/negative mining.
   Known training formulas can define hard training decoys; inference and
   evaluation must use available precursor/adduct information, not a truth formula.
2. **Actual coverage gains:** count new canonical identities and natural-product
   chemical coverage before acquiring more spectra. Audit source/license and
   deduplicate shared identities and model inputs as in L06.
3. **Candidate scoring consistency:** the new public peaks-v2 source applies its
   public/private neural blend to outsider candidates as well as final lists.
   Our current P12 outsider channel uses the control ensemble alone. Test the
   consistent blend as a separate fixed ablation; do not attribute it to training.
4. **LOTUS retrieval rather than arbitrary fills:** the inspected public source
   mostly uses LOTUS to fill vacant output slots after deduplication. A stronger
   hypothesis is to include mass-compatible LOTUS candidates in the scored pool
   before final ranking and measure recall/candidate displacement explicitly.
   Current source-prior tables already contain LOTUS/COCONUT membership; adding
   the same prior again is not new coverage.

Avoid importing the latest rank-1 promotion thresholds merely because two panel
molecules improve. The local panels have been repeatedly reused and their apparent
precision is not independent confirmation. Preserve the0.480 production baseline.

## Proposed resource and experiment contract

- Create a **new run directory and manifest**. L06 is complete; its checkpoints
  and byte pins remain immutable. Extending the old deadline does not add epochs.
- First benchmark small batches of both encoders independently on the RTX2070;
  begin with8–16 examples, adjust from measured VRAM. Do not load both encoders
  plus the trainer at once. Keep a separate dependency environment because
  DreaMS's supported stack differs from the current training environment.
- Cache frozen embeddings in atomic shards and memory-map them during training.
  FLOAT16 storage for2.5 million1024-dimensional spectra is approximately4.77GiB,
  before molecule embeddings, indices and temporary files. Storage precision and
  score stability require a numerical check; cache space is a real budget item.
- Train projection heads only, starting with the competition-only population.
  Use mass-compatible hard candidates and fixed excluded-identity splits. Compare
  two seeds; checkpoint optimizer/scaler/RNG and select checkpoints on development
  only. Source/NP weighting is a later controlled ablation, not assumed beneficial.
- Budget **up to24 hours** for encoding, head experiments and validation, subject
  to an initial measured-throughput estimate. This is a budget, not an ETA or a
  reason to continue after evidence turns negative. Retain fourGiB free RAM and
  desktop GPU headroom; keep CPU thread counts low.
- Audit negative-ion, adduct and high-mass coverage before adopting the encoder.
  DreaMS's default DataFormatA specifies singly positive charge and a1000-Th
  ceiling, while the MSAlign playground accepts broader metadata. Producing an
  embedding is not evidence of reliable extrapolation. Measure those cohorts and
  retain the existing pipeline for unsupported cases rather than deleting them.
- Parameterize the new supervisor for that run and update both its global deadline
  and the trainer limit. Preserve30-minute health checks, bounded restarts,
  provenance checks and resumable checkpoints. No unbounded restart loop.
- Validate on fresh excluded identities with the real candidate generator and
  physical/fusion/dedup stages. Report candidate-pool recall, conditional ranking
  MRR and end-to-end MRR separately, counting absent truths as zero. Use separate
  familiar/unseen-spectrum/candidate-absent cohorts and document pretrained-data
  overlap. Do not call the reused250-NP panel an untouched test.
- If promising, add the new score as a controlled complementary channel to the
  existing pipeline; fit fusion only on development and check it once on reserved
  confirmation. Full-data production fitting and real Kaggle smoke follow that.

[Kaggle's code requirements](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)
cap the submitted CPU/GPU notebook at9 hours with internet disabled and allow
qualifying public external data/models. A longer local training budget does not
extend hidden-rerun inference time. Exact model/data eligibility still follows the
host rulings and deliverable requirements in the discussion review.

## Immediate order of work if resumed

1. Wait for P12's real leaderboard result before choosing the production anchor.
2. Audit candidate coverage and benchmark frozen encoder throughput/compatibility.
3. Run a small held-out projection-head pilot; promote to the longer budget only
   if candidate ranking improves in the actual pipeline without a recall loss.
4. Keep outsider-blend consistency and candidate expansion as separately measured
   changes. Avoid spending all compute or submissions on one combined hypothesis.
