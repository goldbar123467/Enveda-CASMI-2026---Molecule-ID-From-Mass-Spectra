# F07: frozen DreaMS/MoLFormer training

The user resumed work on 7 November and approved two new scoring submissions:
one reproducible public-code trial (P13) and one qualified trained candidate.
P12 failed its hidden rerun and received no score. Preserve P01's verified0.480
pipeline as the production anchor. No automatic scoring submission is configured.

## Execution contract

`configs/frozen-training-20261107.json` and `scripts/frozen_retrieval.py` define
the immutable scientific run in `outputs/local/long-frozen-20261107-r1` (F07R1). Its manifest
pins the script, projection architecture, legacy metadata source and downloaded
encoder manifest. L06's completed artifacts are unchanged.

- RTX2070, at most70% of its VRAM allocator budget; two CPU threads, one interop
  thread, two Arrow threads, minimum fourGiB free RAM.
- A shared24-hour wall-clock cap started2026-11-07T21:42:42UTC. It covers
  preprocessing, encoders, pilot and both heads, including restarts. This is a
  maximum budget, not a predicted completion time.
- Official pretrained encoders run sequentially with batches of8. MoLFormer
  generates768-dimensional molecule features; DreaMS generates1024-dimensional
  spectrum features. Atomic512-row shards support restart; global feature arrays
  use memory-mapped FLOAT16 storage. The models are frozen.
- Raw competition peak arrays retain their m/z precision through FLOAT32 DreaMS
  preprocessing. The earlier FLOAT16 normalized peak cache is not used.
- The actual prepared population has273,833 catalogue structures,195,809 training
  identities and673,551 usable selected spectra (at most4 acquisitions/identity).
  Only supported positive single-charge adducts, precursor<=1000Th and at least
  three usable peaks enter this encoder. R1 removes fragment peaks above1000Th
  before top100 selection and relative-intensity normalization. Production must keep the baseline for
  unsupported inputs.
- The original input-equivalence groups and fold0 are excluded from training
  positives and negatives. Ambiguous shared-input identities are excluded.
  Previously evaluated identities and the NP panel are excluded. Fresh fold0
  development/confirmation sets contain256 identities each; checkpoints are
  selected on development only. Confirmation is read only after selection.
  These exclusions apply to learned heads. Public encoder pretraining may
  overlap these identities; the split is not a claim of unseen pretraining data.
- A5,000-identity/1,000-step pilot must improve over untrained projection heads.
  This is an execution/learning gate, not evidence of a competition gain.
- Two1,024-hidden/512-output projection heads use candidate contrastive loss,
  metadata/adduct inputs, AdamW, warmup/cosine decay and128 training-only
  mass-near negatives. Each seed targets50,000 optimizer steps. Seed202611071's
  first1,000 steps use the pilot population; seed202611072 starts on the full
  population. They are two candidates with different warmup data, not a clean
  isolated random-seed replication.
- The training diagnostic forces the truth into256 mass-near candidates.
  It does not measure real candidate recall or final-pipeline MRR. Production
  qualification, independent full-pipeline validation and real Kaggle smoke
  are required before using the reserved trained-model submission.

`scripts/supervise_frozen_training.py` checks health every30 minutes, notices
early process exit, resumes bounded retries from atomic caches/checkpoints,
checks source/plan pins and stops after repeated identical errors or three
restarts. It kills only its owned process tree. It cannot automatically repair
an arbitrary scientific bug; `needs_attention` reports that condition.

The intake audit found4,290 of673,579 preprocessed views (0.64%) contained
fragment m/z above1000, with a maximum3160.09, despite passing the precursor
filter. F07 was stopped before any head fitting. Its original source and
artifacts remain preserved. F07R1 adds the fragment-range filter, rebuilds raw
spectrum inputs, verifies identical encoder/molecule row hashes and hard-links
the exact completed molecule embedding cache. It keeps the original launch time
and deadline. No spectrum embeddings from the superseded preprocessing are
reused. The R1 manifest and receipts distinguish this correction explicitly.

The first child launch failed at the Torch DLL import, before scientific
preparation or optimizer work. Loading Torch before the NumPy/Arrow stack fixed
the Windows DLL conflict. The supervised relaunch at21:45:53UTC has the same
original deadline. The live dashboard reads its actual status; preparation
counts are separate from optimizer steps.

## Encoders and environment

- [DreaMS official code](https://github.com/pluskal-lab/DreaMS), revision
  `dbec3a0b514a99e5056cfccde4559fda8cfe8129`, MIT.
- Original embedding/SSL checkpoint files from
  [Zenodo10997887](https://zenodo.org/records/10997887), downloaded through the
  author's Hugging Face mirror revision
  `c81a62766b10dd1d39fcda3edec5ef88623e5f6b`. Both original Zenodo MD5 checksums
  matched. SHA256s and every supporting file are in the encoder manifest.
  The [original paper's spectral-similarity method](https://www.nature.com/articles/s41587-025-02663-3)
  uses a roughly5,500-molecule MoNA subset for contrastive embedding tuning;
  the official contrastive launch example names the MoNA-only dataset. NIST
  molecular-property/fluorine fine-tunes are separate and are not loaded here.
  The SSL config's NIST/MoNA probing entries describe frozen-feature monitoring,
  not gradient training of this backbone. The host's conditional official-release
  approval was rechecked in discussion748882 on7 November.
- [IBM MoLFormer-XL-both-10pct](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct),
  Apache-2.0, compat-v4 revision`7b12d946c181a37f6012b9dc3b002275de070314`.
  Public PubChem/ZINC pretraining; local-only load of pinned code/weights.
- [Official MSAlign](https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026), revision
  `97e201b1dd61e792d9086815192f7518a0b62ed8`, supplies the frozen-feature/candidate
  alignment design inspiration. This experiment is not its published benchmark
  reproduction. Its default gated molecular checkpoint is not used.

`scripts/fetch_frozen_encoders.py` records source revisions and hashes. The local
`.venv-encoders` shares the existing `.venv-gpu` Torch/CUDA package path while
overriding legacy encoder dependencies. It leaves `.venv-gpu` unchanged. The
validated encoder overrides include NumPy1.26.4, PyArrow17.0.0,
PyTorch-Lightning2.4.0, torchmetrics1.4.3, numba0.59.1, matchms0.27.0 and
Transformers4.49.0. Torch is2.11.0+cu128 on Python3.12.10. Original checkpoint
loading needs the author's `msml_legacy_architectures` classes and verified
legacy pickle loading; do not use that loading path for untrusted checkpoints.
The legacy class source is pinned at`d35b34b0caee38348098cda852bff30d8b25cd25`.
Encoder weights, vendor clones, feature arrays, environments and runtime logs
remain excluded from Git.

An8-spectrum real-data benchmark produced finite encoder outputs and measured
214 spectra/second with3.20GiB peak reserved VRAM for DreaMS. This small benchmark
does not establish full-dataset throughput or inference-time score stability.
The benchmark and intake/provenance receipts are saved under
`docs/evidence/frozen-2026-11-07/` as execution proceeds.

## Public mining and P13

[The public peaks-v2 source](https://www.kaggle.com/code/zz3t3rn41/casmi26-peaks-to-structures-v2)
uses the same public/private neural blend for outsider candidates and final
lists. P13 isolates that change on P01. It preserves the exact v80 checkpoint
SHA256`c08d2101154d057890de088be63834eb622468892a3b842512fdc5a65e02ff46`,
primary blend, priors, physical rules and official-identity deduplication.
It does not import new rank1 thresholds or arbitrary LOTUS fills. Duplicated
outsider rows align by candidate occurrence; known unavailable input channels
fall back to the available channel. Real execution errors remove the output
CSV instead of silently submitting an old baseline.

Version1 of `clarkkitchen/casmi26-p13-outside-blend` was pushed for real Kaggle
smoke on7 November. Before spending its one approved scoring slot, independently
audit400 output rows,12 real inputs, metric deduplication, checkpoint pin,
primary fusion and outsider fusion. A smoke proves execution on visible inputs;
it does not prove hidden compatibility or leaderboard improvement.

## Monitoring and restart

```powershell
.\training-monitor.cmd --terminal
.venv/Scripts/python.exe scripts/training_monitor.py --web --open
# Only if no existing supervisor/trainer is alive and the contract is unchanged:
.venv/Scripts/python.exe scripts/supervise_frozen_training.py
```

The monitor selects F07 automatically and remains read only. It shows stage
counts, head steps, GPU/RAM and watchdog health. Its head ETA appears only after
all features are cached and real full-training timing is available. The24-hour
deadline always remains a separate budget cap. Closing the dashboard keeps
training running.

## P13 submitted

Version1 finished the real Kaggle smoke in21.4 minutes. Independent checks found
400 valid rows,12 real inputs, zero official-identity duplicates or input errors,
primary scores exactly matching P01, primary fusion error4.44e-16 and outsider
fusion error2.08e-7 across240 candidates/12 queries. No rankings changed on these
12 inputs. The checkpoint pin and physical merge matched the declared recipe.

P13 v1 was submitted once at2026-11-07T22:04:58UTC, ref58105681,
marker`F07-P13-20261107`. Its score is pending. One of the two approved scoring
slots remains for a qualified trained candidate. No automatic second submission
is queued. Evidence is in`docs/evidence/frozen-2026-11-07/p13_*`.

R1's completed intake audit confirms195,809 training identities, fresh256/256
validation identities,5,000 pilot identities, zero head-training input-group or
negative-identity leaks, and673,551 finite usable spectra. All retained fragment
peaks are below1000Th (maximum999.995). The raw-range filter removed out-of-range
peaks from6,237 selected acquisitions;28 selected acquisitions no longer had
three usable peaks and are excluded. Molecular features are byte-identical to
the completed initial cache (SHA256 recorded in the reuse receipt).

The R1 pilot completed1,000 steps in16.36 seconds, with development diagnostic
MRR@25 rising from0.021024 (untrained) to0.125639. This clears only the learning
pilot. Full spectrum encoding then started; the remaining654,526 feature rows
are being cached before both50,000-step targets finish. Early throughput suggests
roughly18:30–19:00 Eastern for training, subject to real full-training timing.
Validation, packaging, Kaggle smoke and scoring take additional time.
