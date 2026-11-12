## CURRENT C76 R1 — full400row interface PASSED locally and actual cloudCPU

CURRENT COMPLETE BINDING is C76/complete-pipeline-binding-r1.json SHA
0ddc4d46d9dbcb2f8804f3970d6763c38528311dcd24dfe4a704a91df2a36518.
Current scripts/complete_c76_target.py, collect/release_c76_source_controls.py,
evaluate_c76_source.py and audit_c76_source_quality.py now reference R1binding.
Quality plan is source-quality-plan-r1.json; production plan is
production-evidence-plan-r1.json. Canonical future production helpers are
release_c76_production_smokes.py, collect_c76_production_smokes.py,
audit_c76_cloud_parent_pair.py. Weights/core/source cohort/gates unchanged.

NO LOCAL ML JOB RUNNING. Original production toolsession35815 CLOSED1 at20:36UTC;
scope30208inactive. Core/P16 executed completely:12queries/all41acquisitions,
Engine2/PC12each, mainerrors0, ICE1611predictions0fails, GL11of11supported healthy,
original P16fixture30queries/113acquisitions/7changed lists passed. NewR0HEAD then
failed because raw NN evidence correctly contains ALL400answerIDs including388
unprocessed placeholder rows; guard wrongly demanded only12activeIDs. Original
failure and wholecore/nativeworking preserved. Do NOT claim originalR0fullrunpassed.

R1fix is ONLY three raw/public/ours filters by complete set of IDs WITH SPECTRA;
NO health/outcome-based filtering, NO activequery/candidate/acquisition dropped,
NO model/refit/scoring math change. All400answerrows/388inactivefallbacks retained.
Old runtime/production/operational sources (62files) archived byteexact under
C76/interface-repair-r1/previous-sources; R0publishedbundle and bindingJSON retained.
Canonical paths advanced to R1; the oldR0binding no longer verifies CURRENTsource.
Use archive/Git to reproduceR0; never silently pass oldC75orR0failed gates.

New private R1bundle READY:clarkkitchen/casmi26-c76-complete-direct-pair-r1-20261112.
ManifestSHA b0e9c31eb0694996245b4a7fc24aca1ca56a7d64bda15bafcac5c7a5198206e4;
repaired score sourceSHA d9112f4cb17d43a533d7453df9789ee4a912eaa12fb034b28fd4ee90f0271c99.
Current production notebook folders keep c76-cpu-production/c76-gpu-production;
new canonical KERNELids end -production-r1. Only last head manifest pin and
metadata/bundle changed; full0:27core/P16prefix is BYTEEXACT. All oldsourcebytes
and models retained. No actual T4 production notebook has been pushed.

HEAD-ONLY recovery succeeded using PROVENunchanged core/P16prefix; GPUcore NOT
restarted. New CPU/GPU stages both12changed tail lists,0leader changes;400rows,
388inactive rows EXACT, sameparent/features. All old all-active12query/51acq
features/scores/lists BITEXACT; deliberate unhealthy ACTIVErow was rejected before
output. First recovery harness lacked json in restoredglobals and stoppedbefore
head inference; originalattempt preserved. Sole corrected retry session3754 CLOSED0;
independent audit session67412 CLOSED0. Prefixaudit57125 CLOSED0. No blind requeue.

Independent composedR1audit PASSED; it explicitly records R0runFAILED, recovered
prefix+headsPASS. Physical rule independently rebuilt on7REAL USEDreference
promotions:7consistent kept; separate counterfactual reference views verify all7
unknown-adduct and +10Da inconsistent promotions revert to exact30originallists.
No original refs changed. Source C76/interface-repair-r1/independent-run-audit.json.

ACTUAL cloudCPU r1job COMPLETE v1:
clarkkitchen/casmi26-c76-r1-cpu-head-parity . Exact original image/source/GPUdisabled
verified. All101features/candidate records BITEXACT,300scores PERHEAD maxdiff0.0,
ALL400final lists exact including388fallbacks. No expected head predictions or
query truth mounted. Collector/auditor source scripts/c76_r1_cloud_head.py;
proof C76/cpu-head-parity-r1/independent-comparison.json. Audit67088 CLOSED0.
New private400row input package READY. Original active-only cloudproof retained.

R1source/production guards tested: missing actualT4legacycollection stops before
C72labels/consumption; missing sourcequality stops before T4production API/intent.
Current fullbinding, sourcequality and production evidence plans refrozen BEFORE
any freshC72source prediction/outcome; no gate/tolerance/recipe/cohort rescue.

T4quota actual21:08UTC API still refresh2026-11-14T00:00UTC=Nov13 7pmEastern.
Current release_c76_source_controls.py push-legacy is HELD without pushing before
that reset. Sourcejobs dictEMPTY. Original C73version0quota rejection is NOT a job.
After reset: push-legacy; collect_c76_source_controls.py fetch/audit legacy;
then push-c76 and fetch/audit c76. Then complete_c76_target.py replay both arms,
score, seal all6controls+bothheads. Evaluate ONE joint source via
scripts/evaluate_c76_source.py run, then independent source quality audit.
Actual T4source/production remain REQUIRED; localbatch8 never substitutes for
T4batch96. Actual T4 production audits independently replay SAMECLOUD headinputs
locally and require identical actualT4parents across both separate head runs.
Old C75fullcorecross-device gate staysFAILED; no 2070fullcore equivalence claim.

Remaining independent preparation: C13compatibility/composed confirmation wrapper
and final scoring qualification/terminal-score plumbing for this R1family. Do NOT
run oldC59/C60/C43release scripts: they bind retired families. Existing untouched
C13 may be consumed at mostonce ONLY after R1source and actualproduction checks;
retain all120incl trailing-space formate fallback; no substitution for targetsource.

C72/C13UNCONSUMED, source-results and jointfreezeABSENT, sharedC42scoringledgerABSENT,
0/2newscored; champion0.480 preserved. Old/hourly controllersOFF; freecomputeonly.
GoalACTIVE; this turn made substantial concrete production/debug/repair/CPUcloud
progress, not blocked/no-progress. Historical checkpoints below superseded.

## LIVE C76 actual production preview — session35815

Actual owned RTX2070 production preview RUNNING. Tool session35815; Windows
supervisor30208; systemd user scope enveda-c76-production-preview-30208.scope.
Linuxdriver416/PGID416. Reconcile /proc/cgroup before relying on oldPIDs.
Read outputs/local/c76-numerical-portability-20261112/local-production-preview/
run-progress.json, supervisor-status.json and pipeline.log (filter longSVG lines).
Inner log native /home/imsa/.local/share/enveda-c76/production-preview/working/
submission_pipeline.log. No automatic retry. Current inner32Engine2; actual
source input/preflight and first12public-query smoke selection already passed.

Frozen launch plan SHA3200f6e4c16c1a14545884d73b6801df9815a22892a05db9fb11dee41cbf2197.
Complete original production recipe port changes ONLY local allocation checks,
resolved owned preflight path, recoveryCUDA1->0, same explicit ICEcache/bs8 policy,
honest payload hash, and distinct GPU-head artifact names. Original 12query/
41acquisition preview selection/all41acquisitions retained;400answer rows,388
inactive fallbacks. CPUthenGPUfixedheads share exactly one complete parent run.
Original P16 real30query/113acquisition/7changed-list fixture stage retained.
No fresh labels, newheadfit, quality evaluation, scoring or actualT4claim.

Source/asset/preview audit PASSED before launch. Initial directory links hid
assets from Path.rglob; unused layout preserved as native input-layout-attempt01.
New real directories expose675links to the EXACTsame original files. Signed
input-layout-amendment and link map retained; no filebytes changed. Initial
sourceauditor reversed a payload literal with different assignment spacing; old
bytes preserved; exact original spelling restoration fixed the auditor only.
Every science/asset/tolerance requirement retained; no inference before repairs.

Namespace /kaggle/input NOWpoints native .../enveda-c76/production-preview/input;
/kaggle/working NOWpoints .../enveda-c76/production-preview/working. Only both
verified owned links switched by their existing root owner; original C75/C76
inputs/workings retained, no permissions or security policy changed.

Resource: original matchedTorch2.10WSLenv;4workers/2threads; high8.75/max10/swap4GiB,
hostfloor4GiB. Current RAM~6GiBfree, swap~2GiB; scope remains active. Do not stop
other applications or alter globalWSLmemory. Scope guardian handles reserve.
After THIS run succeeds execute scripts/audit_c76_local_production_run.py under
matchedWSLruntime. It checks fullsource/41acquisitions/core health, both separate
heads/388inactivefallback rows, original P16fixtures, and independently reconstructs
physical mass filtering on seven REAL reference promotions from that USEDfixture.
No runtime audit has run yet; execution success/quality unproven.

Main complete family binding remains04b47cac... unchanged. T4sourcegate held for
Nov14 00:00UTC=Nov13 7pmEastern. Old source rejectionversion0 is not a job. C72
and C13UNCONSUMED; C42scoringledgerABSENT;0/2newscored; champion0.480retained.
GoalACTIVE: concrete local production progress this turn. Old/hourly campaignsOFF.

## Latest production input readiness — no GPU inference running

2026-11-12: actual Kaggle API test.parquet and sample_submission.csv fetched into
C76/production-inputs. Independent WSL audit PASSED:400queries/all1213original
acquisitions;276positive/97mixed/27negative; exact400sampleIDs; no teststructure
labels; testbytes EXACT with previously audited C37 unlabeled public test.
Source scripts/prepare_c76_production_inputs.py and compact public receipt frozen.
This is ready for a bounded LOCAL production preview, not a source-quality check.
No production preview has launched yet. Namespace pointers are still C75fixture
input and C76fixture-resource working. Preserve all old data before new own links.
Next local port must explicitly audit actual2070allocation, singleGPUrecovery1->0,
local ICEbs8/cache policy, production input/reference configuration, same full
C76 scientific pipeline/P16/sharedheads and real inactive-row fallback. Required
actualT4source and production evidence remain unchanged; cannot claim localT4.

C72/C13UNCONSUMED; scoringledgerABSENT;0/2scored. GoalACTIVE. Main completepair/
source/control/quality work committed and pushed as64d8a43. No paidresources,
no old/hourly campaign restart. Historical checkpoints below superseded.

## Current C76 checkpoint — complete pair frozen; cross-host deployment PASSED

Complete binding SHA04b47cac8fe5476613834d4214d53cb51b7cc7a81162bf9516d2119acf7a03b9
is outputs/local/c76-numerical-portability-20261112/complete-pipeline-binding.json.
It pins new full source notebook, both explicit R3 production notebooks, unchanged
C71 CPU/GPU weights, full P16/physical/ranking code, original C72/masks/controls.
No model refit or fresh outcome read. Never change these files silently.

Both production SOURCE audits pass. Production reverses only validation input/
reference wiring, keeps exact R3 empty-pool recovery, BOTH Engine2 scalar and
fingerprint formate normalization, and both C76 GL paths. Retired C56 parent is
NOT shipped. Three packaging guard failures were preserved before any inference
or remote release. See compact reconciliation; raw partial bundles remain ignored.

Real USED12/all51 P16/physical/shared head exercise PASSED at19:42UTC. Both heads
changed all12tail lists, no leader changes, exact independent arithmetic/list
reconstruction; membership/formula slots preserved. P16 had0promotions on this
masked USED fixture; do not claim it exercised a changed physical-reference path.

Actual private CPU cross-host job COMPLETE v1. Canonical handle returned by Kaggle:
clarkkitchen/casmi26-c76-cpu-complete-head-parity (requested id had no 'complete';
DO NOT requeue the old requested id). Source/image/GPUdisabled verified. All101
features and candidate records BITEXACT; all300scores PERHEAD maxdiff0.0; both
final list sets exact. Auditor session30469 CLOSED0. Full fixture session92177,
N1session45242 and local complete-head session42142 CLOSED0. No ML job live.

Private frozen pair bundle and USED-only head inputs published READY. No query
truth or expected head predictions mounted. Original MLenvs unchanged. Generated
bundle import cache was moved into ignored owned CPU-probe archive before upload;
all signed payload files preserved. CPU engineering probes are NOT scored entries.

One-time source evaluation code/policy/independent auditor frozen in
source-quality-plan.json. Guard test PASSED: missing actual legacyT4 collection
stops before labels or C72 consumption. Do NOT run evaluate_c76_source.py run
until BOTH actual T4 source arms are collected/audited and both complete parents,
all6controls and both candidate predictions are jointly sealed.

T4 source launch helper: scripts/release_c76_source_controls.py. push-legacy is
HELD before known quota refresh; no new cloudGPU push attempted. Actual20:00UTC
quota snapshot retains refresh2026-11-14T00:00UTC=Nov13 7pmEastern. Legacy C73old
request is version0 with explicit30hquota rejection; it is NOT a job to poll.
After reset: launch legacy exact notebook; collect/audit with
collect_c76_source_controls.py fetch/audit legacy; then launch c76 and collect/
audit. complete_c76_target.py replays each arm, scores both fixed heads, then
seals all predictions. Never tune cohorts/recipe after source outcomes.

Remaining: actual T4 source controls; single C72 quality plus independent audit;
actual new production changed-path smokes/local-cloud list audits; only then the
existing once-only C13 if defensible; sequential CPU then GPU real scoring.
Potential independent work before T4 refresh: prepare/audit a bounded local
production preview using the frozen new notebooks and actual unlabeled production
inputs, with explicitly separate2070 resource wiring; it cannot replace real T4
source/production evidence. No extra blind GPU training.

C72/C13 UNCONSUMED, source-results absent, joint source freeze absent, sharedC42
scoring ledger ABSENT, new scored entries0/2. Champion0.480 preserved; old/hourly
controllers OFF. Goal ACTIVE: this turn made concrete engineering, deployment,
freeze and tooling progress. No paid resources. Historical notes below superseded.

## Current C76 checkpoint — complete local engineering audit PASSED

The complete USED12/all51-acquisition resource run finished 2026-11-12 19:22 UTC
in 992 seconds. Independent audit at 19:26 UTC passed all input, reference,
core-health, GL adapter and official-list checks. ICE produced 2204 predictions,
zero failures. The separate real original N1 CPU proxy/merged-CE path passed all
32 outputs. No inference job is still running; prior sessions 92177 and 45242
closed successfully. The earlier oversized-batch run remains intentionally stopped.

The successful local resource policy remains cache release before ICE and --bs8;
no claim of exact ICE parity across batch sizes. Original C75 cloud-reproduction
gate remains FAILED. This new C76 engineering pass is NOT statistical qualification.

Prepared private actual-T4 source notebook:
kaggle/forks/c76-full-masked-validation/ (NOT pushed). Independent source audit
passes: reverse only the explicit GL package integration and recover exact C73.
Original ICE default, source datasets/image and full512/all2215 panel retained.
Original C73 remains a separate required legacy control. Next freeze complete
C76 P16/physical/head binding and controls before any source quality read.

C72/C13 UNCONSUMED; shared C42 scoring ledger ABSENT; new scored entries 0/2.
Champion 0.480 retained; hourly/old campaigns OFF. Actual T4 quota refresh recorded
as 2026-11-14 00:00 UTC (Nov13 7pm Eastern). No paid compute.

Historical checkpoints below describe prior states; this checkpoint supersedes
all earlier claims that the C76 fixture or N1 audit is still running.

## ACTIVE C76 resource retry — prior oversized-batch run intentionally stopped

NEW LIVE TOOL SESSION92177. Windows supervisor29108; unit
enveda-c76-fixture-resource-29108.scope. Linux driver368/PGID368 at launch;
reconcile currentPIDs from run-intent (WSLcan reinitialize between jobs).
Artifacts/status: outputs/local/c76-numerical-portability-20261112/fixture-resource-port/.
Read pipeline.log, supervisor-status.json and run-progress.json. Guardian high8.75/
max10/swap4GiB/hostfloor4, one controlled launch, no automatic restart.

The old session40251 is CLOSED1 after INTENTIONAL owned-scopeSIGTERM (return15).
All old C76 fixture-port artifacts and nativeworking data preserved. That run is
NOT complete, NOT a model-quality rejection. A read-only profiler scalar sample
showed ICE remained in FIRST96item batch (i96, nb96, depth2of3, maxnodes100) after
>30minutes, physicalGPU~7970of8192MiB with sharedGPUmemory. Fullinput was EXACTLY
identical to prior successfulC75ICE2204prediction run. Softlimit8->8.75 stopped
throttling but did not solve performance. Only the verified C76scope was stopped;
other applications retained. Private profiler-locals JSON stays ignored; only
whitelisted scalar counters/hash are in public evidence.

New explicit LOCAL resource adapter is frozen in
configs/c76-local-resource-adapter-v1.json: gc.collect/torch.cuda.empty_cache
immediately BEFORE separateICE, and original supported ICE --bs8. No model,
query, candidate, reference mask or scoring-function change. ICE floating-point
parity across batch sizes is NOT assumed; actualT4source/control/production gates
remain. Original C75 strictgate stays failed. Independent resource sourceaudit
PASSED, verifies ONLY those2newchanges against the prior C76notebook. GLhelper,
checkpoint and both classic/N1GL routes retain their previously proven bytes.

Namespace /kaggle/working NOWpoints
/home/imsa/.local/share/enveda-c76/fixture-resource/working.
/kaggle/input remains originalC75fixture inputs. GLpackage still
/home/imsa/.local/share/enveda-c76/fixture/glacier (frozen owned clone). Root-owner
activation switched ONLY verified taskworking link; no chmod/security/data change.

After THIS new complete run succeeds, use
scripts/probe_c76_resource_n1_path.py run in scope containing
enveda-c76-resource-n1-path (CPU2threads, suggested3GiBcap, startreserve8GiB).
Its input is BYTEIDENTICAL to earlier prepared N1case; never previously executed.
It requires new fixture-resource-port/run-complete.json (not the stoppedoldport).
Then execute scripts/audit_c76_resource_fixture_run.py under recoveredTorch2.10
WSLruntime. Both scripts compiled, not run yet. Prior non-resource N1/auditor are
retained historical plans and MUSTNOT be used against the stoppedoldrun.

Next requires a new complete C76binding/P16/physical/head adapter and frozen
legacy-plus-C76controls before any fresh512quality. Never call oldC75paritypassed
or reuse retiredC56production parent. C76GL799score parity and actualcloudCPU26
array checks PASSED already; do not repeat unless a new relevant change warrants.
New CPU/GPU headweights unchanged. C72/C13UNCONSUMED, sharedC42ledgerABSENT,
0/2new scored entries, champion0.480 retained. GoalACTIVE; old/hourlycontrollersOFF.
Quota refresh2026-11-14T00:00UTC=Nov13 7pmEastern. No paid compute.

## Live C76 resource update — ICE still running; no qualification yet

Complete C76 USED12 job remains toolsession40251, scope
enveda-c76-fixture-19960.scope, Windows supervisor19960, Linuxdriver426,
mainpipeline499 and ICEchild803. ICE input is BYTEIDENTICAL to prior C75successful
run (all12candidate orders/spectra exact). EarlierTorch2.11ICE2204predictions took
409.83prediction seconds; current recoveredTorch2.10ICE is substantiallyslower.
Native and nonblockingPython profiler reads of ONLY ownedchild803 show CUDAwait
and changing graphforwardframes (_seg/AvgPooling -> FragGNN.forward339). No
model/source edits or other-app termination. No completed result claimed.

Measured memory.high throttling and~0.86GiBswap prompted an explicit TASK-ONLY
soft-limit change8->8.75GiB. Hard10GiB/swap4GiB/hostfloor4GiB and liveguardian
unchanged; no OOM observed. Authoritative override receipt:
C76/fixture-port/memory-soft-limit-adjustment.json. Original supervisorstatus
still repeats historicalhigh8; read the override and actualcgroup. Base/newML
envs unchanged. Isolated diagnosticsvenv holds onlypy-spy0.4.3; systemensurepip
was absent, so existingpip was run with --python explicitly targeting diagnostic
prefix (verified). No pip installs into the MLenvs, no security-policy changes.

After the complete fixture exits successfully, run the prepared realCPU N1path
exercise using scripts/probe_c76_n1_path.py run in owned enveda-c76-n1-path scope
(suggest cap3GiB,2threads,startreserve8). Synthetic proxy/adduct/CE input frozen
already;4candidates,4spectra,2mergedCE,expected32finite outputs. It may NOT overlap
the completefixture per its predeclared guard. Then run new
scripts/audit_c76_fixture_run.py under recoveredWSLruntime. Auditor is compiled,
NOT executed yet. Full core/acquisition/reference/GLadapter/identity checks and
bothGLpath evidence required; no oldC75reproduction acceptance or queryquality.

configs/c76-source-controls-addendum-v1.json strengthens future source comparison:
retain original unrepairedR3P01/P16/physical controls plus C76counterparts, freeze
all predictions before the single originalC72qualityread; original actualT4
deployment evidence cannot silently be replaced by different local2070rounding.
No fresh512predictions, C72/C13consumption, or scoring requests made. GoalACTIVE.

## Current C76 checkpoint — complete USED12 replay RUNNING; 0/2 scored

Goal ACTIVE. Champion0.480 preserved. C72 fresh512/all2215acquisitions and C13
remain UNCONSUMED; sharedC42two-entry ledger absent. Old/hourly controllers OFF.
Actual paginated Kaggle account audit:35records,34terminal numeric results,
NO pending submissions, best0.480. No new scoring request. T4quota refresh
2026-11-14T00:00UTC=Nov13 7pmEastern; no paid resources.

Breakthrough: unchanged GL CPU/CUDA raw mass arithmetic is amplified by 4decimal
binning and peak merging. Pinning fragment decisions alone leaves13of16score
differences; pinning decisions AND masses removes all, max~1.335e-7. Simple stable
sort and math-attention changes failed. Float64mass reduction alone still fails4.
CPU/GPU node embeddings differ~6e-6 but breakpoint choices can change. Do not
claim one universal cause or call the original C75 exact-reproduction gate passed.

C76 is an EXPLICIT NEW numerical candidate over fixed C71 heads: same original
learned encoding/fragment functions and weights always CPU (2threads), intensity
network still requestedCPU/CUDA, existing atom-mass summation promotedfloat64.
The16case predeclared pilot PASSES:16exact massbins/0different patternbits,
maxscore3.601e-8. Extended frozenUSEDnine input: BOTHhealthy9of9/2131jobs,
ALL799scores within original1e-5, zero nullmismatches, max9.853e-8. Private exact-
image C76CPUprobev1 COMPLETE, actualsource/image/GPUdisabled verified; all26feature
and predicted-spectrum arrays BITEXACT with localCPU. No fitting or quality read.

Retrieval remains a separate unresolved original-cloud discrepancy: matched
runtime Engine2 all12lists exactlymatch priorlocal, only1of12matchC57cloud;
all12leaders same. A private actualCPU raw-sort probe reproducedall48sort/blend
arrays bitexact acrosshosts. Packageversion/sorting hypotheses did not repair it.
Original C75 evidence remains unchanged; do not assert all parent parity solved.

RUNNING complete C76 USED12/all51 replay: toolsession40251; supervisor PID19960,
scope enveda-c76-fixture-19960.scope; Linux driver426/PGID426. Status/log under
outputs/local/c76-numerical-portability-20261112/fixture-port/supervisor-retry-*
and pipeline-retry.log. One controlled inference, high8/max10/swap4GiB; hostfloor4.
Namespace /kaggle/input stillC75fixture inputs; /kaggle/working NOWpoints
/home/imsa/.local/share/enveda-c76/fixture/working. Oldworking data intact.
GL clonedpackage native .../enveda-c76/fixture/glacier; all original payload
entries read-onlysymlinks except explicitloader and pinnedhelper. Innercell41
changes only GL_PKG path and honestnewrunnerSHA; BOTHclassic/N1route throughsame
loader. Independent sourceaudit PASSED. Runtimehealth/acquisition/branchcoverage
and complete local/C71candidate audits still REQUIRED afterrun; no fresh release.

Operational failures preserved: firstextended attempt missed optionalCLIcheckpoint
arg and stoppedbeforeinference; fixed/archived, sole retry passed799. Firstfixture
attempt stopped beforeinference on root-owned /kagglelink permissions. Verified
tasklink switched using same root-owner pattern as prior C75; no chmod/policy or
data changes, archive retained; retrynowrunning. Do not restart blindly.

Old C71 CPU/GPU modelbytes unchanged. C76 config/freeze-r1 explicitly requires
complete pipeline qualification, same fixedC72protocol and later actualKaggle
production smoke. Never reuse retired C56parent or quietly relax oldC75gate.

## Current C75 investigation — GPU numerics localized; retrieval replay running

Goal ACTIVE, 0/2 actual scored submissions. C72 fresh512 and C13 remain UNCONSUMED;
shared C42 ledger absent, champion0.480 retained, old/hourly campaigns OFF.
Original strict USED12 parity still FAILS. No tolerance or final-list guard changed.

Exact-image CPU feature probe COMPLETE private v1 (casmi26-c75-cpu-feature-parity).
Original checkpoint/package hashes and all26feature/prediction arrays BITEXACT
between KaggleCPU and localCPU on4fixed public USED candidates. All38 scientific
GL payload files match their published manifest (only publication metadata absent).

Fixed16USED-job device test: CPU13scores outside original T4 tolerance, max.01768845;
2070four outside, max.00077075. Valid root/node embeddings differ only~3e-6/6e-6,
amplified to valid breakpoint-logit max.004089 and36different fragment pattern rows.
No invalid padded atom selected; rawmax1.107 was mostly padding. Stable projected
or raw-logit sorting FAILED to fix. Enforced math SDPA FAILED too (four outside).
Both stock paths reproduced previous scores exactly. These are engineering
counterfactuals, never promoted models. Original GL/C71/C74 science unchanged.

Raw-output mass substitution hypothesis stopped at its alignment guard: case2
CPU4472rows vsGPU4368rows. Raw arrays preserved; no sole mass-rounding claim.
Prepared fragment-decision counterfactual to separate selection and arithmetic;
not launched while retrieval replay holds memory. All probes use same frozen
16USED jobs, no truth labels or fresh quality.

RUNNING: one matched-cloud-runtime original Engine2 replay on sameUSED12queries
and reference mask; recovered Torch2.10/NumPy2.0.2/Numba0.60/sklearn1.6.1.
Source scripts/probe_c75_retrieval_runtime.py, supervisor scripts/supervise_c75_retrieval.py.
Scope/status/log: outputs/local/c75-local-core-port-20261112/retrieval-runtime-probe-v1-supervisor.*.
Parent engine code copied unchanged except diagnostic audit-output filename.
Original public fit arrays only; captures original returned channels for diagnosis.
Host reserve4GiB; scope high8/max10/swap4GiB; one launch, no automatic retry.
Tool session31563; scope enveda-c75-retrieval-30680. Reconcile before new heavy job.

Attention probe finished, tool81726 closed0; no other running diagnostic expected.
T4 quota refresh2026-11-14T00:00UTC (Nov13 7pmEastern). Free CPU probes work;
no scoring entry consumed by metadata/feature probes. Current turn made progress.

## Latest C75 checkpoint — NO active inference; exact parity remains unresolved

Goal ACTIVE, submissions0/2used; NOsharedC42ledger, NO C72/C13consumption.
Do not call complete/blocked: this goal turn made substantial concrete progress.
NewC71models and entire C72fresh512panel remain frozen; no fresh predictions or
quality metrics produced. Champion0.480retained; hourly/oldcampaignsOFF.

RAM problem solved: complete USED12/all51P01/P14 replay finished safely.
Exact parity still FAILS (9of12P01lists,10of12P14lists differ; all leaders same).
Original individual public/ours neural numeric evidence within1e-5 on295common
candidates; earlier Engine2/PC/GL evidence differs. No fitted C71head applied.

Runtime investigation completed, NOT a successful fix: CPU-only private probes
casmi26-c75-original-runtime-metadata and casmi26-c75-original-runtime-inventory
both COMPLETE v1 in EXACT original C57Dockerimage37c64f7d...; actualGPUdisabled
and actual remote source verified. Full package inventory private/publicaggregate
saved. Real original runtime Torch2.10+cu128/NumPy2.0.2/pandas2.3.3/scipy1.16.3/
Numba0.60/llvmlite0.43/Arrow24, sklearn1.6.1/LGBM4.6. MainRDKit2026.03.3installed
by notebook; original simulator2025.03.6isolated.

New isolated runtime /home/imsa/.venvs/enveda-c75-torch210 is COMPLETE with real
CUDAmatmul, original major numerical versions plus graph/optional deps matched:
networkx3.6.1, multiprocess0.70.16, platformdirs4.9.6, dill0.4.1, h5py3.16.0,
joblib1.5.3/Pillow11.3/einops0.8.2. Read-only baseCPU fallback .pth is ACTIVE only
in NEWenv at lib/python3.12/site-packages/c75_readonly_base_cpu_fallback.pth;
original2.11andWindows envs unchanged. Disable ONLYthisownedpth before future
pipinstall into NEWenv, to avoid exposing base packages to pipuninstall.

FOUR controlled same-original-nine-input GLprobes all completed9of9. ALL799raw
output scores and outputfileSHA are BITEXACT across Torch2.11/baseCPU, Torch2.10/
baseCPU, exact major numerical stack, and exact graph/optional stack. All four
rawoutputSHA6daff17a32f81fb70417b7cb08aa0d6d6cc0acc46dec2bf9289ca0d45e331dde.
Against actualC57rawGLout:410of799beyond1e-5, max0.007375944; original C57rawGL
versus C57featureGL is EXACT (mapping is correct). Thus tested dependency
versions do NOT explain this GL difference. Never claim runtime mismatch fixed.

A WHOLE original433GLinput replay under matchedstack also diverged: at96
completedmolecules/7610comparedscores,2856beyondtolerance, max0.0294033. Stopped
ONLYits verified owned systemd user scope intentionally at16:39UTC because
whole-parity was already falsified; furtherGPU could not rescue completedscores.
Not a RAM/OOM failure. Preserve gl-full-context-probe-v1 partialoutputs/logs/
early-rejection/failed/source-original. Tool70191closed1 after intentionalSIGTERM.
The scopeguardian source had a missing .scope suffix in exceptional cleanup;
fixed ONLY AFTER job stopped and prior source bytes preserved. No restart.
All GPU/pip/probe/fetch jobs FINISHED; no live tool sessions to resume.

Next investigate actual model/checkpoint/source SNAPSHOTS and CPUfeaturizer
hashes against the original container, rather than more blindGPU/runtime tests.
Local assets were fetched by latest slug and checked against C20names/sizes and
internal manifests, but original C57 attached versions are not yet proven. This
is a HYPOTHESIS, not a diagnosed cause. SDK get_kernel metadata omits versions
(datasetDataSources are bare slug strings). Both nestedGLraw outputs are fetched
under original-cloud-raw-forward; inputs/E2/PC/e004 data under original-cloud-
intermediates. Public/ours pinned checkpoints agree, while unpinned reproduced
core channels differ. Compare exact artifact/checkpoint/source hashes before
any fresh validation or long training. No passing fixture-parity-audit exists,
so C75full512source release is still HELD.

Computer-use skill/read mandatory docs were applied. CUA initialized; IAB2
privateC57page404, Chrome4createdaboutblanktab1615613217 but attachment failed.
NativeChrome window3409530 could not be read because helper could not determine
current URL confidently enough for policy; ComputerUse STOPPED FOR THIS TURN.
NO furtherUIattempts; completion_summary already sent exactly once. No private
dataset versions obtained. Next turn must rewriteDocumentation before anyCUA
continuation. Do not circumvent this browser-policy stop with shell/UItools.
API/filesystem research remains authorized independently; no permission question.

Actual quota refresh2026-11-14T00:00UTC=Nov13 7pmEastern; T4quotaexhausted.
NoC73cloudjob exists(version0rejected); two CPUmetadata jobs completed, not
competition scoring. Shared2entry submission allowance unchanged.

## C75 current — RAM solved; USED parity rejected; exact cloud runtime recovered

The single2070 full USED12/all51acquisition retry COMPLETED16:01UTC, no
RAM-floor stop or OOM. Engine2/PC/main12each; mainerrors0. Actual ICE9of9
supported and GL9of9supported complete; zero prediction/feature failures.
Original frozen scripts/models/queries/masks preserved. Supervisor session38194
closed0. First stopped attempt remains preserved. NO heavy inference running.

Exact parity FAILED, so fullfresh512 MUST REMAIN HELD. P01exact3of12, P14exact
2of12; all12leaders identical, candidate membership identical8of12;5candidate
keys differ out300. Among295common rows: E2rank64differences, GL101beyond1e-5,
ICE1; public/ours individual neural numeric fields within1e-5. NewC71heads were
NOT applied. C72freshquality and C13 remain UNCONSUMED, submissions0/2, C42ledger
absent, champion0.480retained, old/hourly controllersOFF. Not a model quality
rejection: diagnose runtime/input parity before any new source evaluation.

Corrected an audit log-reader bug: original health checker needs outer stage
summaries plus inner channel counts. ALL assertions/tolerances unchanged.
Original auditor bytes and original operational Plan preserved under
C75/audit-log-repair-v1; current operational Plan has explicit linked audit-only
revision. No models/scientific source/cohort/reference changes. Corrected audit
passed health/fidelity before failing exact final list comparison.

Retrieved original C57 intermediates (USED only). Original acquisition inputs
for ICE/GL exact; E2lists exact1of12, PCcandidate order10of12, PCfloat scores
differ. TWO small GL tests completed: repeat exact localinput792scores BITEXACT
(maxdiff0); replacing with original cloud9candidate input still101differences
(max0.007375944) versusT4. Thus unstable randomness and candidate input selection
do not explain the GL differences. No new training or label/quality read.

A PRIVATE CPU-only metadata probe COMPLETED in EXACT immutable C57Dockerimage
37c64f7d..., metadata verified GPUdisabled, no datasets/competition mounted.
Kernel clarkkitchen/casmi26-c75-original-runtime-metadata v1. Actual original:
Python3.12.13, Torch2.10.0+cu128, NumPy2.0.2, pandas2.3.3, scipy1.16.3,
sklearn1.6.1, LightGBM4.6.0, numba0.60.0, llvmlite0.43.0, Arrow24.0.0,
cuDNN9.10.2.21, cuBLAS12.8.4.1. Main RDKit2026.03.3 is installed by original
notebook into its own site; simulatorRDKit2025.03.6 remains isolated.
Local C75base used Torch2.11/NumPy2.2.6/pandas2.2.3/scipy1.15.3/numba0.61.2/
Arrow19.0.1, so exact numerical environment was not matched.

LIVE: one isolated Torch2.10 install, toolsession11252, systemd user scope
enveda-c75-torch210-install (MemoryHigh2G/Max4G/SwapMax2G), target
/home/imsa/.venvs/enveda-c75-torch210. Downloaded official CUDA128 packages;
pip at installing stage16:21UTC. Original2.11/Windows envs UNCHANGED. Reconcile
completion before another pip command. No NumPy/pandas etc yet in NEWenv.
Prepared scripts/probe_c75_torch210_gl.py, NOT RUN. Next: afterinstallcomplete
add a clearly recorded read-only baseCPUsite fallback .pth to NEWenv only,
verify torch comes from NEWenv2.10 and realCUDA works; run ONE original-cloud9
GLprobe (same inputbytes) under newenv in 4G/6G/2Gscope. If needed subsequently
remove only that owned .pth BEFORE pip installs matched numerical stack into
NEWenv; preserve first test evidence. Do NOT mutate the frozen baseenv.

Actual Kaggle quota API lists refresh2026-11-14T00:00UTC (Nov13 7pmEastern),
GPUtime used exceeds allowed; no T4job exists. CPU probe works. Keep GPUquota
limitation separate from local scientific/runtime progress. Retired C56
production first28cells use payloadc057..., while C71binds R3payload3f...
(including genuine emptyE2pool recovery): do NOT silently reuse old wrapper.
Production wrapper must preserve bound R3science and original productionrefs.

## C75 update — memory-bounded engineering retry running

The first USED12/all51acquisition run stopped at15:28:56UTC when hostRAM
fell below4GiB during ICEstage startup. Engine2/PubChem/Engine1 had completed
with0mainerrors. No complete result/parity/quality claim. The original Linux
process group is stopped. ALL original logs/intents and native working data
were preserved under fixture-port/attempt-01-host-memory-stop and the owned
native attempt-01-host-memory-stop-working directory; no prior evidence deleted.

A real inherited systemd user-scope probe PASSED: memory.high8GiB,
memory.max10GiB, memory.swap.max4GiB. Only this task is limited; no system-wide
policy, user apps or original scientific sources changed. ONE reconciled retry
started15:40UTC using scripts/retry_c75_memory_bound.py run, WindowsPID16668,
LinuxdriverPID/PGID401, toolsession38194. Original notebook/driver/models/masks/
all51acquisitions unchanged. Host reserve remains4GiB. Latest15:46UTC reserve
7.19GiB, scopeRAM4.68GiB/swap0.82GiB, oom0/oom_kill0. Innercell32Engine2.
Do not launch a duplicate or interpret captured child stdout as stopped work.

Prepared/froze complete_c75_local_target.py for honest localfull collection,
unchanged C74P16/physical-parent replay and frozen C71pair. It only normalizes
files_sha256 path keys in Windows receipts, never molecular strings. Prepared
whole512 one-time quality evaluator and independent exact-rank/denominator audit;
NEITHER fresh predictions nor quality have been run. C72/C13 remain unconsumed,
0/2scoring approvals used, sharedC42ledger absent. Parity on exact P01/P14lists
and feature rows/tolerance remains REQUIRED before full512/2215inference.

## Immediate C75 state — original complete12query local replay is RUNNING

The previous goal turn made concrete progress (clean C71pair/input audits/CUDA
runtime). This turn proved the single-card source adapter, selected a metadata-
only USED12fixture with ALL51original acquisitions, and launched ONE real full
P01/P14 comparison pipeline. NewC72fresh512quality and C13 remain UNCONSUMED;
scoring approvals remain0/2, champion0.480untouched, old/hourly controllers OFF.

ALL selected assets are now COMPLETE:237originalV4Gfiles/20,082,868,365bytes and
388auxiliaryfiles/13,041,244,810bytes. Both freezes exist. Initial SDK2.2.4 JSON
TypeError was diagnosed: downloaded PROVENANCE.json has list-valued `code`.
The scoped resume helper preserves HTTP/auth errors and original worker/SDK;
legitimate2137bytefile verified. A transient locked status-file replace was
handled with bounded I/O retries. An exact retry of one transient HTTPfailure
succeeded; all388files were retained, none dropped. Immutable attempt1/2evidence
and successful resume evidence remain private. No completed large data was redownloaded during recovery.

Windows RDKit/ArrowParquet extensions began returning explicit Application
Control policy blocks. DO NOT disable/unblock/rename those binaries or alter
security policy. Pure packaging helpers were decoupled from unnecessary chemistry
imports, and the already-created/verified Linuxenvironment runs the port work.
Use WSL for subsequent Parquet/chemistry/model operations; Windows stdlib handles
SDK, git and the supervisor. Windows frozen environments remain unchanged.

C75fixture port/source audit PASSED. Every original inner scientific cell is
EXACT except one Engine2recovery GPU1->GPU0mapping. Original outer neural-public/
ours ensemble and final ranking code are EXACT. Modified outercells only0,6,12,27:
truthful single2070allocation, fixturecount/hash controls, honest repackaged inner
payload hash and GPU0negativecontrol. It does NOT claim actual_two_T4s. Original
cloudC57masks retained byteexact. Original512fresh C73/C74masks are for the later
full run, not this engineering fixture. Required parity: exact final P01/P14lists,
featurekeysets/nonfloats/NaNpatterns exact, floats atol=rtol1e-5, all51acquisitions,
complete actual channels/zero soft failures. See scripts/audit_c75_run.py.

Task-owned /kaggle/input and /kaggle/working links in WSL point ONLY to
/home/imsa/.local/share/enveda-c75/fixture/{input,working}. Ownership guards reject
unknown existing namespaces;651file links, no original data deleted/moved. Links
resolve to the original cached/public assets and query-only fixture; no query
labels mounted. Complete port is original inner PLUS original outer blend/rank.

LIVE supervisor: scripts/supervise_c75_local_port.py fixture, actualWindowsPID10592
(venv launcher20264), tool session69046. LinuxdriverPID/PGID319; runnerPID411 and
Engine2PID454observed. Intents/progress/checkpoints under C75/fixture-port. Do NOT
restart because a tool poll times out. Supervisor runs one inference, monitors
hostRAM>=4GiB and terminates only its verified owned Linuxgroup on failure.
Original code/thresholds/models remain; os.cpu_count4/thread2 is resource control.
Own input/venv cache advice only on reserve pressure; unrelated WSL/apps untouched.

Observed15:22UTC: original pipeline at innercell32Engine2; actualtask Engine2RSS
~3.97GB, driver0.69GB, runner0.55GB. Host freeRAM recovered to7.70GiBwithout any
failure. Main pipeline stdout contains original stage progress, nested engine
stdout is captured until that process finishes; an unchanged outercell15 does
NOT mean it is stopped. Read /kaggle/working/submission_pipeline.log via WSL.
Both original DreaMS/MolFormer graphs passed actualGPU0preflight (peak0.4805GiB).
Added ONLY tokenizers0.21.4(no dependency upgrades), matching prior encoderenv;
local frozen tokenizer.json encoding CCO->[0,4,4,9,1] passed. Version addendum saved.
Torch/NumPy/RDKit versions unchanged. Optional training-only multiprocess/pathos
is stubbed by the original simulator code; do not install unrelated packages.

NEXT: monitor this specific live run, diagnose any actual errors without changing
science/cohorts; after successful exit run WSL audit_c75_run.py fixture. If and
ONLY IF complete flow parity passes, prepare/audit/activate C75full with original
C73source/C72fresh512querypackage and new masks; run all512/2215acquisitions.
Then execute audited C74physicalP16 and frozen C71CPU/GPUheads, freeze both before
one whole-source evaluation, actual production checks, sole joint C13once, and
CPUscore then GPUscore through one shared durable C42ledger. Never bypass gates.
KaggleT4launch remains rejected weekly30hourquota(version0); no cloudjob exists.

## C75 update — isolated RTX2070 CUDA execution verified; assets still downloading

At14:22UTC the actual WSL runtime probe PASSED: Python3.12.3, Torch2.11.0+cu128,
RDKit2026.03.3, NumPy2.2.6, pandas2.2.3, Arrow19.0.1, scipy1.15.3,
sklearn1.6.1, LightGBM4.6.0, numba0.61.2. All runtime dependencies installed.
One actual RTX2070/8GiB; real CUDA matmul executed, not just driver discovery.
Host free RAM after probe15.82GiB. Keep4GiBhost reserve at every heavy stage.
Windows environments are unchanged; Linuxenv /home/imsa/.venvs/enveda-c75.

Cached the exact4file public neural bundle from outputs/local/kit under C75input;
all frozen MODEL_PINS match, including public checkpoint44436b2d. Cached v80ours
checkpoint c08d2101 from outputs/research/v80-output. Both SHA256verified.
Assets are STILL DOWNLOADING, two active workers/checkpoints. Latestobserved:
V4G232/237files ~7.13GB/20.08GB; auxiliary165/388files/13.04GBtotal. Reconcile
actual processes/checkpoints before retries. No local pipeline inference started.
The core large PubChem/reference arrays remain; small-file completion counts
are NOT a completed asset package. Tool sessions39744(V4G),97967(auxiliary).

Next port details: the decoder's first literal MODULES={} is intentionally empty;
recover ALL subsequent AST Subscript Assign/AugAssign literal updates. Already
extracted10exact modules + original3f973ab notebook into C75private workspace.
Full P01 also requires original outer neural-public/ours blend and final ranking,
not merely the inner public0.465core. Port the complete original outer flow.

Prefer a bounded task-owned /kaggle input/working namespace in WSL (initially
absent), because original manifest-checked asset modules can hardcode these
paths. Never overwrite unknown existing mounts. Switch only task-owned links,
sequential fixture/full runs, or use path-only adapters with exact reverse-diff
proof. Map explicit GPU1calls to localGPU0; retain algorithms/thresholds/models.
Cap CPU/threads as an operational resource control, without changing features.
Do NOT report actual_two_T4s for a local run. Persist real hardware/port evidence.

Require metadata-only selected USED real12from completedC57, ALL their original
acquisitions and ORIGINAL C57reference masks (not new C73masks), then compare
actual final P01lists/features against the corresponding completed cloud rows.
Only after that engineering parity run the untouched512with new C73/C74masks.
The original single C13 and new C72quality remain UNCONSUMED; scoring0/2.

Git5837e70pushed the complete C66-C74audit/source package. New .gitattributes
preserves all new hash-bound source/contract/evidence bytes EXACTLY; indexbytes
were independently compared against executed files. Historical C50build-v0
remains untracked and untouched. Current native goal remains ACTIVE.

## Immediate runtime state — Kaggle weekly quota rejected; C75 local port underway

At 13:52:36UTC Nov12, the actual C73 T4 push was REJECTED before a version was
created: "Maximum weekly GPU quota of 30.00 hours reached." Response version0,
emptyURL. The private query dataset is ready. Do NOT blindly requeue or call
this a running cloud job. No new scored submission exists; approvals remain0/2.
This is one demonstrated external limitation, not a complete goal blocker while
local work can progress. No paid compute is authorized or requested.

All C72 acquisition/official-identity/chemistry audits passed:512queries/2215actual
timsTOFacquisitions,870logged unusable acquisitions, full source fidelity. All
3220actual fitting acquisitions vs2215new acquisitions have ZERO full/top96input
aliases; the513raw alias keys have no fitted-query official-identity overlap.
C73 masks independently passed:91846mainSIDs/rawkeys excluded;182742allowed
reference identities independently checked against ALLUSED/fresh/global231chemistry.
C74 P16 external-reference mask independently passed:6186retained of200724rows,
11978precursor-reachable, exact retained peaks/metadata and zero forbidden/input-
alias rows. Source labels/quality outcomes and C13 remain unconsumed.

WSL Ubuntu-24.04 is installed and sees the actual local RTX2070(8GiB); Linux
Python3.12.3. Created ISOLATED /home/imsa/.venvs/enveda-c75 with PyPAvirtualenv
zipapp, without changing frozen Windows GPU/encoder environments or systemPython.
CUDA Torch2.11.0+cu128 install succeeded; CPU/runtime dependencies are installing
from configs/c75-wsl-requirements.txt. Installation file cache briefly reduced
host reserve; targeted posix_fadvise ONLY on this new venv/pip cache restored
14.47GiBfree. Preserve at least4GiBhost reserve; never shut down unrelated WSL
processes or user apps to recover memory.

Active C75asset workers: scripts/prepare_c75_v4g_assets.py (complete original237file,
20,082,868,365byte public dataset; remote census matches original; reuse pinned
C20copies) and prepare_c75_aux_assets.py (other5publicdataset mounts,388files,
13,041,244,810bytes). It explicitly omits ONLY12unused64GiBfingerprint-index binary
shards: the frozen pipeline/PC runner/helpers never import that index. Full remote
census is retained. All remaining original inference arrays/helpers are fetched.
Workers/checkpoints are in outputs/local/c75-local-core-port-20261112. Actual
process IDs and tool sessions may differ from venv launchers; reconcile them.
The old compute-paused.json marker is historical (Nov11 before22:29newgoal) and
MUST stay to disable old controllers. C75uses newer explicit C38authorization.

NEXT: finish isolated deps/assets; verify CUDA; port only paths/GPU1->GPU0/worker
resource controls while preserving every scientific threshold/model/algorithm.
Require parity against a metadata-selected USED real12subset of completedC57
with its ORIGINAL masks before fresh512full inference. Original3f973abpayload
and decoded helper modules are extracted privately under C75; no port or local
inference has yet run. Do not claim a Linuxport is an already-qualified pipeline.
C71models stay frozen; C72fresh quality/C13reserved outcomes remain sealed.

## Active goal — clean C71 pair frozen; C72 fresh source audits underway

The native goal remains ACTIVE until two actual new terminal numeric Kaggle
scores. 0/2 approved scored entries used; no repeat approval is needed. Preserve
the 0.480 champion, all historical evidence and the single shared C42 submission
ledger. Keep old/hourly controllers OFF. C13 has NEVER been consumed.

C64/C65 rejected. Adding omitted parent continuous ranks, documentation and NP
priors produced development-positive C66 CPU and C67 GPU. A further audit found
two global reserved structures used as negative training candidates. Those fits
are permanently HELD from qualification; their positive metrics remain history.
The unused fit_c67_final_pair.py entry point now explicitly refuses to run.

C68 resolved the ENTIRE public 231-identity reserved inventory for exclusion only,
without reading the selected120 query mapping or any reserved outcomes. Its
identity/formula/nonempty-Murcko/Morgan2>=0.70 quarantine excludes connected query
components and candidate rows from BOTH pair loss and fitted feature scaling.
All 869 USED queries/3,452 acquisitions/21,719 candidates remain in evaluation;
780 queries/3,220 acquisitions/18,833 candidate rows are eligible for fitting.

C69 strict CPU OOF passes every predeclared cohort/source/mode marginal guard:
MRR +0.008493724 vs P01, 36 improved truths and 2 harmed. C70 strict GPU OOF also
passes those guards: +0.013019999, 47 improved and 4 harmed. GPU has one disclosed
negative intersection, old Enveda180 mixed46: -0.001811594; do not claim every
intersection is nonnegative. CPU intersections are all nonnegative. These are
adaptive USED results, not fresh confirmation or public score predictions.

Independent audits reconstruct features, train-only normalization, exact ranks
and fractions, convex CPU optima, all five GPU training replays and NumPy serving.
C71 refit both fixed recipes using ONLY eligible rows, independently replayed the
actual final fit and bound the complete frozen P01/P16/physical-adapter/head
recipe. CPU/GPU models are immutable; no new source outcomes have been read.

C72 was selected AFTER that complete binding: 512 untouched Enveda180 identities,
353 positive/124 mixed/35 negative, all 2,215 usable original acquisitions; 870
unusable acquisitions explicitly logged. Selection excludes every USED/global
reserved chemistry and every prior head-input candidate identity, without using
quality outcomes. NP examples are exhausted; plant metadata remains unresolved.
This is controlled target-domain in-corpus validation, not independent labs.

Current work: C72 all-raw acquisition alias census, independent acquisition and
chemistry audit; C73 global reference chemistry/mask preparation. Only then build
and run the same frozen dual-T4 core, execute physical P16 and both frozen heads,
freeze predictions before one whole-panel quality evaluation, and independently
audit it. If both survive, actual production checks and the sole joint unused
C13 confirmation remain mandatory before CPU submit -> numeric score -> GPU
submit -> numeric score. C59/C62 old C56 release paths stay HELD.

Current artifacts: outputs/local/c68-reserved-fit-boundary-20261112;
outputs/local/c69-clean-parent-evidence-oof-20261112;
outputs/local/c70-clean-gpu-parent-evidence-oof-20261112;
outputs/local/c71-final-clean-direct-pair-20261112;
outputs/local/c72-clean-target-holdout-20261112;
outputs/local/c73-clean-target-validation-20261112; docs/evidence/c66-c72-20261112.

## Active CPU development — C64/C65 reject; no GPU or scoring release

The native goal remains active until TWO actual new terminal numeric scores.
0/2 approved entries used; C13 NEVER consumed. Preserve0.480and all prior evidence,
keep old/hourly controllers off. There is no qualified CPU/GPU pair currently.

C56failed full prospective482C58source: zero improved truths,5/6head harms, plus
2wrong P16dehydrated-reference promotions. All482are permanently USED. The full
C57core, source-acquisition/ref-mask audits and P30/P31production checks succeeded,
but engineering checks cannot override that scientific rejection.

C63isolated both P16errors: same raw precursor, reference[M+H-H2O]+ versus query
[M+H]+, inferred neutral masses differ by18Da. A label-blind neutral-mass check
rejects both and restores P01. All869USED queries/3452acquisitions are retained.
456of482C53truths were protected at ranks1-2; old tail restriction had only23truth
opportunities. Semantic target regression and correlated CPU/GPU votes failed.

C64prepared21,719candidate rows/87whitelisted raw core features, query metadata,
741chemistry groups and5whole-component folds over ALL869USED queries. No C56
fitted projection is reused. Same-formula direct pairwise CPU fitting allows
leader changes, preserves membership/formula slots, and evaluates every query,
including absent truths as zero. No fold normalization/fit sees its held-out data.

The fixed C64linear residual improves pooled OOF MRR by+0.000847763versus P01 and
+0.001998511versus P16. Original192/352and whole482improve, BUT oldNP80, both C51
sources and C53Enveda180/positive guards regress. C64is REJECTED for GPUfollow-up.
C65weights pairs by RR swap cost: -0.000673968versus P01, also REJECTED. No outer
fold recipe rescue, no dropped source, no GPUfit or scored entry was launched.

Independent audits reconstruct saved OOF scores/lists, all source/mode exact
fractions, original source labels and chemistry isolation. Training-only scaling
matches exactly; convex objective gradients <8e-7verify saved linear optima.
These remain adaptive USED development, not fresh validation or LBcalibration.

Next safe actions: inspect source failures/physical-channel disagreement in C64,
test a CPU-only conditional/physical-consensus ranking approach without narrowing
the cohort, and inventory new source eligibility against the NEW calibration
fitting boundary (869query identities/chemistry, not the retired153kprojection).
Any survivor needs separately frozen untouched source outcomes and actual full
production checks before the sole reserved C13evaluation and two scoring trials.
C59/C60/C61/C62prepared C56release paths stay HELD; no C13push was made.

Current artifacts: outputs/local/c63-source-loss-diagnosis-20261112;
outputs/local/c64-pairwise-oof-20261112; outputs/local/c65-rrweighted-oof-20261112;
configs/c64-physics-pairwise-oof-20261112.json; configs/c65-rrweighted-oof-20261112.json;
docs/evidence/c64-c65-20261112. All workers have finished; no training/cloud job
is currently active. Historical sections below preserve prior observations.

## Active goal — C56 rejected on full C58; CPU loss/ranking work next

The native goal remains active until TWO actual new terminal numeric scored
entries. 0/2 approvals used; no repeat approval needed. Preserve 0.480, keep all
old/hourly campaigns off. C13 has NEVER been consumed. No pair is qualified now.

C57 private v1 completed successfully at12:06UTC Nov12. Independent collection,
all2,125original acquisitions/482queries, reference quarantine and full-channel
health audits pass. P30CPU/P31GPU private previews also pass engineering checks.
Those checks do not override the failed prospective prediction-quality gate.

C58 replayed exact P16, froze/scored both clean C56 heads, consumed the ENTIRE
482-query source panel once, and independently confirmed both rejections:
CPU full delta -0.002593361, GPU -0.002616413 versus P01. P16 itself harmed two
correct rank1answers to rank2 (-1/482): high raw-precursor cosine matches to
[M+H-H2O]+ references with incompatible neutral masses. No truths were recovered
or displaced from lists; all3absent truths remain zeros.

Beyond P16, CPU harmed5 truths (-0.000518672), GPU6 (-0.000541724); neither improved
one. All queries, both sources and actual307positive/141mixed/34negative modes
were retained. All482C53queries are now USED, never call them fresh again. Combined
USED data now869queries/3452acquisitions (old352+35 plus482).

C63 diagnosis preserves every changed-truth case privately. Parent MRR0.92596on
this in-corpus panel is not leaderboard calibration; public parent/encoder historic
exposure persists.456truths were protected at ranks1-2; only23were in the old
ranks3-10 window. Even perfect ordering there can gain only0.002812on this panel.
CPU/GPU votes are correlated through shared data and the CPU anchor.

Next authorized work: test adduct-implied neutral-mass consistency against ALL
USED complete parents; assemble chemistry-group out-of-fold direct isomer-ranking
CPU tests using frozen core evidence. Test CPU before any further GPU fitting.
A new family needs honest held-out development, a separately frozen untouched
source panel and both production checks before the sole joint C13evaluation.
No cohort/recipe rescue or gate weakening inside rejected C56.

C59 faithful query-only C13 main/fragment code is prepared ONLY, NOT PUSHED.
Its input-only channel preflight derives exact PC119support from the unchanged
original120queries; Engine2/global output must keep all120, including one trailing-
space adduct fallback. Oldv2PCcount98proved the missing main adapter. Four coverage
tests pass. New C59auditor/collector/futureconfirmation bind actual channel IDs and
all original metadata/peaks. C13 FT/HCD is not target timsTOF calibration.

C59/C60/C61/C62prepared C56release guards remain HELD by the actual rejection.
No collector/continuation is still running: C57observer26852and C58worker26556
finished; original idle30916was retired before any quality read. All models,
predictions, failed families and original evidence remain intact. No new scoring
ledger exists; eventual new entries must share .local/c42-two-new-submissions-
20261111.json, cap2accepted,1pending, CPUactualnumeric score before GPUrelease.

Evidence: outputs/local/c58-complete-target-20261112;
outputs/local/c63-source-loss-diagnosis-20261112; docs/evidence/c58-c62-20261112.
Historical sections below preserve previous observations and rejected families.

## Active C56 pair — both production previews pass; fresh C57 core running

The native goal remains active until TWO actual new terminal numeric scores.
0/2 approvals used; no repeat approval needed. Preserve 0.480 and keep old/hourly
campaigns off. C53 source quality and the single joint C13 confirmation are unread.

C56 is a clean CPU ridge / bounded GPU semantic pair fit on 153,401 identities and
527,097 acquisitions, excluding the entire C53 chemistry and raw/official alias
closure. All 387 USED development queries pass every fixed guard. Both old352
deltas are +11/26880 (+0.000409226), old35 unchanged: small uncertain development
results, never fresh confirmation. Preserve rejected C49/C54/C55 evidence.

Actual private P30 CPU and P31 GPU v1 previews BOTH completed and passed independent
production audits. All final lists match local execution; respectively 9 and 8 of
the 12 real preview queries change. Each has 400 valid rows / 388 exact fallbacks,
unchanged parent, rank1/membership/formula slots protected, actual dual T4 execution.
No scored entries were released.

C57 full masked private v1 (482 molecules / 2,125 original usable acquisitions /
447 chemistry groups) remains RUNNING as of 11:36 UTC Nov12. Actual observer PID
26852, deadline16:05UTC: collect/acquisition-audit only, no scoring or evaluation.
Live SDK logs prove both Engine2 rankers finished all482 in3434s on actual2T4s.
PubChem also completed all482 (3593s); main library/pool cache and ranker models
loaded at119.9min. Main candidate lists also completed all482 in903s, errors0,
popularity-reordered482. Both GLACIER/ICEBERG forward models completed4029s,
two-engine fusion150s and metric audit/safe tail149s done. Final ranking/export
continues; complete job and source quality remain pending.
A one-shot current-task C58 continuation (actualPID26556,source-health-once) waits for C57 audit,
then independently requires full Engine2/PubChem coverage, zero main errors and
all simulator/fusion completion markers before P16 replay, both predictions and the fixed source
evaluation/independent audit. PubChem has a soft-failure fallback, so this explicit
pre-outcome check is mandatory. Original idle30916retired before any labels read;
cloud job unchanged. Both actual12previews and old352full run pass the added check.
The old352full core took3h35; current larger job remains live. This continuation
cannot fit, push, read C13 or submit.
Never requeue a healthy handle. Exact frozen P16 reference masking is independently
complete: 9,177 reachable retained rows, zero forbidden identity/chemistry/input
aliases. Replay and fresh-head inference await actual C57 collection/fidelity.

Prepared C58 workflow freezes BOTH complete predictions before one full-panel
source evaluation, with unchanged strict all482 full/incremental positive and
source/mode nonnegative gates. Crash recovery permits ONLY the identical committed
calculation, never deleting a consumption marker or changing a recipe/cohort.

C59 faithful C13 main/fragment adapter notebook is prepared ONLY, NOT PUSHED.
Old C13v2 repaired PV only; same production adapter recovers 119/120 main masses,
retaining the original unsupported trailing-space token. Whole231 identity census
has zero main-reference and prior-used identity overlaps. Oldv2also had only98
PubChem query entries. Exact patched chemistry now predeclares119supported PC
queries, Engine2/global output must retain all120 including one unchanged trailing-
space fallback. Independent channel audit compares actual dictionary IDs to this
input-only declaration;4syntheticcoverage tests pass. No query or model changed.
C60 whole231 P16 mask /
replay and C61 single joint confirmation are prepared, conditional on C58 passing.
All120 original C13 queries and absent-truth zeros must remain; no TIMS calibration
claim follows from this FT/HCD confirmation.

C62 qualification / shared cap2 ledger / CPU-then-actual-score-then-GPU observer
are prepared, NOT RELEASED. Twelve offline C62tests cover numeric-score sequencing and actual platform quota,
network/score failures, other pending entries, missing/tampered qualification,
shared ledger cap, exact durable-intent recovery and retired-controller rejection.
C62 may activate only its exact evidence-bound qualification; retired C42/C43
controllers retain their nonempty hold. No third scoring entry is authorized.

Next: complete C57 acquisition audit, exact P16 replay, both fresh predictions and
C58 quality. Only if passing: faithful C59/C60, one C61 confirmation, independently
verify admission, then qualify and score P30/P31 sequentially. No outcome rescue.

Current paths: outputs/local/c56-bounded-semantic-heads-20261112;
outputs/local/c57-target-validation-20261112; outputs/local/c58-complete-target-20261112;
docs/evidence/c58-c62-20261112. Historical sections below preserve earlier evidence.

## Active C51 decision — C49 pair rejected on fresh source

Both frozen C49 candidates failed the prospectively frozen complete and incremental
nonregression gates on all 35 auxiliary queries and the five-query MassBank source.
One truth moved 12 to 14: all35 delta -1/2940 (-0.000340136); MassBank delta -1/420.
The 30 NIST APCI queries were unchanged. Independent exact-rank audit agrees.
Preserve this failure; no recipe/cohort rescue inside C49. Auxiliary outcomes are
now USED development, C13 remains untouched and 0/2 approvals have been used.
The native goal remains active. Old/hourly campaigns remain off; preserve 0.480.

CPU P25 private v1 completed at07:23UTC; its independent production audit is
running. GPU P27 private v1 is running. Engineering audits cannot override the
scientific rejection. Next work diagnoses structural isomer ranking and inventories
a new separated validation population before another model fit. The remaining
C13 compatibility gap must still be repaired before any eventual once-only use.

Evidence: docs/evidence/c51-20261112 and outputs/local/c51-complete-auxiliary-20261112.

## Active C49 pair — fresh source replay and private production checks

The native goal remains active: complete at most two approved scored entries,
CPU first and then GPU after its actual numeric score. Neither approval is used.
C13 outcomes remain untouched; the 0.480 champion is preserved and hourly/old
campaigns remain off. No repeat approval is needed.

C48 refitted clean CPU ridge and a bounded GPU residual on 174,051 identities /
600,060 acquisitions, excluding the entire new auxiliary chemistry and raw alias
closure. The original C48 GPU serving recipe failed used development. C49 retains
those weights but requires both heads to support every tail inversion. Its full
352-query / 1,292-acquisition used-development audit passes every original source
and mode guard; the improvement is small and uncertain. C39 is USED development.

The fresh auxiliary panel contains all 35 usable supplied exports: 30 public
experimental NIST APCI timsTOF standards and five remaining MassBank ESI queries.
All are positive; this panel cannot establish mixed ESI target calibration.
C50 cloud core completed and passed exact acquisition and exclusion audits at
07:08 UTC. C51 now masks and replays the complete frozen P16 outer stage before
both frozen predictions are evaluated. No auxiliary quality outcomes read yet.

Frozen production candidates: P25 CPU (C48 clean ridge) and P27 GPU (C49 agreement),
private v1 previews pushed 06:59 and 07:16 UTC. Both need actual production audits.
P26 is rejected. Keep scientific_gate_hold until fresh source checks, faithful
C13 complete-parent confirmation (at most once), and both production audits pass.
Inspect legacy formate compatibility in original C13 v2 before reusing its parent;
its PV-only repair is not proof of the later main/fragment adapter.

Current evidence: outputs/local/c49-consensus-pair-20261112 and
outputs/local/c50-auxiliary-validation-20261112; policy configs/c49-consensus-pair-20261112.json.

# Enveda CASMI 2026 — working rules

## ACTIVE goal after C46 no-go — two scoring approvals remain unused

Read docs/CURRENT-GOAL.md first. C45's complete-parent gap is repaired/audited.
The original GPU head failed development; one prospectively frozen top2-protected
GPU alternative then failed the MassBank and mixed-query source gates. CPU passed
with a small uncertain gain. No pair qualifies, C13 is untouched and0/2scored
entries were used. C39 source outcomes are NOW USED; do not tune another recipe
against them and call it fresh qualification. Preserve all seals and gate hold.
Current work is outcome-blind acquisition/source research under C47, not training
or automatic submission. The native goal is ACTIVE, old/hourly campaigns OFF.

## ACTIVE C45 repair — same two approved new submissions, no outcomes consumed

Historical pre-outcome repair record (superseded by the C46 decision above).
Production P20/P21 include P16's measured-reference
overlay; C41's baseline omitted that outer stage. The old automatic evaluation,
confirmation and submission observers are stopped and held. The same cloud base
inference completed and passed collection/acquisition audits at04:28UTC; the
base-only collector exited normally. The C45 masked overlay replay and frozen
head scoring are current; old automatic observers remain stopped.
Repair and audit the complete parent, compare full and incremental candidates,
then qualify before any C13 outcome or scored entry. Models, recipes and cohorts
remain frozen; two approvals remain unused. Do not restart historical C42/C43
paths or clear scientific_gate_hold without the completed C45 evidence.

## ACTIVE C38 goal — new CPU/GPU work and2new submissions approved

Latest user instruction: "Do the next steps as a goal until new gpu and cpu
submissions2approved". The new native goal is ACTIVE. Read docs/CURRENT-GOAL.md
and outputs/local/c38-source-support-20261111/authorization.json. Build audited
source-supported validation with full acquisition sets and proper mixed modes;
integrate the formate fix; develop/qualify new candidates; continue through two
actual new scoring results. Provisionally one CPU-backed and one GPU-backed.
No repeat submission approval is needed. Local2070/free Kaggle CPU/T4 compute
is authorized as scientifically justified; no new paid compute is authorized.
Keep hourly automation and old campaigns off. Preserve0.480 and prior artifacts;
no exhausted past-trial resubmission, truth injection, favorable-cohort tuning,
or premature C13 access. The old pause marker only keeps old workers inactive;
new controlled workers require explicit current authorization and trial caps.

## Completed C37 CPU-first goal — 11 November; GPU remains paused

Read docs/C37-SELECTION-AND-POPULATION-2026-11-11.md and compact evidence in
docs/evidence/c37-20261111. The user cleared the old paused silver goal; the
new CPU-first goal reached its evidence-based go/no-go endpoint. Silver is
unmet; no new public improvement is claimed. Preserve0.480 and fresh C13.

All26 losses are explained by observed budgets/cutoffs:10PubChem five-slot
budget,4main60-to25,1main22 displaced by interleave,4fusion beyond40,
3physical demotions across25 and4already outside25. Exact executed Engine2
rank2 replay matches all240 outputs, loses0true answers; later parent membership
is unchanged. No correctness defect justifies a replacement retention rule.

A separate legacy formate alias bug prevents main/PubChem eligibility on25used
validation queries. The tested local source adapter preserves all59supported
adducts, vocabulary/indices,215existing masses and unknown behavior. It is NOT
installed in production. Public test uses canonical formate already, so this
is no demonstrated hidden-test gain. No inference/training was run.

Full public test metadata:400queries/1213spectra,97mixed-polarity queries,
all timsTOF. Primary240has one FT spectrum/query and0mixed queries. Current
panels lack target instrument support. A query-only prospective two-source
validation protocol is frozen; do not tune populations to old leaderboard
outcomes or call resampling a new representative source.

Correction: C36 historical negative31 is first-record-negative (14pure negative
+17mixed), not31pure-negative queries. Proper source groups are30positive,
14negative,36mixed. Frozen ranker all80 and clean60 scores are unchanged; C36
still rejected. See corrected-source-polarity receipt.18targeted tests pass;
independent arithmetic,64identity spot checks and original C36 seals pass.

Keep .local/compute-paused.json. No GPU/encoder training, GPU inference, cloud
run, submission, paid compute or hourly worker was resumed or newly authorized.
No candidate qualified for fresh C13 access. Next justified work is acquiring
and auditing validation acquisitions with documented target/source support.
Older silver-campaign/training authorizations below are historical, not a
request to restart them.

## Completed C36 cycle — historical evidence; GPU stopped

Latest user instruction: "repair validation, locate candidate losses, then test
ranking changes that distinguish isomers before spending more GPU time."
The requested C36 CPU-only cycle is complete. Keep .local/compute-paused.json
(phase cpu_validation_repair_finished). No GPU/encoder training, GPU inference,
cloud job, scoring submission or hourly campaign was resumed. All CPU workers
exited. The hourly automation and silver goal remain PAUSED; silver is unmet.

Read docs/C36-VALIDATION-REPAIR-2026-11-11.md and docs/evidence/c36-20261111.
Strict all-query/official-identity evaluation, nested chemistry-group selection,
held-out-source admission gates and mass diagnostics are repaired. Empirical
leaderboard calibration remains UNPROVEN: P14 is still a wrong-sign control.
Do not confuse repaired safeguards with a newly validated population proxy.

Candidate route audit:150/240 truths in final lists,176 in retained route union;
26retrieved-but-dropped(19positive/7negative),64absent from saved routes. This is
not proof those64 are absent from full original databases.25missing cached nm
fields are formate queries; offline diagnostic recomputation supports all240,
max7.95047ppm. Production/ranking inputs remain unchanged.

CPU isomer ranker primary grouped OOF MRR.283748->.312032(delta+.028284,
CI[.005327,.051536]), but clean-source60 delta-.023909 and source-negative31
delta-.037154. REJECTED for source/calibration failure; do not promote, fit an
encoder, upload or submit it. Physical ablation and fixed physical rule are also
unqualified. Source80 is historical/used, not fresh. Fresh C13 reserved metrics
remain UNREAD. No final deployment model was qualified.10tests pass;12CPU fits
and640learned prediction lists independently reproduced; original qualified
production sources/models remain byte-exact. See compact evidence.

The previous two past-work entries are exhausted: P18/ref58212322 and
P19/ref58219034 COMPLETE0.478/no reported error. Best remains0.480; never resubmit
these entries. Old C09 coverage remains last observed CANCEL_REQUESTED; no
acknowledged termination or requeue. Older campaign authorizations below are
historical and cannot resume automatic work. New GPU/submission work requires
explicit user instruction after a defensible validation result.

## USER STOPPED CAMPAIGN — 10 November,14:59 UTC

Latest user instruction: "after this next full run finishes stop". Both current
full private Kaggle runs have now completed: C29 development240 and C31 reserved120
predictions. Their outputs are retained. C29 collection passed source/coverage/
baseline checks; C31 retained25 artifacts with its optional recovery-receipt
attention documented. No development or reserved quality metrics were opened.
Both collectors and the exact-job stop controller have exited. The general
tracker, hourly metadata updater and waiting C29 postprocessor are stopped.
Native hourly automation is PAUSED; project compute-paused phase is stopped.
Pause the active Codex goal before replying. Silver remains unachieved; do not
mark the objective complete. Do not launch training, cloud runs, confirmation,
smoke tests or submissions without explicit new user instruction. The additional
public submission approval remains unused, and the0.480 champion is preserved.
P17/C33 remains an UNSEALED LOCAL DRAFT, never pushed or qualified on Kaggle.
Read docs/C33-2026-11-10.md and docs/evidence/c33-20261110/shutdown-status.json.
The older measured-state sections below are historical and cannot authorize
resuming this campaign. Review the stop markers before any explicit resumption.

## Current measured state — 10 November,12:49 UTC

Silver goal UNMET. P16/ref58181280 COMPLETE0.480/noerror. Latest board12:22UTC:
rank972/5066; nominal silver253 at0.485. Last tracker account check12:19UTC:zero pending.
One extra user-approved trial remains unconsumed. Never resubmit P16.

C20v2 COMPLETE on actual two Tesla T4s, collected11:23UTC. All80 fullP01replay;
original79 Engine2 lists exact; the missing high-mass query has25 real generated
candidates,0placeholders/dropped queries. Both neuralmodels supportall80. P14
is exactP01fallback for all80. Cloud80 remains authoritative: local one-query
recovery had the same25identity keys but different order/representatives.
Do not repeat the completed cloud/local recovery or restart exited collectors.

C21/C24 six-group cycle CLOSED11:25UTC, ALL REJECTED. Full240P01=P16 MRR
0.28374784097,150truths. Every fixed negative-only0.20 recipe had a95%CI
crossing0: C08+.000555895,C09+.000366418,C14+.000267968,C15+.000265715,
C18+.000364026,C24+.000313789. No subgroup/model re-selection in this cycle.
P14 control aggregate+.001658614 is WRONG SIGN vs public0.458,negative noop;
P15 aggregate-.000128152 has correct direction but CI crosses0. Keep limitations.
Workers7712/6784 and C25worker7592 EXITED terminal; C25selectednull and preserved
fresh120. C13 evaluation-consumption marker remains ABSENT, labelscores UNREAD.
C25/C26 sealed negative-only deployment registry is now historical; no selected
model, upload or submission resulted. Never revive these closed workers.

C28MV is a separate prospective one-model/all-mode0.20tail cycle. Original C08
40000-update parents receive5000updates/fold, samepolarity distinct acquisition
added with probability0.5; average z/rawFPlogits before unchanged cosineCE/BCE.
Separate view/jitter RNGs; no embedding renormalization. Extra dropout forwards
mean full Torch training RNG parity is NOT claimed. No validation training data.
Pilot100 completed11:48:23,finite/0AMPskips; full continuation resumed11:49:29
with exact optimizer/scaler/RNG/warmup and original14:47:59UTCdeadline. Actual
trainer25924 completed12:23:55 and exited; never refit completed folds.
Only final45000 cumulative twofold checkpoints qualify for fixed used240 gate:
positive95%lower vsboth parents, nonnegative bothpolarities and matchedC09mean,
rank1/sets exact/allabsenttruthzero. See docs/C28-2026-11-10.md. Frozen training
and evaluation configs MUST remain unchanged. If it passes, freeze new all-mode
source/model/recipe before once-only C13full120/alias/polarity gates, realKaggle
changed-path smoke and independent audits before the approved trial.

C27 public-source refresh read text only:240catalogue entries/33new or updated,
5initial notebooks plus2followups. Helios0.488 has6unavailable private inputs;
Ozertuu0.500 has2. New Alex0.487 has all public dependencies and image37c64,
claims ref58185447/0.487; this is not yet account-reproduced. No downloaded code
executed, no public research candidate added to C28. Source pins are evidence.

C28 final twofold training COMPLETE12:23:55:10000successfuladditionalupdates,
0AMPskips/restarts,final45000cumulative,1.8145GiBpeak. Fixedfull240 evaluation
COMPLETE12:24:13 REJECTS it:delta-.000207697CI[-.002140737,.001767892],22up28down;
positive mean-.000822472/negative+.001021853. MatchedC09 delta+.000002024 with
CIcrossing0; no supportedmodelgain. Rank1/candidate sets exact,150truths retained.
Package auditedfinite/private-local only. C28trainer/evaluationworkers exited.
Do not refit or rescue/reselect this recipe; C30now awaits existing C29 only.

C29 private full240 public-recipe replay v1 is RUNNING at the EXACT RETURNED
handle clarkkitchen/casmi26-c29-fixed-public-replay-development. Do not duplicate
its shorter requested slug. Remote private/offline/image37c64 and scientificSHA
0ca0c305... verified. Query-only private datasetv1 READY; no labels/reservedC13.
One fixed MASS/0.12priors/dual-reference0.80 recipe plus mandatory alias-excluded
view; officialidentitydedup/compatibility changes declared. Collector8600 and
developmentworker25024 bounded to22:08:41UTC. See docs/C29-C30-2026-11-10.md.
C30 OUTER POLICY FROZEN BEFORE BOTH C28/C29metrics: WAIT BOTH separate cycles.
Only their fixed passing recipes are eligible; greatest full240MRR/tiesC28.
Never send C28 directly to reserved confirmation while C29 remains unresolved.
C30 selector does not confirm or submit. New all-mode source/model/recipe freeze,
once-onlyC13full120/alias/polarity gate, realKaggle smoke and audits still required.

C31 same frozen C29recipe on reserved120 accepted at exacthandle
clarkkitchen/casmi26-c31-reserved-public-replay v1; remote RUNNING/private/offline/
image37c64/SCI1b089ce0... verified. Only contract/count/run metadata differ; no
new recipe/model/threshold. Existing query-only datasetv1 privacy/READY verified.
Predictions QUARANTINED: no labels or quality metrics. Collector24632 until
22:08:41UTC; requires full120/source/twoT4/originalP01bytes before collection.
C32 once-only confirmation script implemented, actual pending-selection guards
and5 existing gate tests PASS. No C32protocol/metric consumption yet. It requires
C30selected C29 and C33production-source seal BEFORE any reserved metric access.
C33production builder/seal is the next available work; not yet implemented.
No worker automatically calls C32metrics. See docs/C31-C32-2026-11-10.md.

Native24-hour hourly heartbeat ACTIVE(last inspected12:22), bounded read-only
tracker296 continues to22:08UTC. C09oldcoverage remains CANCEL_REQUESTED and
superseded. No paid compute. Historical checkpoints below are superseded.

## Authorization and earlier checkpoints — C09, 2026-11-09

Latest explicit user approval at10November01:37UTC: **another submission**.
One additional trial is reserved/unconsumed in the C09 ledger. P16/ref58181280
is still pending; existing ONE-pending and qualification rules remain. Do not
resubmit P16. Prepare a fresh qualified candidate in parallel. C10 now has a
new160-query CASMI2022 panel,40 development/120 reserved, withzero detected
bounded raw-spectrum overlap. Query-only private inference v1 isCOMPLETE at
`clarkkitchen/casmi26-c10-unlabelled-casmi2022-validation`; actual remote source,
privacy and image pin are verified. Read `docs/C10-CASMI2022-2026-11-10.md`.
No C10 labels were mounted inKaggle. Local40dev evaluation isCOMPLETE:
all four reference recipes tieP01 atMRR0.2722727273,20/40truths. Fixedtie rule
retainsP16;no newreference candidate/no reference confirmation/no additionaltrial.
The shared120 were subsequently evaluated once for frozenC12 at05:23UTC and
are now CONSUMED; never call them fresh/unread or use them for reselection.
C10 is not a leaderboard submission and has
not consumed the extra approval. Never launch a duplicate to the shorter slug.

C10 reference preparation and label-blind search areCOMPLETE:31,030 attributed
CC0/CC BY supplemental spectra/3,966 added connectivity keys; four fixed
parent/expanded and any-positive/same-normalized-adduct recipes are predeclared.
All160 searched,zero bounded input aliases,exact originalP16 parent parity.
Coverage is not a validation gain. Collector5044/13360 anddevelopmentworker
22268/16844 completedandexitednormally. The2070 P15control scoredall160with
0inputerrors. P14controlsupports154/160with6explicitP01fallbacks. P14devdelta
+0.006126CIcrosses0/failsknownpublicregression;P15delta-0.000187,0improved/
2harmed. Do not restartcompletedC10jobs. Read currentC10notes andsealedreports.

C12C development-only coverage audit found25/40truths in earlierinputpools
versus20/40final:5missingtruths areretrievable,15remainoutsideallprovidedpools.
Only2areinexisting20-outsiderpool. Ten engine/PubChem suppliedkeys differfrom
officialidentity;useofficialdedup. C12 now freezes one deeper-pool recipe using
the unchanged P01 public/v80 pair. On40dev, MRR0.2722727273→0.2735802720,
delta+0.0013075,95%CI[-0.0004891,0.0044118],1improved/1harmed,truths20→21.
Selected prospectively by positive-point/new-truth rule;NOT submission-qualified.
Confirmation protocol froze05:08UTC before120metrics. All160 preparation and
2070scoring completed05:23UTC,14184candidates,0errors,rawparitymax2.87e-6.
Exact40dev output parity passed before the once-only120 evaluation05:23UTC.
C12 is REJECTED:delta-0.0000232001,CI[-0.00160504,0.00171527],2improved/
8harmed,truths80→80 (2recovered/2lost). No newtrial or changed-pathsmoke
is justified. All C12 workers exited. Do not rerun confirmation/reselect on120.
See docs/C12-DEEP-CANDIDATES-2026-11-10.md and sealed local receipts.

Current05:44UTC: C13LOTUS source audit traces149983/149983 structures to
checksum-verified originalZenodo19360665; originalCC-BY4.0 conflicts mirrorCC0
claim,so preserve attribution.950retrieved rawkeys differfromofficialidentity.
LOTUS adds0oracletruths beyond earlierdeep pools on160used queries. One fixed
joint-top public/v80 agreement recipe also FAILSdevelopment:delta-0.0001199614,
0improved/3harmed,8insertions,0newtruths. No C13candidate/newtrial selected.

Fresh C13confirmation120 (60positive/60negative) is prepared with strict scan,
priorpanel/training identity and bounded raw-spectrum exclusions;0detected
aliases. Query-onlydataset isREADY/private. Actual kernel
clarkkitchen/casmi26-c13-fresh-mixed-casmi2022-validation v1 isRUNNING;remote
source/privacy/internet/imagepins verified. Collector2748/4928 islive to15:35:59UTC,
downloadsonly/neverlabels/metrics/submission. Fresh120predictionmetrics remain
UNREAD;keys were usedonly for training-exclusion/sourcepanel construction.
Do not duplicate the private kernel or run a failedC13 confirmation. Read
docs/C13-C14-2026-11-10.md and actualprocess/status receipts before recovery.

C14CE completed06:08UTC:10000successfulupdates across2folds,0AMPskips,
no restarts,peak1.4121GiB. KnownCE masked50%,same C08parents/data/seed/cosineBCE/
5000successful-updatesperfold schedule as completedC09cosinecontrol. Audit
shows90.308%trainingCEknown versusallvalidationCEunknown. Separate checkpointed
CE RNG preserves other drawstreams;3tests pass. ZeroC10/freshC13keyoverlap with
theentirebasecatalogue. ActualCUDAcheckpoint RNG restoration isbit-exact.
Both trainers andC14evaluator exited;do not restartcompletedfits. Final artifact
audit passes/localpackage only,notuploaded. Development06:09 rejectsC14:
delta-0.00138476 vscompleteP01/P16 CI[-.00466296,.00150427],14improved/16harmed;
vsmatchedC09control-0.000520553 CIcrosses0. Same100/160truths,rank1/candidate
sets exact. Fresh120metrics remainunread;no newtrial. Evaluatorv2 onlyreuses
pinnedC12features (1319commonoldcache rows bit-exact),beforemetrics;v1 archived.

CurrentC15NP source-balanced2070 training starts06:21UTC,launcher25528/
worker26336,deadline09:21:11UTC. ExactLOTUS raw-key matches mark10578/270127
eligibleidentities (3.916%;conservative sourceflag,not exhaustive NPclass).
Each fold has8462/8461matched trainingidentities. C15samples192members+
192others perbatch;allsameC08parents/data/negativepools/seed/objective/schedule
asC09control,withseparatecheckpointedsource RNG. No CE masking/newidentities.
Three sampler tests pass. At06:22fold0step400,0skips,6.81updates/s,peak1.4121GiB/
freeRAM15.54GiB. Verifylivecommand/handle;no duplicatelaunch. Final-only .20RRF
development protocol froze06:22,onlyused160;positiveCI vsP01/P16 andnonnegative
pointvsmatchedcontrol requiredbeforefresh120. No modelgain/publictrialclaim.
Read docs/C15-NP-2026-11-10.md. C13privatecollector stayslive;freshlabels staylocal.

Historical C10 reference evaluator/protocol was prepared and5gate tests pass;
do not execute it because P16 won/tied reference development. The shared120
were consumed byC12; a futureC13 needs fresh confirmation identities. Full
original-source audit also completed:
all31030 supplemental records match officialMassBank2026.11 commit72980b3,
including peaks/structures/license/authors/precursors/adducts.0unresolvedissues;
91missingoptionaldeclaredkeys resolve through originalSMILES/InChI. Parent
Takumu provenance and mirrorcompilation terms remain explicit; no blanket
rights/gain claim or newdatasetdeployment. See currentC10 notes/evidence.

C11F feasibility isCOMPLETE:the31,030 added references have only58 rawkeys
absentfromrawtrain (57afterprior-panel exclusions;upperbounds),and4,169of4,572
rawkeys directlymatchtraining-eligibleC08/C09canonicalkeys. Lookupnovelty is
not trainingnovelty;no newfit launchedfromthisinventory. Prior-panel identity
keys,includingC10reservedkeys,were readONLYfortrainingexclusion;no predictions/
reservedmetricsorC10-specifictruthcoveragecomputed. C10recipes/protocols stay
frozen. The2070controlworker hascompleted;choose furthertrainingchanges
from completedpipelineerror evidence. No novel-acquisition claim.

C11PUB refreshed247publiccatalogue entries/five sources. Fourunchanged;
peaks-v2 adds a dual-polaritymassguard and changesoverlaythresholds. No exact
version score verified,no adoption. C10 has0pairedqueries (preview5,C09used36,
C08used34);it cannot validate that newpath. Treat it as a separate untested
hypothesis requiringfreshpaired fullpipeline evidence;do not change frozen
P16/C10. Source/risk review is in currentC10 notes/evidence.

Latest user override at22:07UTC: continue UI/work until verified silver; run an
hourly worker for24 hours. Active Codex goal and heartbeat
`casmi-silver-campaign-hourly-for-24-hours` target this chat. Read
`configs/c09-silver-worker-20261109.json` and current C09 notes before acting.
Latest05:23UTC10November rank934/5048, score0.480; nominal silver cutoff252
at0.483. Matching a tied cutoff score is insufficient; verify actual rank.
No paid compute or
unqualified submission is authorized by the rank target. Continue useful work
between hourly wake-ups; schedule expiration is not goal achievement.

Current candidate C09E/P16: fixed measured-reference overlay on unchanged P01,
official identity dedup, source/data hashes, fail-closed scoring filename.
Old30 development MRR.737222→.911111 (7 improved,0 harmed,2 missing truths
recovered);0 detected input aliases and0 cosine>=.999 matches. No threshold
search. These are used external-source development records, not hidden proof.
Implementation froze22:28UTC while new80 was still RUNNING/unread. P16 v1
failed after P01 because the public checkpoint audit used the mutable second-model
bundle alias. V2 corrected the explicit bundle audit without changing the rule,
models or reference files, and is now COMPLETE. Independent execution audit
passes:400 valid rows,12 real queries supported by both models,zero metric
duplicates;30 isolated real development fixtures exercise7 promotions/2 insertions.
The preview12 are protected library matches, so the fixtures are necessary
changed-path coverage and do not modify the scoring CSV. Frozen source/model/
reference hashes and local/cloud scores pass. Both confirmation finishers are
now COMPLETE. Fixed P16 improves full80 MRR.713859→.837445,delta.123586,
CI[.05625,.196503],13 improved/3 harmed,truths71→77. Raw-clean60 is entirely
MassBank:MRR.826687→.890575,delta.063889,CI[.005556,.133333],6 improved/2 harmed.
All20 GNPS queries are raw-signature flagged; there is no clean GNPS evidence.
Reference-input alias exclusion gives identical results with0 detected matches.
Three harms include one NH4/H adduct mismatch; the fixed rule remains unchanged.
The preplanned source reporter ran once and its report is sealed; do not rerun it.

P16v2 was submitted at10November01:22UTC,ref58181280,marker
`C09-P16-EXTREF-20261109`. Authenticated API confirms PENDING and native UI
shows Notebook Running. ExactlyONE pending submission; DO NOT resubmit or send
another trial until its actual score resolves. Keep champion0.480. Hash-bound
review: `outputs/local/c09-external-reference-20261109/submission-qualification.json`.
All scientific/source/execution checks passed before submission, including remote
version/code/privacy/internet checks and a pending0 dry-run. Actual hidden gain
is unproven. The full80 panel fails to reproduce P14/P15 public regressions;
source/provenance limitations and complete review are in current campaign notes.

Read-only tracker PID is recorded in `campaign-tracker.json`; its new deadline
is10November22:08:41UTC and per-thread system-sleep inhibition is confirmed true.
It keeps hourly rank/health evidence without changing display timeout/locking.

Latest user request: **update and retry Kaggle cloud T4s**. The
C09C2 private dual-T4 v2 retry is COMPLETE, verified10November01:03UTC. Full80
P01 replay is exact:MRR.713859,71 truths. P14 is.717609,delta+.00375,
CI[-.0071875,.0146875],7 improved/4 harmed, and does NOT reproduce its known
public regression. This panel cannot be treated as a calibrated hidden-score
predictor. GPU1 DreaMS/MolFormer preflight is finite on a realT4, Torch2.10.0+
cu128/CUDA12.8,peak.4805GiB. Both local frozen-parent/control and C09E confirmation
finished. Never rerun completed C09C2 or restart its completed finishers.

Local C09T, C09CT and C09H all finished normally:10000+10000+40000 successful
new updates, zero restarts. C09H had10 AMP-skipped attempts, not counted as
updates; peak reserved1.5625GiB. Queue and trainers exited; never relaunch them.
Development-only comparison of14 model files/seven groups ties all new variants
with the P15 parent atMRR.740000. Predeclared tie rule freezes cosine-control
at0.20; there is no demonstrated additional training gain or submission qualification.
New80 confirmation results were not used in selection. Freeze receipt21:32UTC,
while the T4 run was RUNNING: `outputs/local/c09-followup-evaluation-20261109/`.

Final80 confirms no additional training benefit: frozen control delta+.001891
versusP01 has an interval crossing0; versusP15parent it is-.000223 with0 improved/
1 harmed. On raw-clean60, all trained recipes equal the parent and harm1 query
versusP01. No trained C09 model is submission-qualified. Do not repeat these fits.

Read-only campaign/hourly tracking continues. Scoring runs use platform
`kaggle-submission` and actual account status, never the COMPLETE commit kernel.
The readers were recovered on current code with their original24h deadline;
current PIDs are recorded in the tracker/process receipts. Dashboard
http://127.0.0.1:8766 shows the completed training queue.
C09S3 CPU audit completed:20/80 queries match raw spectrum signatures despite
cached-group exclusions. Full80 remains primary; predeclared label-blind
no-raw-overlap cohort is secondary. Do not call this universally leakage-free.

The user's attached handoff explicitly resumes a **fresh improvement campaign**
using the local2070 and free Kaggle concurrently. Training/recovery/private model
uploads/qualified sequential submissions/GitHub pushes are authorized. Historical
C08/F07 exhausted-authorization and pause instructions below do not limit C09.
Exactly **ONE pending leaderboard submission globally** is allowed. At21:06UTC
P15/ref58163163 is COMPLETE0.479/no error, below champion0.480. P16/ref58181280
is now the only pending submission. Never automatically resubmit either trial.
Use `configs/c09-campaign-20261109.json`, `.local/c09-submission-ledger.json`,
`docs/C09-CAMPAIGN-2026-11-09.md`; inspect processes before any restart.
C09T completed its continuation of existing C08 fits; C09V is the obsolete
cancel-requested diagnostic, replaced by completed C09C2.
The existing128-query panel is already used, not an untouched new holdout.
Keep P01/P13 at0.480 champion until an actual better nonempty score returns.
Use `scripts/c09_submission_guard.py` for fail-closed account status, not the old
two-pending submitters. No paid/rented compute. All original resource floors,
real smoke, source/artifact/numeric/identity/fallback/parity requirements remain.

## Current handoff — C08 complete/P15 accepted, 2026-11-09

C08 completed all8 x40000 fits at09:45UTC with320000 successful updates and
zero restarts. Development-only selection froze legacy16-hard /0.20 tail RRF;
reserved external delta+0.0002537 has95% interval[-0.0022017,0.0028206], so no
reliable improvement/statistical promotion is established. P15 v1 passed the
real12 Kaggle smoke and independent execution/artifact/arithmetic/identity/
local-Kaggle audit, then was accepted at12:01UTC, ref58163163, unique marker
`C08-P15-20261108`. The API confirms exactly one match, PENDING/no score.
The one new authorization is exhausted; do not re-submit or start another fit.
Training supervisor, hourly tracker and completion runner exited normally.
Keep P01/P13 at0.480 as champion while scoring runs. Read-only score updates are
allowed; source/results/GitHub sync remains authorized. See
`docs/C08-CAMPAIGN-2026-11-08.md` and its receipts. The startup instructions below
are historical and must not trigger a duplicate campaign.

## Latest resumption — 2026-11-08

Latest user override: **"Start it now and track it with a runner every hour then
submit when finished"**. Launch C08 now, keep its bounded30-minute recovery
supervisor plus hourly read-only runner, and carry through validation/packaging/
real Kaggle smoke to **one** trained-model scoring submission. No further approval
is needed for that single trial, including if scientific improvement remains
unproven (as with the prior explicit trial override); preserve the0.480 baseline
and report the evidence honestly. Execution/artifact/numeric/smoke checks remain
mandatory. Campaign authorization: `configs/compatible-campaign-20261108.json`.
Do not edit the pinned scientific trainer/config while it runs. Continue in this
task through submission/score rather than creating a job-tracking app automation.

The following preparation-only status predates that latest authorization.

User requested a fresh public-notebook strategy scan and preparation of a long
2070 run. This supersedes the older no-training-queued scope for **preparation
and short pilots**. C08 is prepared at `outputs/local/long-compatible-20261108`;
the full run/supervisor/24h clock have not started. Its isolated C08S pilot tests
all8 recipes/folds and actual checkpoint recovery. Follow
`docs/COMPATIBLE-LONG-RUN-2026-11-08.md` and its final readiness receipt. Do not
retrain F07 or rewrite its scientific sources/checkpoints.

Latest actual scores: P13 ref58105681 **0.480**, P14 v2 ref58111204 **0.458**,
both COMPLETE/no reported error. P01/P13 remain best0.480. Conditional F07 gains
did not transfer. Both previously approved scoring slots are spent; no new
submission is authorized or queued. Training completion is not qualification.
Public catalogue247 entries/20 selected sources, no downloaded code executed.
The prior request to save code/results to GitHub remains authorization to sync
this preparation; model/data/runtime/credential files remain excluded.

The dated sections below are historical and do not override this scope.

## Active resumption — 2026-11-07

Current checkpoint —7Nov20:52 Eastern: training is COMPLETE (both50k-step heads,
zero restarts). **Both approved scoring submissions are accepted:** P13 v1
ref58105681 and P14 **v2 ref58111204**, scores pending. The two-slot budget is
exhausted; no further candidate or training is queued. P14 v1 was not submitted.
V2's full real12 Kaggle smoke and independent model/fusion/fallback/dedup checks
passed; deterministic NumPy sorting fixed cross-platform peak selection and
local/Kaggle raw-score parity is0.000489. The saved budget ledger and receipts
make retries idempotent. Preserve P01's verified0.480 baseline pending scores.
The earlier active/reserved instructions below describe history; do not re-submit
or start another model based on them.

The user explicitly requested starting the proposed frozen-feature training,
public mining while it starts, and approved **two new submissions**. This
supersedes the wait below. Allocate one submission to a reproducible public-code
trial and reserve the other for the new trained candidate after validation.
Real Kaggle smoke and exact artifact checks remain required; no extra approval
is needed for those two submissions. Avoid additional candidates beyond that cap.

P13 v1 is now submitted, ref58105681 at2026-11-07T22:04UTC; score pending.
Independent real12 smoke/400valid and exact P01 primary/outsider-blend/dedup
checks passed. One trained-model slot remains approved; no second submission is
queued. The current training is **F07R1**, output
`outputs/local/long-frozen-20261107-r1`. It retains the original24h deadline,
reuses the byte-verified molecule cache, fixes fragment m/z<=1000 before DreaMS,
and passed the1000-step learning pilot. The superseded F07 was stopped before
any head fit and is retained for provenance. Follow
`docs/FROZEN-TRAINING-2026-11-07.md`; do not restart the superseded run.


P12 ref58071964 has finished with an unhandled hidden-rerun error and **no score**;
do not treat COMPLETE without a nonempty score as a successful leaderboard run.
The verified best remains P01 **0.480**. Keep that working production baseline.

Update7Nov20:29 Eastern: F07R1 finished both50k-step heads at18:22, zero
restarts. P14 v2 (`clarkkitchen/casmi26-p14-frozen-channel`) is running its
mandatory real Kaggle smoke for the second/last approved slot. It uses P01 plus
seed202611071 at50% on final lists; P01 outsiders are unchanged. Development-only
selection and fresh reserved conditional fusion check passed (+.104787 MRR,
CI[.070643,.140729]); this is not candidate recall or end-to-end validation.
Before submitting, use `audit_frozen_smoke.py` on downloaded real outputs and
verify actual COMPLETE,400 valid rows,12 real inputs, byte pins, exact fusion
and unsupported fallback. Budget ledger `.local/f07-submission-budget.json` has
P13 used and P14 reserved. Marker for P14: `F07-P14-20261107`; version2 only if
it passes, no extra approval needed, maximum two scoring submissions total.


## Current handoff — GitHub sync and wait, 2026-11-06

The latest user instruction is **"Update and push everything it GitHub with
results then wait"**. Save the current code/configs/notebook/research notes and
compact numeric evidence to GitHub, then wait. Do not launch more training,
experiments, submission controllers or extra scoring submissions until an
explicit request to resume. P12 v1/ref58071964 is already submitted and still
pending at the latest API check; its existing Kaggle scoring rerun may finish.
The verified public best remains P01 **0.480**. The local pause marker records
this handoff after the push succeeds. Read-only status updates remain allowed.

Completed evidence: L06 ten models/1,280 epochs/640,000 batches, zero restarts;
control/external reserved retrieval MRR0.680621/0.646321. P12 real Kaggle GPU
smoke and independent fivefold/50:50-blend/metric-dedup checks passed. See
`docs/LONG-SUBMISSION-2026-11-06.md` and `docs/evidence/long-2026-11-06/`.

## Current resumption — 2026-11-06

The user explicitly resumed work: research Deotte's systems and all CASMI
discussions, launch a longer local2070 training and purge obsolete artifacts.
Yesterday's end-of-day pause is superseded for that scope. Current verified best
submitted public MRR is **0.480** (P01); P02 .479 and P11 .468. Follow
`docs/LONG-TRAINING-2026-11-06.md` and its resource/recovery/validation contract.
No automatic submission is queued. Obsolete-file deletion was blocked by execution
policy; see the exact inventory in outputs/research/2026-11-06/purge-manifest.json.

The user's latest explicit instruction after completion is **"Submit it use a
submission"**. This authorizes one L06 leaderboard trial despite the earlier
statistical promotion gate. Package the control fivefold family, the strongest
reserved-confirmation family, as P12 with the fixed original50/control50 neural
recipe. Required real Kaggle smoke, exact checkpoint and numeric ensemble checks
still apply. Submit once after those execution checks; do not ask again or infer
an authorization for additional candidates.

That one trial is now submitted: **P12 v1, ref58071964, 2026-11-06 23:50UTC**.
The real Kaggle GPU smoke and independent numeric/metric-dedup audit passed.
Score pending; no additional candidate is queued or authorized by this request.

Kaggle code competition [`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).
Account `clarkkitchen`. Deadline 2026-12-14 23:59 UTC. Metric MRR@25 on ~400 hidden molecules; the public
board uses ~31% (~125 molecules), so one rank-2→rank-1 flip ≈ +0.004 public.

These rules come from the user's instructions of 2026-10-28 and replace everything in `docs/archive-2026-10-08/`
(old README, AGENTS.md, PHASE_1.md, EXPERIMENTS.md — historical only, do not follow them).

## Compute

- Training, inference, preprocessing and evaluation run on Kaggle cloud (CPU or GPU notebooks) **or on this
  workstation's GPU (RTX 2070, 8 GB)** — allowed by the user on 2026-10-29. Local GPU jobs use `.venv-gpu`
  (torch + CUDA); the plain `.venv` stays CLI-only.
- **Keep the workstation CPU light**: no CPU-heavy pipelines locally (the frozen 0.465 pipeline, simulators,
  library search stay on Kaggle); local jobs use few threads. Submissions always come from Kaggle notebooks, so
  locally trained weights go up as private Kaggle datasets.
- The rented Vast server from Phase 1 is **not** used any more.
- GPU quota: **30 h/week** (T4 ×2 or P100, one bucket) + TPU v3-8 20 h/week, both reset Saturday 00:00 UTC. The
  account page (kaggle.com, "Your Kaggle Quota") is authoritative; the API's `totalTimeAllowed` reports 21600 s (6 h),
  which is wrong, but its `timeUsed` is right:
  `.venv\Scripts\python.exe -c "from kaggle.api.kaggle_api_extended import KaggleApi as K; a=K(); a.authenticate(); print(a.quota_view())"`
- A T4 ×2 session has **two** GPUs and is billed per session: run two trainings in parallel (`CUDA_VISIBLE_DEVICES`
  0 and 1), as `scripts/build_gpu_job.py` does.
- Two GPU commit runs can execute at once. Commit runs of the public-lineage notebooks cost ~20–30 GPU min
  (`COMMIT_MODE='smoke'`); the scoring rerun on the hidden test runs on Kaggle's side and takes hours.

## Submissions

- **Current override (2026-11-05): submit only after the two local rounds and
  evidence of improvement over the best submitted method.** Do not spend the
  daily allowance just to fill it. See `docs/SCIENCE-2026-11-05.md`.
- Historical default, superseded by the current override: use all 5 submissions every UTC day.
- **2026-11-05 reserve:** keep at least **2 submissions available until the first
  batch has scored**, for later decisions before 00:00 UTC. P01/P02 use 2 of 5;
  3 remain. Do not send another batch before their scores just to fill the cap.
- **Submit at most 2 at a time** (user, 2026-11-02). When working a full day, spread the batches across the day so
  later picks can use earlier scores (scoring takes ~5 h); never send all 5 at once.
- Only submit kernel versions that completed (`KernelWorkerStatus.COMPLETE`) and wrote `submission.csv`.
- Log every submission in `SUBMISSIONS.md` (slot, kernel/version, change vs. its parent, ref, public score).
- Kaggle rate-limits polling: poll no faster than once per 60 s; run long waits in the background.
- Download outputs with `scripts/fetch_output.py <slot> <dest> [regex] [--log-only]`, not `kernels output`: the CLI
  lists 20 files per request and the pipeline writes ~2000, which trips HTTP 429 for an hour or more. The helper
  asks for 100 per page (the API rejects 1000) and backs off on 429.

## Run tracking

- **Add a row to `runs.csv` the moment a run is launched** (id, platform `kaggle-gpu|kaggle-tpu|kaggle-cpu|local-2070`,
  kind, ref = `<slot> v<N>` or the local log path, started_utc, status, expect = number of `DONE` lines for local
  logs, purpose) — `scripts/add_run.py <id> <platform> <kind> "<ref>" "<purpose>" [expect]` does it.
  **Fill `result` when it finishes** (the numbers, not "done"). Then run
  `.venv/Scripts/python.exe scripts/update_runs.py` to regenerate `RUNS.md`. Never edit `RUNS.md` by hand.
- During active work, keep `scripts/update_runs.py --loop` running in the background (hourly refresh). If `.local/compute-paused.json` exists, do not restart it or any training/watcher/controller. Remove the pause marker only after an explicit user request to resume project work; merely opening a session does not resume work.
- Conclusions belong in `STRATEGY.md`, scores in `SUBMISSIONS.md`; `runs.csv` holds the per-run facts.

## Layout

| Path | What |
|---|---|
| `kaggle/forks/<slot>/` | One private Kaggle notebook per experiment: `notebook.ipynb` + `kernel-metadata.json` (id `clarkkitchen/casmi26-<slot>`) |
| `RUNS.md` / `runs.csv` | **Every compute run** (Kaggle GPU/TPU/CPU, local 2070): what, where, when, status, result. Read `RUNS.md` first when resuming |
| `scripts/update_runs.py` | Refreshes statuses + quota in `runs.csv` and regenerates `RUNS.md`; runs hourly (`--loop`, background) |
| `scripts/local_seeds.py` | Scorer seeds on the local 2070 (`.venv-gpu`): holdout/refit training, ensemble diagnostics, `--split np` |
| `scripts/build_gpu_job.py` | Kaggle T4 ×2 notebook running `local_seeds.py` jobs, one list per GPU |
| `scripts/push_queue.py` | Pushes forks in order as GPU sessions free up (optional `needs` kernels), registers them in `runs.csv` |
| `kaggle/forks/tpu_train.py` | Refits on TPU v3-8 via PyTorch/XLA (probe: `x7-tpu-probe`) |
| `SUBMISSIONS.md` | Submission log and scores |
| `STRATEGY.md` | Where the score comes from, what has been tried, next ideas |
| `scripts/make_variant.py` | Builds a fork folder from a parent fork by exact-once string patches (re-pins the fragment-check SHA when patched) |
| `scripts/autopilot.py` | Background orchestrator: pushes queued forks when <2 GPU sessions run, submits completed versions under the daily cap. Queue: `kaggle/forks/queue.json`, log: `outputs/autopilot.log` |
| `cloud/workload.py`, `scripts/kaggle_cloud.py` | Our original pipeline (best 0.338). Superseded by the public lineage; kept for reference |
| `outputs/` | Downloaded kernel outputs (ignored by git) |

## Commands (PowerShell or Git Bash, from the repo root)

```bash
.venv/Scripts/python.exe -m kaggle kernels push -p kaggle/forks/<slot> --accelerator NvidiaTeslaT4 --timeout 32400
.venv/Scripts/python.exe -m kaggle kernels status clarkkitchen/casmi26-<slot>
.venv/Scripts/python.exe scripts/fetch_output.py <slot> outputs/forks/<slot> "<regex>"
.venv/Scripts/python.exe -m kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k clarkkitchen/casmi26-<slot> -v <N> -f submission.csv -m "<slot>: <change>"
.venv/Scripts/python.exe -m kaggle competitions submissions enveda-CASMI26-molecule-id-mass-spectra
```

Private datasets (locally trained weights): put the files and a `dataset-metadata.json` in a folder, `cd` into it and
run `kaggle datasets create -p .` — a multi-level relative `-p` path breaks the CLI's upload cache. New datasets are
private unless `--public` is passed.

New fork: add an entry to `VARIANTS` in `scripts/make_variant.py` (parent, patches) and run it; it keeps
`is_private: true`, `enable_internet: false`, the dataset list and the `docker_image` pin.

Variants built by `make_variant.py` get two safety patches by default:
- **Robust rerun detection**: `IS_RERUN` is also true when `test.parquet`'s content signature differs from the visible
  file's (`ab75c9ee…`, computed by `kaggle/forks/x2-test-signature`). The frozen pipeline receives it via
  `CASMI_FORCE_RERUN` (its source and SHA-256 pin are rewritten). A public author lost a submission to 0.000 when the
  rerun was not detected.
- **Fast commit** (`fast=True`): the commit run writes 25 unique dummy SMILES per row and skips every stage
  (a few GPU minutes instead of ~25); the scoring rerun runs everything. Note the official sample file repeats `CCO`
  25 times per row, so it is not used as the placeholder.

**A fast commit tests nothing.** Any new or changed code path (a new stage, a patched module, a patched `_post`)
must first pass a smoke commit (`COMMIT_MODE='smoke'`, 12 real molecules through every stage) before any variant
using it is submitted. t3/t4 of 2026-10-29 were fast commits of an untested consensus patch that overwrote the
molecule id; both scored 0.465 — a crash after the frozen pipeline stage does not error the submission, Kaggle scores
the baseline `submission.csv` already on disk. **A score of exactly 0.465 means "something crashed".**

## Credentials and sharing

### End-of-day override — 2026-11-05

The user explicitly requested: finish the current run, submit the strongest
validated candidate, pause for today, and stop all training/experiment runs
overnight. Do not start additional training after the current q75 dose trial.
One end-of-day submission is authorized even if P01/P02 remain pending; preserve
two unused daily slots. This is a one-time exception to the normal two-pending
batch rule. Stop the project watchers and hourly updater before pausing. Submitted
Kaggle scoring reruns continue so they can return scores. Do not auto-resume.

- The Kaggle token lives in `%USERPROFILE%\.kaggle\access_token` (also `kaggle key.txt`, git-ignored). Never print,
  log, commit or upload it; never put it in notebooks or metadata.
- Keep forks private. No GitHub pushes unless the user asks.
- Public-lineage assets are mostly CC BY-NC; credit the upstream authors in any notebook that is made public.

P14 v1 was NOT submitted. Its full GPU smoke passed, but the independent local/
Kaggle comparison exposed NumPy1/2 unstable quicksort selecting different tied
peaks at the top100 cutoff (4 spectra,3 queries; max score drift0.26495).
`score_frozen_channel.py` now specifies stable sorting, with exact NumPy1.26/2.0
preprocessing parity on all28 supported visible spectra. The original selected
seed/50% weight remain fixed; the robustness check still improves reserved
conditional MRR .637481→.742594, CI delta[.070929,.141013]. V2 was pushed for
another full real12 Kaggle smoke. Use the deterministic-sort audit and compare
Kaggle output with `outputs/local/frozen-p14-stable-smoke.parquet` (not v1's
local file). `submit_frozen_once.py --version 2` enforces the one remaining slot
and requires a fresh stable-sort audit. No third scoring trial is authorized.
