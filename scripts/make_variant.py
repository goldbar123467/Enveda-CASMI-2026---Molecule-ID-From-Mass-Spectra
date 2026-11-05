"""Create a fork folder from a parent fork by exact-once string patches. Local packaging only; runs nothing."""
import ast
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "kaggle/forks"


def code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def patch(nb, old, new, count=1):
    """Replace `old` exactly `count` times across the code cells."""
    total = sum("".join(c["source"]).count(old) for c in code_cells(nb))
    assert total == count, f"expected {count} match(es), found {total}: {old[:80]!r}"
    for c in code_cells(nb):
        src = "".join(c["source"])
        if old in src:
            c["source"] = src.replace(old, new).splitlines(keepends=True)


def repin_post(nb):
    """Recompute the SHA-256 pin of the fragment-check `_post` source after patching it."""
    for c in code_cells(nb):
        src = "".join(c["source"])
        if "_post = " not in src or "write_text(_post)" not in src:
            continue
        tree = ast.parse(src)
        value = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                     and getattr(n.targets[0], "id", None) == "_post")
        line = next(l for l in src.splitlines() if "sha256(_post.encode())" in l)
        old_hash = line.split("== '")[1].split("'")[0]
        new_hash = hashlib.sha256(value.encode()).hexdigest()
        c["source"] = src.replace(old_hash, new_hash).splitlines(keepends=True)
        return new_hash
    raise AssertionError("fragment-check cell not found")


def patch_module(nb, name, old, new):
    """Exact-once patch inside a pinned helper module (MODULES[name]) and re-pin its MODULE_SHA256 entry."""
    for c in code_cells(nb):
        src = "".join(c["source"])
        if f"MODULES['{name}'] = " not in src:
            continue
        lines = src.split("\n")
        i = next(k for k, l in enumerate(lines) if l.startswith(f"MODULES['{name}'] = "))
        text = ast.literal_eval(lines[i].split(" = ", 1)[1])
        assert text.count(old) == 1, f"expected one match in {name}: {old[:80]!r}"
        text = text.replace(old, new)
        lines[i] = f"MODULES['{name}'] = {text!r}"
        src = "\n".join(lines)
        tree = ast.parse(src)
        pins = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", None) == "MODULE_SHA256")
        src = src.replace(pins[name], hashlib.sha256(text.encode()).hexdigest())
        c["source"] = src.splitlines(keepends=True)
        return
    raise AssertionError(f"module {name} not found")


FAST_EXIT = (
    "if not PIPELINE_DONE and not IS_RERUN and COMMIT_MODE == 'fast':\n"
    "    shutil.copyfile(COMP / 'sample_submission.csv', ROOT / 'submission.csv')\n"
    "    raise SystemExit('fast commit: sample file written, the scoring rerun runs every stage')\n"
)


def make_fast(nb):
    """Fast commit without an error exit: the commit writes the sample file and skips every later stage.

    The scoring rerun (IS_RERUN) is unaffected: FAST_DONE stays False and every stage runs as before.
    """
    patch(nb, "COMMIT_MODE = 'smoke'", "COMMIT_MODE = 'fast'")
    cells = code_cells(nb)
    run = next(i for i, c in enumerate(cells) if FAST_EXIT in "".join(c["source"]))
    head, rest = "".join(cells[run]["source"]).split(FAST_EXIT)
    body = "".join("    " + l if l.strip() else l for l in rest.splitlines(keepends=True))
    cells[run]["source"] = (
        head + "FAST_DONE = not PIPELINE_DONE and not IS_RERUN and COMMIT_MODE == 'fast'\n"
        "if FAST_DONE:\n"
        "    _dummy = pd.read_csv(COMP / 'sample_submission.csv', dtype=str)\n"
        "    _dummy['smiles'] = ';'.join('C' * (i + 1) + 'O' for i in range(25))\n"
        "    _dummy.to_csv(ROOT / 'submission.csv', index=False)\n"
        "    print('fast commit: unique dummy answers written, the scoring rerun runs every stage')\n"
        "else:\n" + body).splitlines(keepends=True)
    for k, c in enumerate(cells[run + 1:], run + 1):
        src = "".join(c["source"])
        c["source"] = (f"if not FAST_DONE:\n    exec(compile({src!r}, 'cell_{k}', 'exec'), globals())\n"
                       f"else:\n    print('fast commit: cell {k} skipped')\n").splitlines(keepends=True)


ID_GUARD = ("IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or "
            "TEST_SIGNATURE != '654e030e4173780e945252b4a5adf70b'\n")
FROZEN_GUARD = ("IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or "
                "_sig != '654e030e4173780e945252b4a5adf70b'\\n")
# Visible test.parquet content signature, computed by kaggle/forks/x2-test-signature with the same code.
VISIBLE_CONTENT_SIGNATURE = 'ab75c9eebd6f3148c810f643310584e3'


def robust_rerun(nb, content_sig):
    """Detect the hidden-test rerun by content as well as by env var and spectrum ids.

    The visible test holds copies of training spectra; the hidden test does not, so its content signature differs
    even if Kaggle's env var were missing and the spectrum ids matched. The frozen pipeline (a subprocess) reads the
    decision through CASMI_FORCE_RERUN; its source and SHA-256 pin are updated accordingly.
    """
    assert content_sig and len(content_sig) == 32
    patch(nb, ID_GUARD, (
        "def _content_signature(path):\n"
        "    t = pd.read_parquet(path, columns=['spectrum_id', 'precursor_mz', 'ms2_mzs'])\n"
        "    t = t.assign(spectrum_id=t.spectrum_id.astype(str)).sort_values('spectrum_id', kind='stable')\n"
        "    rows = [[s, round(float(p), 4), int(len(m)), round(float(np.sum(np.asarray(m, dtype=np.float64))), 3)]\n"
        "            for s, p, m in zip(t.spectrum_id, t.precursor_mz, t.ms2_mzs)]\n"
        "    return hashlib.md5(json.dumps(rows).encode()).hexdigest()\n"
        "TEST_CONTENT_SIGNATURE = _content_signature(COMP / 'test.parquet')\n"
        + ID_GUARD.rstrip("\n") + f" or TEST_CONTENT_SIGNATURE != '{content_sig}'\n"
        "os.environ['CASMI_FORCE_RERUN'] = '1' if IS_RERUN else ''\n"
        "print('RERUN DETECTION', IS_RERUN, 'env', bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')),\n"
        "      'ids', TEST_SIGNATURE, 'content', TEST_CONTENT_SIGNATURE, flush=True)\n"))
    cell = next(c for c in code_cells(nb) if "FROZEN_B64 = " in "".join(c["source"]))
    lines = "".join(cell["source"]).split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("FROZEN_B64 = "))
    j = next(k for k, l in enumerate(lines) if l.startswith("FROZEN_SHA256 = "))
    frozen = base64.b64decode(ast.literal_eval(lines[i].split(" = ", 1)[1]))
    old = FROZEN_GUARD.encode()
    assert frozen.count(old) == 1, "frozen rerun guard not found exactly once"
    frozen = frozen.replace(old, FROZEN_GUARD.replace(
        " or _sig", " or bool(os.getenv('CASMI_FORCE_RERUN')) or _sig").encode())
    lines[i] = f"FROZEN_B64 = {base64.b64encode(frozen).decode()!r}"
    lines[j] = f"FROZEN_SHA256 = {hashlib.sha256(frozen).hexdigest()!r}"
    cell["source"] = "\n".join(lines).splitlines(keepends=True)


def make(parent, slot, title, patches, post=False, modules=(), fast=False, robust=True, kernels=(), datasets=()):
    src, dst = ROOT / parent, ROOT / slot
    nb = json.loads((src / "notebook.ipynb").read_text(encoding="utf-8"))
    for p in patches:
        patch(nb, *p)
    for name, old, new in modules:
        patch_module(nb, name, old, new)
    if post:
        repin_post(nb)
    if robust:
        robust_rerun(nb, VISIBLE_CONTENT_SIGNATURE)
    if fast:
        make_fast(nb)
    for c in code_cells(nb):
        ast.parse("".join(c["source"]))
    meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id=f"clarkkitchen/casmi26-{slot}", title=title)
    meta["kernel_sources"] = sorted(set(meta.get("kernel_sources", [])) | set(kernels))
    meta["dataset_sources"] = meta.get("dataset_sources", []) + [d for d in datasets if d not in meta.get("dataset_sources", [])]
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", dst)


RELAX_MERGE = (
    "        elif sorted(p) != sorted(c):\n            reason = 'candidate_set_differs'\n",
    "        elif p[0] not in c:\n            reason = 'leader_not_in_custom'\n",
)
TAIL3 = ("CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)", "CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=3)")
LIB092 = ("if allow_top and lib<.87 and len(spec)>=3:", "if allow_top and lib<.92 and len(spec)>=3:")
# Looser rank-1 rule: library gate 0.92, two spectra suffice, contenders from ranks 2-8, neural margin 0.15 SD.
LOOSE = [("if allow_top and lib<.87 and len(spec)>=3:", "if allow_top and lib<.92 and len(spec)>=2:"),
         ("enumerate(control[1:min(6,n)],1)", "enumerate(control[1:min(8,n)],1)"),
         ("if not np.isfinite(nmargin) or nmargin<.23:continue", "if not np.isfinite(nmargin) or nmargin<.15:continue")]
# Neural scorer: average the refit checkpoint with the last train-split checkpoint of the same run.
CKPT_ENS_STAGE = ("BUNDLE, 'final_model.pt',", "BUNDLE, 'final_model.pt+last_validation_model.pt',", 2)
CKPT_ENS_MODULE = [
    ("neural_scores.py",
     "    model,feat,cfg=load_model(bundle,checkpoint,device)\n",
     "    models=[load_model(bundle,c,device) for c in checkpoint.split('+')];model,feat,cfg=models[0]\n"),
    ("neural_scores.py",
     "            F=torch.tensor(np.stack(fp),device=device,dtype=torch.float32);M=model.encode_molecules(F)\n"
     "            cos_all=[];fp_all=[]\n"
     "            for st in range(0,len(records),64):\n"
     "                b=records[st:st+64]\n"
     "                x=torch.tensor(np.stack([v[0] for v in b]),device=device)\n"
     "                lengths=torch.tensor([v[1] for v in b],device=device,dtype=torch.long)\n"
     "                meta=torch.tensor(np.stack([v[2] for v in b]),device=device)\n"
     "                adduct=torch.tensor([v[3] for v in b],device=device,dtype=torch.long)\n"
     "                Z,P=model.encode_spectra(x,lengths,meta,adduct)\n"
     "                cos_all.append((Z@M.T).cpu().numpy());fp_all.append((P@F.T/cfg['fp_bits']).cpu().numpy())\n"
     "            cos=np.concatenate(cos_all);fps=np.concatenate(fp_all)\n",
     "            F=torch.tensor(np.stack(fp),device=device,dtype=torch.float32);cos_m=[];fps_m=[]\n"
     "            for model,_,_ in models:\n"
     "                M=model.encode_molecules(F);cos_all=[];fp_all=[]\n"
     "                for st in range(0,len(records),64):\n"
     "                    b=records[st:st+64]\n"
     "                    x=torch.tensor(np.stack([v[0] for v in b]),device=device)\n"
     "                    lengths=torch.tensor([v[1] for v in b],device=device,dtype=torch.long)\n"
     "                    meta=torch.tensor(np.stack([v[2] for v in b]),device=device)\n"
     "                    adduct=torch.tensor([v[3] for v in b],device=device,dtype=torch.long)\n"
     "                    Z,P=model.encode_spectra(x,lengths,meta,adduct)\n"
     "                    cos_all.append((Z@M.T).cpu().numpy());fp_all.append((P@F.T/cfg['fp_bits']).cpu().numpy())\n"
     "                cos_m.append(np.concatenate(cos_all));fps_m.append(np.concatenate(fp_all))\n"
     "            cos=np.mean(cos_m,0);fps=np.mean(fps_m,0)\n"),
]

# Neural scorer seed ensemble: the original final_model.pt plus the full-data refits of other seeds that
# `n1-scorer-seeds` trains (seeds/casmi26_seed_<k>.pt in that kernel's output, attached as a kernel source).
# Without those files the list is just final_model.pt, i.e. s3.
SEED_ENS = [
    ("def score_final_lists():\n",
     "NEURAL_CKPTS = '+'.join(['final_model.pt'] + sorted(str(p) for p in Path('/kaggle/input').rglob('casmi26_seed_*.pt')))\n"
     "print('NEURAL CHECKPOINTS', NEURAL_CKPTS.count('+') + 1, NEURAL_CKPTS, flush=True)\n\n\n"
     "def score_final_lists():\n"),
    ("BUNDLE, 'final_model.pt',", "BUNDLE, NEURAL_CKPTS,", 2),
]
# SEED_ENS without the kit's 16-epoch final_model.pt: only the attached casmi26_seed_*.pt (absolute paths;
# load_model's Path(bundle)/checkpoint keeps an absolute path as is). Changed code, so smoke-commit it.
SEED_ENS_ONLY = [
    (SEED_ENS[0][0], SEED_ENS[0][1].replace("['final_model.pt'] + sorted(", "sorted(")),
    SEED_ENS[1],
]
# Checkpoints trained with `local_seeds.py --set key=value` carry `cfg_overrides` (e.g. width/heads/layers); build each
# model from the kit config updated with them. Kit-shaped checkpoints have none, so they load as before.
CFG_OVERRIDE_MODULE = [
    ("neural_scores.py",
     "    cfg=json.loads((Path(bundle)/'config.json').read_text());model=cls(cfg).to(device)\n"
     "    ck=torch.load(Path(bundle)/checkpoint,map_location=device,weights_only=True)\n",
     "    cfg=json.loads((Path(bundle)/'config.json').read_text())\n"
     "    ck=torch.load(Path(bundle)/checkpoint,map_location=device,weights_only=True)\n"
     "    cfg.update(ck.get('cfg_overrides') or {});model=cls(cfg).to(device)\n"),
]
MID10 = ("CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)", "CUSTOM = dict(w_top=0.25, w_mid=1.00, rescue=0)")


def SINGLE(name):
    """Narrow SEED_ENS(_ONLY)'s checkpoint glob to one file name."""
    return ("rglob('casmi26_seed_*.pt')", f"rglob({name!r})")
TOP35 = ("CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)", "CUSTOM = dict(w_top=0.35, w_mid=0.75, rescue=0)")
TOP50 = ("CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)", "CUSTOM = dict(w_top=0.50, w_mid=0.75, rescue=0)")
E128X8 = ["clarkkitchen/casmi26-n7a-e128-refits", "clarkkitchen/casmi26-n7b-e128-refits",
          "clarkkitchen/casmi26-n11-e128-refits"]
E128X12 = E128X8 + ["clarkkitchen/casmi26-n14-e128-refits"]

FPVOTE_SRC = (Path(__file__).resolve().parents[1] / "kaggle/forks/fpvote.py").read_text(encoding="utf-8")


def fpvote_patches(lam):
    """Fold-safe FPNet vote blended into the ranks 2-25 order of CUSTOM: z(neural) + lam * z(f.z)."""
    run_old = "OUTSIDERS_OK = score_outside_candidates(needed=True)\n"
    run_new = run_old + (
        f"(CODE / 'fpvote.py').write_text({FPVOTE_SRC!r})\n"
        "try:\n"
        "    run_stage([sys.executable, CODE / 'fpvote.py', ROOT / 'neural_candidate_features.parquet',\n"
        "               ROOT / 'effective_query_spectra.parquet', ROOT / 'neural_candidate_features.parquet', 'cuda'],\n"
        "              ROOT / 'fpvote.log', 'Fold-safe FPNet vote')\n"
        "except Exception as exc:\n"
        "    print('FPVOTE failed; CUSTOM falls back to the neural order:', repr(exc)[:300], flush=True)\n")
    stage = ("rescue=int(CUSTOM.get('rescue', 0)), label='Custom weights')",
             "rescue=int(CUSTOM.get('rescue', 0)), label='Custom weights',\n"
             "                                     fp_lambda=float(CUSTOM.get('fp_lambda', 0)))")
    config = ("CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)", f"CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0, fp_lambda={lam})")
    modules = [
        ("consensus.py", "def leader_safe_order(neural_scores, w_top, w_mid, offset=3.0):",
         "def leader_safe_order(neural_scores, w_top, w_mid, offset=3.0, mid_scores=None):"),
        ("consensus.py", "    if w_mid == w_top:\n        return first\n",
         "    if w_mid == w_top and mid_scores is None:\n        return first\n"),
        ("consensus.py", "    rest = [int(i) for i in rrf_order(s, w_mid, offset) if int(i) != lead]\n",
         "    m = s if mid_scores is None else np.asarray(mid_scores, dtype=float)\n"
         "    rest = [int(i) for i in rrf_order(m, w_mid, offset) if int(i) != lead]\n"),
        ("consensus.py",
         "                order = leader_safe_order(g.neural_score.values, cfg['w_top'], cfg['w_mid'], offset)\n",
         # `mid` is the molecule id in build_variants; the blended scores need another name (bug in t3/t4 v1).
         "                lam = float(cfg.get('fp_lambda', 0) or 0)\n"
         "                fp_mid = None\n"
         "                if lam and 'fp_vote_z' in g and g.fp_vote.notna().any():\n"
         "                    v = g.neural_score.astype(float).values\n"
         "                    fp_mid = (v - v.mean()) / max(v.std(), 1e-6) + lam * g.fp_vote_z.astype(float).values\n"
         "                order = leader_safe_order(g.neural_score.values, cfg['w_top'], cfg['w_mid'], offset, fp_mid)\n"),
    ]
    return [(run_old, run_new), stage, config], modules


def fpgate_patches(lib):
    """Fragment-check rank-1 moves need the fold-safe FPNet vote to prefer the contender over the leader; with that
    agreement the library gate opens to `lib`. Without FPNet scores the 0.87 rule is unchanged. Patches `_post`
    (a repr literal, hence the escaped newlines); needs fpvote_patches for the scores and post=True for the pin."""
    return [
        ("    if allow_top and lib<.87 and len(spec)>=3:",
         "    has_fp='fp_vote_z' in listed.columns and bool(listed.fp_vote_z.notna().all())\\n"
         f"    if allow_top and lib<({lib} if has_fp else .87) and len(spec)>=3:"),
        ("finite(ev.neural_fp)<finite(lead_ev.neural_fp)-.012:continue",
         "finite(ev.neural_fp)<finite(lead_ev.neural_fp)-.012:continue\\n"
         "            if has_fp and not finite(ev.fp_vote_z)>finite(lead_ev.fp_vote_z):continue"),
    ]


# Ahmed Berat Ozer's v33 (`p-v33`, upstream copy) guards its own fast commit; it copies the sample file, whose rows
# repeat one SMILES 25 times. Write unique placeholders instead, as make_fast does.
V33_PLACEHOLDER = (
    "    shutil.copyfile(COMP/'sample_submission.csv', ROOT/'submission.csv')\n",
    "    _dummy = pd.read_csv(COMP / 'sample_submission.csv', dtype=str)\n"
    "    _dummy['smiles'] = ';'.join('C' * (i + 1) + 'O' for i in range(25))\n"
    "    _dummy.to_csv(ROOT / 'submission.csv', index=False)\n")

VARIANTS = {
    "s3-relaxmerge": dict(parent="s1-p477", title="casmi26 s3 relaxmerge", patches=[RELAX_MERGE], fast=True),
    "s4-relaxmerge-tail3": dict(parent="s1-p477", title="casmi26 s4 relaxmerge tail3", patches=[RELAX_MERGE, TAIL3],
                                fast=True),
    "s5-relaxmerge-lib092": dict(parent="s1-p477", title="casmi26 s5 relaxmerge lib092", patches=[RELAX_MERGE, LIB092],
                                 post=True, fast=True),
    # 2026-10-29 candidates
    "t1-loose": dict(parent="s1-p477", title="casmi26 t1 loose", patches=[RELAX_MERGE, *LOOSE], post=True, fast=True),
    "t2-ckptens": dict(parent="s1-p477", title="casmi26 t2 ckptens", patches=[RELAX_MERGE, CKPT_ENS_STAGE],
                       modules=CKPT_ENS_MODULE, fast=True),
    "t3-fpvote": dict(parent="s1-p477", title="casmi26 t3 fpvote", patches=[RELAX_MERGE, *fpvote_patches(0.5)[0]],
                      modules=fpvote_patches(0.5)[1], fast=True),
    "t4-fpvote10": dict(parent="s1-p477", title="casmi26 t4 fpvote10", patches=[RELAX_MERGE, *fpvote_patches(1.0)[0]],
                        modules=fpvote_patches(1.0)[1], fast=True),
    "t5-loose-fpvote": dict(parent="s1-p477", title="casmi26 t5 loose fpvote",
                            patches=[RELAX_MERGE, *LOOSE, *fpvote_patches(0.5)[0]], modules=fpvote_patches(0.5)[1],
                            post=True, fast=True),
    # 2026-10-29 slot-5 candidates. u3 commits in smoke mode: it runs both FPNet code paths on real pipeline files.
    "u1-fpgate092": dict(parent="s1-p477", title="casmi26 u1 fpgate092",
                         patches=[RELAX_MERGE, *fpvote_patches(0)[0], *fpgate_patches(.92)],
                         modules=fpvote_patches(0)[1], post=True, fast=True),
    "u2-fpveto": dict(parent="s1-p477", title="casmi26 u2 fpveto",
                      patches=[RELAX_MERGE, *fpvote_patches(0)[0], *fpgate_patches(.87)],
                      modules=fpvote_patches(0)[1], post=True, fast=True),
    "u3-fpgate092-fp05": dict(parent="s1-p477", title="casmi26 u3 fpgate092 fp05",
                              patches=[RELAX_MERGE, *fpvote_patches(.5)[0], *fpgate_patches(.92)],
                              modules=fpvote_patches(.5)[1], post=True),
    # t3/t4 with the build_variants fix (v1 overwrote the molecule id; their 10-29 submissions will fail).
    "t3b-fpvote": dict(parent="s1-p477", title="casmi26 t3b fpvote", patches=[RELAX_MERGE, *fpvote_patches(0.5)[0]],
                       modules=fpvote_patches(0.5)[1], fast=True),
    "t4b-fpvote10": dict(parent="s1-p477", title="casmi26 t4b fpvote10", patches=[RELAX_MERGE, *fpvote_patches(1.0)[0]],
                         modules=fpvote_patches(1.0)[1], fast=True),
    # v33 real-list residual (ranks 2-25 where library < 0.87); smoke commit validates the residual on real files.
    "v1-v33residual": dict(parent="p-v33", title="casmi26 v1 v33residual", patches=[V33_PLACEHOLDER]),
    # Neural scorer seed ensemble (5 models); smoke commit, the checkpoint-list code path is new.
    "w1-seedens": dict(parent="s1-p477", title="casmi26 w1 seedens", patches=[RELAX_MERGE, *SEED_ENS],
                       modules=CKPT_ENS_MODULE, kernels=["clarkkitchen/casmi26-n1-scorer-seeds"]),
    # w1 with n2's seeds 5-8 as well: 9 models.
    "w3-seedens9": dict(parent="s1-p477", title="casmi26 w3 seedens9", patches=[RELAX_MERGE, *SEED_ENS],
                        modules=CKPT_ENS_MODULE,
                        kernels=["clarkkitchen/casmi26-n1-scorer-seeds", "clarkkitchen/casmi26-n2-scorer-seeds"]),
    # w3 with ranks 2-25 at neural weight 1.00 (rank 1 unchanged, so the merge's V2 == CUSTOM leader test still holds).
    # Only a constant changes on w3's smoke-tested path, so a fast commit.
    "w4-seedens9-mid10": dict(parent="s1-p477", title="casmi26 w4 seedens9 mid10",
                              patches=[RELAX_MERGE, *SEED_ENS, MID10], modules=CKPT_ENS_MODULE, fast=True,
                              kernels=["clarkkitchen/casmi26-n1-scorer-seeds", "clarkkitchen/casmi26-n2-scorer-seeds"]),
    # w3 + n3's seeds 9-16: 17 models. Only the attached checkpoints differ from w3's smoke-tested notebook.
    "w5-seedens17": dict(parent="s1-p477", title="casmi26 w5 seedens17", patches=[RELAX_MERGE, *SEED_ENS],
                         modules=CKPT_ENS_MODULE, fast=True,
                         kernels=["clarkkitchen/casmi26-n1-scorer-seeds", "clarkkitchen/casmi26-n2-scorer-seeds",
                                  "clarkkitchen/casmi26-n3-scorer-seeds"]),
    # Neural scorer averaged over final_model.pt + 8 full-data refits trained for 32 epochs instead of 16 on the
    # workstation GPU (`scripts/local_seeds.py refit 101..108 --epochs 32`, private dataset casmi26-seeds-e32).
    # On the natural-product holdout one 32-epoch model scored 0.428 against 0.382 for 16-epoch models (mean of 5)
    # and 0.405 for their 5-model average. Only the attached checkpoints differ from w1's smoke-tested notebook.
    "w6-seedens-e32": dict(parent="s1-p477", title="casmi26 w6 seedens e32", patches=[RELAX_MERGE, *SEED_ENS],
                           modules=CKPT_ENS_MODULE, fast=True, datasets=["clarkkitchen/casmi26-seeds-e32"]),
    # Eight 64-epoch full-data refits only (no 16-epoch final_model.pt): 201-204 local (dataset casmi26-seeds-e64),
    # 211-214 Kaggle (n6). Holdout MRR at 64 epochs: kit 0.661, natural products 0.550 (16 epochs: 0.484 / 0.382).
    "w7-seedens-e64": dict(parent="s1-p477", title="casmi26 w7 seedens e64", patches=[RELAX_MERGE, *SEED_ENS_ONLY],
                           modules=CKPT_ENS_MODULE, datasets=["clarkkitchen/casmi26-seeds-e64"],
                           kernels=["clarkkitchen/casmi26-n6-e64-refits"]),
    # Four 128-epoch full-data refits only (n7a, n7b); the long end of the 16/32/64/128 epoch dose-response.
    "w8-seedens-e128": dict(parent="s1-p477", title="casmi26 w8 seedens e128", patches=[RELAX_MERGE, *SEED_ENS_ONLY],
                            modules=CKPT_ENS_MODULE,
                            kernels=["clarkkitchen/casmi26-n7a-e128-refits", "clarkkitchen/casmi26-n7b-e128-refits"]),
    # 2026-11-03 slate candidates (built 11-02 while the 10-30 epoch slate scores). Only the attached checkpoints or a
    # CUSTOM constant differ from w8's smoke-tested notebook, so fast commits.
    # Eight 128-epoch refits: n7a/n7b (221-224) + n11 (225-228).
    "w9-e128x8": dict(parent="s1-p477", title="casmi26 w9 e128x8", patches=[RELAX_MERGE, *SEED_ENS_ONLY],
                      modules=CKPT_ENS_MODULE, fast=True, kernels=E128X8),
    # Four 256-epoch refits: n12 (241-242) + local (243-244, dataset casmi26-seeds-e256). n9: NP holdout 0.58 at 128
    # epochs -> 0.61 at 256; the random holdout is flat from ~160.
    "w10-e256x4": dict(parent="s1-p477", title="casmi26 w10 e256x4", patches=[RELAX_MERGE, *SEED_ENS_ONLY],
                       modules=CKPT_ENS_MODULE, fast=True, kernels=["clarkkitchen/casmi26-n12-e256-refits"],
                       datasets=["clarkkitchen/casmi26-seeds-e256"]),
    # w9 + w10: twelve long-trained models.
    "w11-long12": dict(parent="s1-p477", title="casmi26 w11 long12", patches=[RELAX_MERGE, *SEED_ENS_ONLY],
                       modules=CKPT_ENS_MODULE, fast=True, kernels=E128X8 + ["clarkkitchen/casmi26-n12-e256-refits"],
                       datasets=["clarkkitchen/casmi26-seeds-e256"]),
    # w9 with ranks 2-25 at neural weight 1.00 (rank 1 unchanged).
    "w12-e128x8-mid10": dict(parent="s1-p477", title="casmi26 w12 e128x8 mid10",
                             patches=[RELAX_MERGE, *SEED_ENS_ONLY, MID10], modules=CKPT_ENS_MODULE, fast=True,
                             kernels=E128X8),
    # w9 with rank-1 neural weight 0.35: the #2 candidate takes rank 1 when the neural ranks it first and the leader
    # 7th or lower (0.25: 18th or lower). Where CUSTOM's leader differs from V2's, the merge keeps CUSTOM as is.
    "w13-e128x8-top35": dict(parent="s1-p477", title="casmi26 w13 e128x8 top35",
                             patches=[RELAX_MERGE, *SEED_ENS_ONLY, TOP35], modules=CKPT_ENS_MODULE, fast=True,
                             kernels=E128X8),
    # w11 (8 x e128 + 4 x e256) with rank-1 neural weight 0.35 / 0.50. Panel replay (scripts/panel_replay.py, kit-split
    # holdout models): 0.35 moves 2-5 of 120 leaders and gains with every model set (e128 +2/-0); 0.50 loses with
    # 16-epoch models (+3/-5, as on the board) but gains with e128+e256 (+4/-1).
    "w15-long12-top35": dict(parent="s1-p477", title="casmi26 w15 long12 top35",
                             patches=[RELAX_MERGE, *SEED_ENS_ONLY, TOP35], modules=CKPT_ENS_MODULE, fast=True,
                             kernels=E128X8 + ["clarkkitchen/casmi26-n12-e256-refits"],
                             datasets=["clarkkitchen/casmi26-seeds-e256"]),
    "w16-long12-top50": dict(parent="s1-p477", title="casmi26 w16 long12 top50",
                             patches=[RELAX_MERGE, *SEED_ENS_ONLY, TOP50], modules=CKPT_ENS_MODULE, fast=True,
                             kernels=E128X8 + ["clarkkitchen/casmi26-n12-e256-refits"],
                             datasets=["clarkkitchen/casmi26-seeds-e256"]),
    # Twelve 128-epoch refits: w9 + n14 (229-232).
    "w17-e128x12": dict(parent="s1-p477", title="casmi26 w17 e128x12", patches=[RELAX_MERGE, *SEED_ENS_ONLY],
                        modules=CKPT_ENS_MODULE, fast=True, kernels=E128X12),
    # w17 + two width-256 / 4-layer refits (n16; n13: +0.02 holdout MRR at 64 epochs). The checkpoints carry
    # cfg_overrides, so load_model is patched: a new code path, hence a smoke commit.
    "w18-e128x12-w256x2": dict(parent="s1-p477", title="casmi26 w18 e128x12 w256x2", patches=[RELAX_MERGE, *SEED_ENS_ONLY],
                               modules=CKPT_ENS_MODULE + CFG_OVERRIDE_MODULE,
                               kernels=E128X12 + ["clarkkitchen/casmi26-n16-w256-refits"]),
    # w18 (smoke-tested 11-02: 14 checkpoints, 0 input failures) at rank-1 neural weight 0.35.
    "w19-e128x12-w256x2-top35": dict(parent="s1-p477", title="casmi26 w19 e128x12 w256x2 top35",
                                     patches=[RELAX_MERGE, *SEED_ENS_ONLY, TOP35],
                                     modules=CKPT_ENS_MODULE + CFG_OVERRIDE_MODULE, fast=True,
                                     kernels=E128X12 + ["clarkkitchen/casmi26-n16-w256-refits"]),
    # 11-02 board: every ensemble of our refits scored ~0.459 (w3 16 ep x 9 incl. the kit model 0.460, w6 e32 0.458,
    # w8 e128 0.459) against 0.477 for the kit model alone. Single-model diagnostics: does one of OUR refits alone,
    # same recipe as the kit (16 epochs) or long-trained (128), also lose? Same code path as w8 (one checkpoint).
    "w20-seed1-e16": dict(parent="s1-p477", title="casmi26 w20 seed1 e16",
                          patches=[RELAX_MERGE, *SEED_ENS_ONLY, SINGLE("casmi26_seed_1.pt")], modules=CKPT_ENS_MODULE,
                          fast=True, kernels=["clarkkitchen/casmi26-n1-scorer-seeds"]),
    "w21-seed221-e128": dict(parent="s1-p477", title="casmi26 w21 seed221 e128",
                             patches=[RELAX_MERGE, *SEED_ENS_ONLY, SINGLE("casmi26_seed_221_e128.pt")],
                             modules=CKPT_ENS_MODULE, fast=True, kernels=["clarkkitchen/casmi26-n7a-e128-refits"]),
    # Kit model at half weight: final_model.pt listed 12 times next to the 12 x e128 refits (uniform average).
    "w22-kit50-e128x12": dict(parent="s1-p477", title="casmi26 w22 kit50 e128x12",
                              patches=[RELAX_MERGE, *SEED_ENS, ("['final_model.pt'] + sorted(", "['final_model.pt'] * 12 + sorted(")],
                              modules=CKPT_ENS_MODULE, fast=True, kernels=E128X12),
    # Every refit at 32+ epochs plus the kit's final_model.pt: 1 + 8 x e32 + 8 x e64 + 8 x e128.
    "w14-mix25": dict(parent="s1-p477", title="casmi26 w14 mix25", patches=[RELAX_MERGE, *SEED_ENS],
                      modules=CKPT_ENS_MODULE, fast=True,
                      kernels=E128X8 + ["clarkkitchen/casmi26-n6-e64-refits"],
                      datasets=["clarkkitchen/casmi26-seeds-e32", "clarkkitchen/casmi26-seeds-e64"]),
    # w1 + FPNet vote 0.5 (t3b); for when both score above s3.
    "w2-seedens-fp05": dict(parent="s1-p477", title="casmi26 w2 seedens fp05",
                            patches=[RELAX_MERGE, *SEED_ENS, *fpvote_patches(.5)[0]],
                            modules=CKPT_ENS_MODULE + fpvote_patches(.5)[1],
                            kernels=["clarkkitchen/casmi26-n1-scorer-seeds"]),
}

if __name__ == "__main__":
    for slot in sys.argv[1:] or VARIANTS:
        spec = dict(VARIANTS[slot])
        make(spec.pop("parent"), slot, spec.pop("title"), spec.pop("patches"), **spec)
