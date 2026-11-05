"""Cloud-only E3 input compatibility check; no fitting or prediction."""
import argparse
import json
import pickle
from pathlib import Path
import workload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    proof = workload.require_server_cloud(args.authorization,"cpu")
    root = Path(proof["project_root"]).resolve()
    output = Path(args.output).resolve()
    if not output.is_relative_to(root/"outputs"):
        raise ValueError("Output must stay inside the authorized project")
    workload.INPUT_ROOT = root/"input"
    import numpy as np
    co = Path(workload.find_input("coco_fp.npy")).parent
    bio = Path(workload.find_input("bio_fp.npy")).parent
    with (co/"coco_meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    with (bio/"bio_meta.pkl").open("rb") as handle:
        added_meta = pickle.load(handle)
    fp = np.load(co/"coco_fp.npy",mmap_mode="r")
    added_fp = np.load(bio/"bio_fp.npy",mmap_mode="r")
    mass = np.load(bio/"bio_mass.npy",mmap_mode="r")
    keys = np.asarray(added_meta["keys"],dtype=object)
    smiles = np.asarray(added_meta["smiles"],dtype=object)
    invalid_keys = sum(not isinstance(k,str) or len(k)!=14 for k in keys)
    invalid_smiles = sum(not isinstance(s,str) or not s.strip() for s in smiles)
    same = (added_fp.ndim==2 and added_fp.shape[1]==fp.shape[1] and added_fp.dtype==fp.dtype
            and added_meta.get("nbits",meta["nbits"])==meta["nbits"]
            and len(added_fp)==len(mass)==len(keys)==len(smiles) and not invalid_keys and not invalid_smiles)
    valid = np.flatnonzero(np.isfinite(mass))
    novel = workload.unseen_candidate_rows(keys[valid],set(meta["keys"]))
    workload._init_fp_worker(np.load(co/"fp_bits.npy"))
    sample = np.unique(np.linspace(0,len(keys)-1,min(64,len(keys)),dtype=int))
    matches,failed,max_mass_ppm = 0,0,0.0
    for row in sample:
        generated = workload._fp_worker(smiles[row])
        if generated is None:
            failed += 1
            continue
        packed,generated_mass = generated
        matches += int(np.array_equal(packed,added_fp[row]))
        max_mass_ppm = max(max_mass_ppm,abs(generated_mass-float(mass[row]))/generated_mass*1e6)
    semantics = matches==len(sample) and not failed and max_mass_ppm<0.1
    result = {"source":"prvsiyan/chebi-lipidmaps-casmi26","same_fp_format":same,"nbits":meta["nbits"],
              "bio_shape":list(added_fp.shape),"bio_dtype":str(added_fp.dtype),"bio_rows":len(keys),
              "nonfinite_mass":len(keys)-len(valid),"novel_unique_before_training":len(novel),
              "duplicates_skipped":len(valid)-len(novel),"scoring_experiment_launched":False}
    result.update(invalid_keys=invalid_keys,invalid_smiles=invalid_smiles)
    result.update(fingerprint_sample_rows=len(sample),fingerprint_sample_exact_matches=matches,
                  fingerprint_sample_failed_smiles=failed,max_sample_mass_error_ppm=max_mass_ppm,
                  same_fingerprint_semantics=semantics)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    if not same or not semantics:
        raise RuntimeError("E3 input compatibility failed")


if __name__=="__main__":
    main()
