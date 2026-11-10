"""Check an actual C14 checkpoint's augmentation RNG without changing training."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import hashlib
import io
import json
from datetime import datetime,timezone

import torch

from precision_cache import ROOT,sha,write
from ce_mask import mask_known_ce


def main():
    torch.set_num_threads(2)
    out=ROOT/'outputs/local/c14-ce-mask-20261110'
    target=out/'rng-resume-audit.json';assert not target.exists()
    paths=list((out/'models/ce-mask').glob('fold-*/resume.pt'));assert paths
    path=max(paths,key=lambda p:p.stat().st_mtime)
    # Read one opened atomic-checkpoint snapshot. The trainer can continue and
    # replace its filename independently; the audited bytes remain fixed.
    blob=path.read_bytes();pin=hashlib.sha256(blob).hexdigest()
    state=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
    step,counts=state['step'],state['ce_counts'];rng=state['ce_mask_rng'].clone()
    del state,blob
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.1)
    first=torch.Generator(device='cuda');first.set_state(rng)
    second=torch.Generator(device='cuda');second.set_state(rng)
    metadata=torch.tensor([[.2,.3,1.,1.],[.4,.5,1.,-1.],[.6,0.,0.,1.]],device='cuda').repeat(128,1)
    before=metadata.clone();batch=(torch.zeros(384,1,4,device='cuda'),torch.ones(384,device='cuda',dtype=torch.long),metadata,torch.ones(384,device='cuda',dtype=torch.long))
    default=torch.cuda.get_rng_state().clone()
    a,sa=mask_known_ce(batch,.5,first);b,sb=mask_known_ce(batch,.5,second)
    assert torch.equal(a[2],b[2]) and sa==sb
    assert torch.equal(default,torch.cuda.get_rng_state()) and torch.equal(before,metadata)
    assert torch.equal(a[2][:,[0,3]],before[:,[0,3]])
    write(target,dict(checked_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
        checkpoint_snapshot_sha256=pin,checkpoint_path=str(path.relative_to(ROOT)),successful_update=step,
        observed_augmentation_counts=counts,device=torch.cuda.get_device_name(),
        restored_CUDA_masks_bit_exact=True,default_CUDA_rng_unchanged=True,input_metadata_unchanged=True,
        unaffected_fields_exact=True,trainer_checkpoint_not_modified=True,synthetic_queries=384,
        prediction_metrics_read=False,training_restarted=False))
    print('DONE ACTUAL C14 CUDA RNG RESUME AUDIT',json.dumps(dict(step=step,bit_exact=True)),flush=True)


if __name__=='__main__':main()
