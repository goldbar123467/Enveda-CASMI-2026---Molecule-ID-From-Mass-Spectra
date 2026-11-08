"""Export verified encoder inference graphs and selected head for offline Kaggle."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
import torch
from torch import nn
import numpy as np
import pandas as pd
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/local/long-frozen-20261107-r1'
ENC=ROOT/'outputs/pretrained/frozen-20261107'
OUT=ROOT/'outputs/publish/frozen-p14-20261107'


def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*1024*1024):d.update(b)
    return d.hexdigest()


class MolecularWrapper(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,ids,mask):return self.model(input_ids=ids,attention_mask=mask).pooler_output


def unshare_for_trace(model):
    """JIT rejects aliased registered parameters; clone identical inference values."""
    modules={id(model)};count=0
    def visit(parent):
        nonlocal count
        for name,child in list(parent._modules.items()):
            if child is None:continue
            if id(child) in modules:
                child=copy.deepcopy(child);parent._modules[name]=child;count+=1
            modules.add(id(child));visit(child)
    visit(model);parameters=set()
    for module in model.modules():
        for name,p in list(module._parameters.items()):
            if p is None:continue
            if id(p) in parameters:
                module._parameters[name]=nn.Parameter(p.detach().clone(),requires_grad=False);count+=1
            parameters.add(id(module._parameters[name]))
    return count


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    em=json.loads((ENC/'manifest.json').read_text())
    for item in em['files']:assert sha(ENC/item['file'])==item['sha256']
    spec=json.loads((RUN/'manifest.json').read_text())
    for p,h in spec['source_sha256'].items():assert sha(ROOT/p)==h
    selection=json.loads((RUN/'qualification/selection.json').read_text())['winner']
    assert selection['model'] in ['202611071','202611072'],'Explicit single-head export required'
    receipt=json.loads((RUN/'models'/('seed-'+selection['model'])/'complete.json').read_text())
    head=RUN/'models'/('seed-'+selection['model'])/'best.pt';assert sha(head)==receipt['model_sha256']
    shutil.copy2(head,OUT/'head.pt');shutil.copy2(ROOT/'scripts/frozen_model.py',OUT/'frozen_model.py')
    import dreams.api as api
    from dreams.models.heads.heads import ContrastiveHead
    api.PRETRAINED=ENC/'dreams'
    model=api.PreTrainedModel.from_ckpt(ENC/'dreams/embedding_model.ckpt',ContrastiveHead,100).model.cuda().eval()
    model.requires_grad_(False)
    selected=pd.read_parquet(RUN/'selected.parquet',columns=['valid']);rows=np.flatnonzero(selected.valid.to_numpy())[:13]
    a=np.load(RUN/'spectra.npy',mmap_mode='r');sample=torch.tensor(np.asarray(a[rows]),device='cuda')
    with torch.inference_mode():
        pristine=model(sample).clone()
        aliases=unshare_for_trace(model)
        assert float((pristine-model(sample)).abs().max())==0
        graph=model.to_torchscript(method='trace',example_inputs=sample[:4],check_trace=False,strict=False)
        errors=[]
        for n in [1,3,8,13]:
            before=model(sample[:n]);after=graph(sample[:n]);error=float((before-after).abs().max())
            assert error<1e-4 and torch.isfinite(after).all(),error;errors.append(error)
        graph.save(str(OUT/'dreams.ts'))
    print('DREAMS EXPORTED',max(errors),flush=True)
    del graph,model;torch.cuda.empty_cache()
    from transformers import AutoModel,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ENC/'molformer',trust_remote_code=True,local_files_only=True)
    model=MolecularWrapper(AutoModel.from_pretrained(ENC/'molformer',trust_remote_code=True,local_files_only=True,deterministic_eval=True)).cuda().eval()
    model.requires_grad_(False)
    mol=pd.read_parquet(RUN/'molecules.parquet');smiles=mol.smiles.iloc[:13].tolist()
    def tokenize(values):
        v=tokenizer(values,padding='max_length',truncation=True,max_length=202,return_tensors='pt')
        return v['input_ids'].cuda(),v['attention_mask'].cuda()
    with torch.inference_mode():
        graph=torch.jit.trace(model,tokenize(smiles[:4]),check_trace=False,strict=False)
        molecular_errors=[]
        for n in [1,3,8,13]:
            inputs=tokenize(smiles[:n]);before=model(*inputs);after=graph(*inputs)
            error=float((before-after).abs().max());assert error<1e-4 and torch.isfinite(after).all(),error
            molecular_errors.append(error)
        # Check padding does not change the pooled representation materially.
        inputs=tokenizer(smiles,padding=True,truncation=True,max_length=202,return_tensors='pt')
        variable=model(inputs['input_ids'].cuda(),inputs['attention_mask'].cuda())
        padded=graph(*tokenize(smiles));padding_error=float((variable-padded).abs().max());assert padding_error<1e-3,padding_error
        graph.save(str(OUT/'molformer.ts'))
    shutil.copy2(ENC/'molformer/tokenizer.json',OUT/'tokenizer.json')
    mol[['key','smiles']].to_parquet(OUT/'molecules.parquet',index=False)
    # Private dataset is immutable; copying avoids writable aliases to training features.
    shutil.copy2(RUN/'molecule_embeddings.npy',OUT/'molecule_embeddings.npy')
    export=dict(torch_version=torch.__version__,identical_inference_alias_copies=aliases,dreams_max_abs_error=max(errors),molformer_max_abs_error=max(molecular_errors),
                molformer_padding_max_abs_error=padding_error,checked_batch_sizes=[1,3,8,13],dtype='float32 graphs; float16 feature storage',
                head=receipt,selection=selection,encoder_manifest_sha256=sha(ENC/'manifest.json'),run_manifest_sha256=sha(RUN/'manifest.json'))
    (OUT/'export_receipt.json').write_text(json.dumps(export,indent=2))
    shutil.copy2(ROOT/'outputs/vendor/DreaMS/LICENSE',OUT/'DREAMS-LICENSE.txt')
    shutil.copy2(ENC/'molformer/README.md',OUT/'MOLFORMER-MODEL-CARD.md')
    if not (OUT/'MOLFORMER-LICENSE.txt').exists():
        with urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt',timeout=30) as response:
            (OUT/'MOLFORMER-LICENSE.txt').write_bytes(response.read())
    names=['head.pt','frozen_model.py','dreams.ts','molformer.ts','tokenizer.json','molecules.parquet','molecule_embeddings.npy','export_receipt.json',
           'DREAMS-LICENSE.txt','MOLFORMER-MODEL-CARD.md','MOLFORMER-LICENSE.txt']
    manifest=dict(run_id='F07R1',selected_seed=int(selection['model']),fusion_weight=selection['weight'],
                  files={name:sha(OUT/name) for name in names},source_encoders=em,preprocessing='R1 positive single charge; precursor/fragment<=1000; top100+precursor1.1; float16 embeddings',
                  scope='Complement P01 final-list neural score only; unsupported input keeps P01',qualification='Development-only selection plus positive reserved conditional-ranking confirmation; no end-to-end recall claim')
    (OUT/'frozen-manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'dataset-metadata.json').write_text(json.dumps(dict(title='CASMI26 F07R1 frozen retrieval P14 20261107',id='clarkkitchen/casmi26-frozen-p14-20261107',licenses=[dict(name='CC-BY-NC-4.0')]),indent=2))
    shutil.copy2(ROOT/'NOTICE.md',OUT/'NOTICE.md')
    (OUT/'README.md').write_text('Private P14 inference package. Official DreaMS (MIT) and IBM MoLFormer (Apache-2.0) provenance is retained in frozen-manifest.json. New head trained on competition data only. Exported graphs contain identical author model computations; no teacher labels or external generated spectra. See repository docs/FROZEN-TRAINING-2026-11-07.md and export_receipt.json.\n')
    print('DONE FROZEN EXPORT',sha(OUT/'frozen-manifest.json'),json.dumps(export),flush=True)


if __name__=='__main__':main()
