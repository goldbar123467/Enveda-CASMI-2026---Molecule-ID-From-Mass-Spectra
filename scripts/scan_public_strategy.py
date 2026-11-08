"""Snapshot public notebook TEXT and receipts, without running downloaded code.

Use --refresh for a fresh authenticated Kaggle catalogue and selected sources.
The API's sorting/title is not proof of a leaderboard score. Credentials are
handled by the installed CLI and are never written to these output receipts.
"""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COMP='enveda-CASMI26-molecule-id-mass-spectra'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=True),encoding='utf-8')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refs',type=Path,default=ROOT/'configs/public-notebooks-20261108.json')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/research/2026-11-08-scan')
    parser.add_argument('--evidence',type=Path,default=ROOT/'docs/evidence/scan-2026-11-08')
    parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
    refs=json.loads(args.refs.read_text());args.out.mkdir(parents=True,exist_ok=True)
    if args.refresh:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate();catalogue={}
        for sort in ['dateRun','scoreDescending','voteCount']:
            rows=api.kernels_list(page=1,page_size=100,competition=COMP,sort_by=sort)
            rows=[r.to_json() for r in rows];save(args.out/f'{sort}.json',rows)
            catalogue.update({r['ref']:r for r in rows if r.get('ref')})
        save(args.out/'union.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),kernels=list(catalogue.values())))
        for ref in refs:
            folder=args.out/'notebooks'/ref.replace('/','__');folder.mkdir(parents=True,exist_ok=True)
            api.kernels_pull(ref,str(folder),metadata=True,quiet=True)
    catalogue=json.loads((args.out/'union.json').read_text());sources=[]
    for ref in refs:
        folder=args.out/'notebooks'/ref.replace('/','__');metadata=json.loads((folder/'kernel-metadata.json').read_text())
        file=folder/metadata['code_file']
        if file.suffix=='.ipynb':
            notebook=json.loads(file.read_text(encoding='utf-8'));cells=notebook['cells']
            code='\n\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='code')
            narrative='\n\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='markdown')
        else:code=file.read_text(encoding='utf-8');narrative=''
        (folder/'source.py').write_text(code,encoding='utf-8');(folder/'narrative.md').write_text(narrative,encoding='utf-8')
        sources.append(dict(ref=ref,url='https://www.kaggle.com/code/'+ref,code_file=file.name,sha256=digest(file),
            source_sha256=digest(folder/'source.py'),bytes=file.stat().st_size,datasets=metadata.get('dataset_sources',[]),
            kernels=metadata.get('kernel_sources',[]),docker=metadata.get('docker_image'),verified_public_score=None))
    index=dict(checked_utc=catalogue['checked_utc'],catalogue_entries=len(catalogue['kernels']),source_count=len(sources),sources=sources,
               limitation='Three first-page sorts,100 each; not an exhaustive catalogue.20 selected sources read as text. No downloaded notebook code executed. Titles and sort order do not verify scores.')
    save(args.evidence/'source-index.json',index)
    save(args.evidence/'catalogue.json',dict(checked_utc=catalogue['checked_utc'],kernels=catalogue['kernels']))
    print(json.dumps(dict(catalogue_entries=index['catalogue_entries'],source_count=len(sources),evidence=str(args.evidence))))


if __name__=='__main__':main()
