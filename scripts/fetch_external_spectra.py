"""Download the two disclosed mirrors without loading their contents into RAM."""
import argparse
from pathlib import Path
import urllib.request

from build_external_cache import ROOT, SOURCES, digest_file


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/external/2026-11-06')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    for name,(url,expected,upstream) in SOURCES.items():
        path=args.out/(name+'.parquet')
        if not path.exists():
            partial=path.with_suffix('.part')
            with urllib.request.urlopen(url,timeout=60) as response,partial.open('wb') as f:
                for block in iter(lambda:response.read(4*1024*1024),b''):
                    f.write(block)
            assert digest_file(partial,'md5')==expected, f'{name} integrity failure'
            partial.replace(path)
        assert digest_file(path,'md5')==expected, f'{name} local file does not match pinned snapshot'
        manifest=args.out/(name+'-MIRROR.md')
        with urllib.request.urlopen(url.rsplit('/',1)[0]+'/MIRROR.md',timeout=30) as response:
            manifest.write_bytes(response.read())
        print(name,path.stat().st_size,expected,upstream,flush=True)


if __name__=='__main__':main()
