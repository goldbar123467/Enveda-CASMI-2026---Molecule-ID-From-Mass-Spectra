"""Read the public archive directory using bounded verified HTTP byte ranges.

No spectra are scored and no library annotation is accepted as truth here.
Partial archive requests cannot verify the published whole-file MD5; this is
explicitly a source inventory, not a validated acquisition download.
"""
import io,json,hashlib,urllib.request,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c47-remaining-tims-inventory-20261112/source-research'


class RemoteArchive(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.position=0;self.requests=[]
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('Negative archive position')
        return self.position
    def read(self,size=-1):
        if size<0:size=self.size-self.position
        size=min(size,self.size-self.position)
        assert 0<=size<=8*1024**2,'Inventory range exceeds8MiB cap'
        if size==0:return b''
        start=self.position;end=start+size-1
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=30) as response:
            assert response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}'
            data=response.read(size+1);assert len(data)==size
        self.requests.append(dict(start=start,end=end,bytes=size,sha256=hashlib.sha256(data).hexdigest()))
        self.position+=size;return data


def main():
    record=json.loads((OUT/'tims-bench-zenodo-record.json').read_text(encoding='utf8'));entry=record['files'][0]
    target=OUT/'tims-bench-archive-inventory.json';assert not target.exists()
    remote=RemoteArchive(entry['links']['self'],entry['size'])
    with zipfile.ZipFile(remote) as archive:
        files=[dict(path=v.filename,bytes=v.file_size,compressed_bytes=v.compress_size,CRC32=f'{v.CRC:08x}',compression=v.compress_type) for v in archive.infolist()]
        interesting=[f for f in files if 'library_spectra/' in f['path'] or 'cache/' in f['path'] or 'groundtruth_dataset/' in f['path']]
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_record='https://zenodo.org/records/20816379',archive_url=entry['links']['self'],archive_size=entry['size'],
        published_whole_archive_checksum=entry['checksum'],whole_archive_checksum_verified=False,
        file_count=len(files),files=files,range_receipts=remote.requests,
        source_quality_or_prediction_outcomes_read=False,C13_labels_read=False,qualified_validation=False)
    target.write_text(json.dumps(result,indent=2),encoding='utf8')
    print('TIMS-BENCH ARCHIVE',len(files),'files',sum(r['bytes'] for r in remote.requests),'bytes inspected')
    print(json.dumps(interesting,indent=2))


if __name__=='__main__':main()
