"""Resume unchanged auxiliary downloader with a narrowly scoped SDK JSON fix.

SDK2.2.4 mistakes a legitimate downloaded JSON file's list-valued `code` field
for an API numeric error code. Preserve all HTTP/authentication errors, require
a200response from the selected public JSON filename, and leave the installed
SDK and the receipt-pinned downloader unchanged.
"""
import json,time
from pathlib import Path
from urllib.parse import urlparse,unquote
from kagglesdk.kaggle_http_client import KaggleHttpClient
import prepare_c75_aux_assets as worker
from c75_port_utils import read,write,sha,stamp


def main():
    out=worker.OUT;census=read(out/'aux-asset-inventory.json')
    allowed={Path(name).name for files in census['datasets'].values() for name in files if name.endswith('.json')}
    original=KaggleHttpClient._prepare_response;original_write=worker.write;repaired=[]
    def share_retry(path,value):
        for attempt in range(20):
            try:return original_write(path,value)
            except PermissionError:
                if attempt==19:raise
                time.sleep(min(1.,.1+.1*attempt))
    def numeric_code_only(client,response_type,response):
        if response.status_code==200 and 'application/json' in response.headers.get('Content-Type',''):
            data=response.json();url=urlparse(response.url);name=Path(unquote(url.path)).name
            file_host=(url.hostname or '').endswith(('.googleapis.com','.googleusercontent.com','.kaggleusercontent.com'))
            if isinstance(data,dict) and 'code' in data and not isinstance(data['code'],(int,float)) and name in allowed and file_host:
                response.raise_for_status()  # Never suppress any HTTP/authentication failure.
                repaired.append(dict(file_basename=name,code_field_type=type(data['code']).__name__,HTTP_status=response.status_code,bytes=len(response.content)))
                if client._response_processor:client._response_processor(response)
                return None if response_type is None else response_type.prepare_from(response)
        return original(client,response_type,response)
    write(out/'aux-transport-resume-intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),
        original_worker_sha256=sha(worker.__file__),inventory_sha256=sha(out/'aux-asset-inventory.json'),
        prior_terminal_failure='SDK2.2.4 TypeError: legitimate public PROVENANCE.json contains a list-valued code field',
        installed_SDK_unchanged=True,HTTP_and_authentication_errors_preserved=True,all_original_assets_and_sizes_required=True,
        no_quality_outcomes_read=True,C13_read=False,scoring_entries=0))
    KaggleHttpClient._prepare_response=numeric_code_only;worker.write=share_retry
    try:worker.main()
    finally:
        KaggleHttpClient._prepare_response=original;worker.write=original_write
        write(out/'aux-transport-repair-evidence.json',dict(checked_utc=stamp(),source_sha256=sha(__file__),
            original_worker_sha256=sha(worker.__file__),repaired_file_responses=repaired,
            installed_SDK_unchanged=True,HTTP_and_authentication_errors_preserved=True,
            query_or_asset_omission=False,no_quality_outcomes_read=True,C13_read=False,scoring_entries=0))


if __name__=='__main__':main()
