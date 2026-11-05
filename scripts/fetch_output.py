"""Download a kernel's run log and matching output files with as few listing requests as possible.

Usage: python scripts/fetch_output.py <slot> <dest> [regex] [--log-only]

The CLI's `kernels output` lists 20 files per request, and the pipeline notebooks write hundreds of files, so a
single fetch can trip the 429 limit on ListKernelSessionOutput. This asks for 100 per page (the API rejects 1000), saves the log from the
first page, and on 429 waits (Retry-After or 120 s) and retries the same page.
"""
import re
import sys
import time
from pathlib import Path

import requests
from requests.exceptions import HTTPError
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

OWNER = "clarkkitchen"


def list_page(api, slug, token):
    for attempt in range(30):
        try:
            with api.build_kaggle_client() as kaggle:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = OWNER
                req.kernel_slug = slug
                req.page_size = 100
                req.page_token = token or ""
                return kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        except HTTPError as e:
            if e.response is None or e.response.status_code != 429:
                raise
            wait = int(e.response.headers.get("Retry-After") or 120)
            print(f"429 on {slug} (attempt {attempt + 1}); waiting {wait} s", flush=True)
            time.sleep(wait)
    raise RuntimeError("still rate-limited after 30 attempts")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    log_only = "--log-only" in sys.argv
    slot, dest = args[0], Path(args[1])
    pattern = re.compile(args[2]) if len(args) > 2 else None
    slug = f"casmi26-{slot}"
    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    token, page, got = None, 0, []
    while True:
        resp = list_page(api, slug, token)
        page += 1
        if page == 1 and resp.log:
            (dest / f"{slug}.log").write_text(resp.log, encoding="utf-8")
            got.append(f"{slug}.log")
        if log_only:
            break
        for item in resp.files or []:
            if pattern and not pattern.search(item.file_name):
                continue
            out = dest / item.file_name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(requests.get(item.url, timeout=300).content)
            got.append(item.file_name)
        token = resp.next_page_token
        if not token:
            break
    print(f"{slot}: {page} listing page(s); saved {len(got)}: {' '.join(got)[:600]}", flush=True)


if __name__ == "__main__":
    main()
