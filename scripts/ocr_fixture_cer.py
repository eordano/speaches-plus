#!/usr/bin/env python3
"""Runner for the cer_max_model / cer_max_model_neutral gates of the
071-ocr-layout fixtures: POSTs each fixture's input.png to a live /v1/ocr,
scores the reply raw and markup-neutral (ocr_cer_eval.markup_neutral, the port
of nv-ocr tests/e2e_fixtures.rs::markup_neutral) against expected_text, and
exits non-zero when a neutral gate fails. input.png is gitignored; render it
with conformance/tools/regenerate-fixtures.sh first."""

import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_cer_eval import edit_distance, markup_neutral  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'conformance', 'fixtures')

def post_ocr(url, png, fields, timeout):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="page.png"\r\n'
                 'Content-Type: image/png\r\n\r\n'.encode() + png + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    req = urllib.request.Request(url.rstrip('/') + '/v1/ocr', data=b''.join(parts), method='POST',
                                 headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8'), dict(r.headers)

def cer(ref, hyp):
    return edit_distance(ref, hyp) / len(ref) if ref else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8000')
    ap.add_argument('--backend', required=True)
    ap.add_argument('--mode', default='markdown')
    ap.add_argument('--format', default='text')
    ap.add_argument('--family', default='071-ocr-layout')
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--dump', help='directory to write each fixture hypothesis into')
    a = ap.parse_args()
    fields = {'backend': a.backend, 'mode': a.mode, 'format': a.format}
    failed = 0
    rows = []
    for d in sorted(glob.glob(os.path.join(FIXTURES, a.family + '-*'))):
        fx = json.load(open(os.path.join(d, 'fixture.json')))
        png_path = os.path.join(d, fx.get('input', 'input.png'))
        if not os.path.exists(png_path):
            print(f'{fx["name"]}: SKIP no {png_path} (render it first)', file=sys.stderr)
            continue
        t0 = time.time()
        try:
            hyp, headers = post_ocr(a.url, open(png_path, 'rb').read(), fields, a.timeout)
        except urllib.error.HTTPError as e:
            print(f'{fx["name"]}: HTTP {e.code} {e.read().decode(errors="replace")[:300]}')
            failed += 1
            continue
        secs = time.time() - t0
        if a.dump:
            os.makedirs(a.dump, exist_ok=True)
            open(os.path.join(a.dump, fx['name'] + '.txt'), 'w').write(hyp)
        ref = fx['expected_text']
        raw, neutral = cer(ref, hyp), cer(markup_neutral(ref), markup_neutral(hyp))
        gate = fx.get('gates', {}).get('cer_max_model_neutral')
        ok = gate is None or neutral <= gate
        failed += not ok
        flags = ','.join(k for k in ('x-ocr-truncated', 'x-ocr-looped', 'x-ocr-suspect')
                         if any(h.lower() == k for h in headers))
        rows.append((fx['name'], raw, neutral, gate, ok, secs, flags))
    print(f'backend={a.backend} mode={a.mode} format={a.format}')
    print(f'{"fixture":34} {"raw":>7} {"neutral":>8} {"gate":>6} {"secs":>7}  verdict')
    for name, raw, neutral, gate, ok, secs, flags in rows:
        g = f'{gate:.3f}' if gate is not None else '-'
        print(f'{name:34} {raw:7.4f} {neutral:8.4f} {g:>6} {secs:7.1f}  {"pass" if ok else "FAIL"} {flags}')
    if rows:
        print(f'{"mean":34} {sum(r[1] for r in rows) / len(rows):7.4f} '
              f'{sum(r[2] for r in rows) / len(rows):8.4f} {"":>6} {sum(r[5] for r in rows):7.1f}')
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
