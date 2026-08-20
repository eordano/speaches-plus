#!/usr/bin/env python3
import http.client, os, statistics, threading, time, uuid
from pathlib import Path

HOST = os.environ.get("BENCH_HOST", "127.0.0.1")
PORT = int(os.environ.get("BENCH_PORT", "18801"))
MODEL = os.environ.get("BENCH_MODEL", "")
FIXTURES = sorted(Path("/tmp/sp-fixtures").glob("p*.wav"))
assert len(FIXTURES) == 21

def post(p):
    b = uuid.uuid4().hex
    parts = [(f"--{b}\r\n"
              f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
              "Content-Type: audio/wav\r\n\r\n").encode(), p.read_bytes()]
    if MODEL:
        parts.append(f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{MODEL}".encode())
    parts.append(f"\r\n--{b}--\r\n".encode())
    body = b"".join(parts)
    t0 = time.monotonic()
    c = http.client.HTTPConnection(HOST, PORT, timeout=180)
    c.request("POST", "/v1/audio/transcriptions", body=body, headers={
        "Content-Type": f"multipart/form-data; boundary={b}",
        "Content-Length": str(len(body)),
        "Authorization": "Bearer dummy"})
    r = c.getresponse(); txt = r.read().decode("utf-8", "replace"); c.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {txt}")
    return time.monotonic() - t0, txt

def pct(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q; lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

def run(c, fxs):
    out, errs, lock = [], [], threading.Lock()
    def w(i):
        for f in fxs[i::c]:
            try:
                lat, _ = post(f)
                with lock: out.append(lat)
            except Exception as e:
                with lock: errs.append(str(e))
    t0 = time.monotonic()
    ts = [threading.Thread(target=w, args=(i,)) for i in range(c)]
    for t in ts: t.start()
    for t in ts: t.join()
    return out, time.monotonic() - t0, errs

print(f"target {HOST}:{PORT}  model={MODEL or '(default)'}")
lat, txt = post(FIXTURES[0])
print(f"first  : {lat:.2f}s  {txt.strip()!r}")
for c in (1, 2, 4):
    lats, wall, errs = run(c, FIXTURES)
    if errs:
        print(f"c={c} ERRORS {len(errs)}/{len(FIXTURES)}: {errs[0]}")
        continue
    audio = len(lats) * 25.0
    print(f"c={c:<2} wall={wall:6.2f}s  thru={audio/wall:6.2f}x rt   "
          f"p50/75/90/95={pct(lats,.5):.2f}/{pct(lats,.75):.2f}/{pct(lats,.9):.2f}/{pct(lats,.95):.2f}s   "
          f"min/mean/max={min(lats):.2f}/{statistics.mean(lats):.2f}/{max(lats):.2f}s")
