#!/usr/bin/env python3
import http.server
import os
import re
import sys

RANGE = re.compile(r"bytes=(\d+)-(\d*)$")

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_PUT(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        name = os.path.basename(self.path)
        if not (name.startswith("q38-result-") and name.endswith(".json")):
            self.send_response(403)
            self.end_headers()
            return
        with open(name, "wb") as f:
            f.write(body)
        self.send_response(201)
        self.end_headers()

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_GET(self):
        m = RANGE.match(self.headers.get("Range") or "")
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(left, 8 << 20))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

if __name__ == "__main__":
    port = int(sys.argv[1])
    os.chdir(sys.argv[2])
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
