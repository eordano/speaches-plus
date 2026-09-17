#!/usr/bin/env python3
import http.server
import os
import sys

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

if __name__ == "__main__":
    port = int(sys.argv[1])
    os.chdir(sys.argv[2])
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
