#!/usr/bin/env python3
import base64, json, os, socket, struct, sys, urllib.request

def ws_connect(url):
    rest = url[len("ws://"):]
    hostport, path = rest.split("/", 1)
    host, port = hostport.split(":")
    s = socket.create_connection((host, int(port)), timeout=30)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall(
        (f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\n"
         f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(4096)
    return s

def ws_send(s, text):
    data = text.encode()
    mask = os.urandom(4)
    head = bytearray([0x81])
    n = len(data)
    if n < 126:
        head.append(0x80 | n)
    elif n < 65536:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack(">Q", n)
    s.sendall(bytes(head) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

def recv_exact(s, n):
    out = b""
    while len(out) < n:
        chunk = s.recv(n - len(out))
        if not chunk:
            raise EOFError
        out += chunk
    return out

def ws_recv(s):
    payload = b""
    while True:
        b0, b1 = recv_exact(s, 2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", recv_exact(s, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", recv_exact(s, 8))[0]
        frame = recv_exact(s, n)
        payload += frame
        if b0 & 0x80:
            return payload.decode(errors="replace")

port = sys.argv[1]
version = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=30))
s = ws_connect(version["webSocketDebuggerUrl"])
ws_send(s, json.dumps({"id": 1, "method": "SystemInfo.getInfo"}))
while True:
    msg = json.loads(ws_recv(s))
    if msg.get("id") == 1:
        break
info = msg.get("result", {})
gpu = info.get("gpu", {})
print("devices:", json.dumps(gpu.get("devices"), indent=1))
print("feature status:", json.dumps(gpu.get("featureStatus"), indent=1))
aux = gpu.get("auxAttributes", {})
print("aux:", json.dumps({k: v for k, v in aux.items() if any(t in k.lower() for t in ("vulkan", "angle", "gl_", "software", "optimus", "amd", "nvidia", "webgpu", "hardware", "renderer", "vendor", "driver"))}, indent=1))
print("driver bug workarounds:", gpu.get("driverBugWorkarounds"))
print("command line:", info.get("commandLine", "")[:600])
