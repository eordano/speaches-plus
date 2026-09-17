"""ONNX model -> graph.json + weights.safetensors, for the candle mini-executor in nv-tts
(`kokoro::onnx_exec`), used to run the WeSpeaker ResNet293 embedding model natively.

Pure stdlib: the protobuf is walked by hand (ModelProto.graph = field 7; GraphProto.node = 1,
initializer = 5, input = 11, output = 12; NodeProto input=1, output=2, name=3, op_type=4,
attribute=5; AttributeProto name=1, f=2, i=3, s=4, t=5, floats=7, ints=8, type=20;
TensorProto dims=1, data_type=2, float_data=4, int64_data=7, name=8, raw_data=9).
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return r, i

def _fields(b, start, end):
    i = start
    while i < end:
        key, i = _varint(b, i); fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i); yield fn, wt, v
        elif wt == 1:
            yield fn, wt, b[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = _varint(b, i); yield fn, wt, (i, i + ln); i += ln
        elif wt == 5:
            yield fn, wt, b[i:i + 4]; i += 4
        else:
            raise ValueError(f"wire type {wt}")

DTYPES = {1: ("F32", "<f4", 4), 7: ("I64", "<i8", 8), 6: ("I32", "<i4", 4), 9: ("BOOL", "?", 1), 10: ("F16", "<f2", 2), 11: ("F64", "<f8", 8)}

def _tensor(b, span):
    dims, name, dt, raw, floats, ints = [], None, None, None, [], []
    for f, w, v in _fields(b, *span):
        if f == 1: dims.append(v)
        elif f == 2: dt = v
        elif f == 8: name = b[v[0]:v[1]].decode()
        elif f == 9: raw = b[v[0]:v[1]]
        elif f == 4:
            if w == 2:
                floats.extend(struct.unpack(f"<{(v[1]-v[0])//4}f", b[v[0]:v[1]]))
            else:
                floats.append(struct.unpack("<f", v)[0])
        elif f == 7:
            if w == 2:
                j = v[0]
                while j < v[1]:
                    x, j = _varint(b, j); ints.append(x if x < 2**63 else x - 2**64)
            else:
                ints.append(v if v < 2**63 else v - 2**64)
    kind, fmt, size = DTYPES[dt]
    if raw is None:
        n = 1
        for d in dims: n *= d
        raw = struct.pack(f"{fmt[0]}{n}{fmt[1:]}", *(floats if kind.startswith("F") else ints)) if n else b""
    return name, dims, kind, raw

def _attr(b, span):
    name, out = None, None
    for f, w, v in _fields(b, *span):
        if f == 1: name = b[v[0]:v[1]].decode()
        elif f == 2: out = ("f", struct.unpack("<f", v)[0])
        elif f == 3: out = ("i", v if v < 2**63 else v - 2**64)
        elif f == 4: out = ("s", b[v[0]:v[1]].decode(errors="replace"))
        elif f == 5: out = ("t", _tensor(b, v))
        elif f == 7:
            out = ("floats", list(struct.unpack(f"<{(v[1]-v[0])//4}f", b[v[0]:v[1]])) if w == 2 else [struct.unpack("<f", v)[0]])
        elif f == 8:
            if w == 2:
                vals, j = [], v[0]
                while j < v[1]:
                    x, j = _varint(b, j); vals.append(x if x < 2**63 else x - 2**64)
                out = ("ints", (out[1] if out and out[0] == "ints" else []) + vals)
            else:
                out = ("ints", (out[1] if out and out[0] == "ints" else []) + [v if v < 2**63 else v - 2**64])
    return name, out

def _value_info(b, span):
    name, shape = None, []
    for f, w, v in _fields(b, *span):
        if f == 1: name = b[v[0]:v[1]].decode()
        elif f == 2:  # TypeProto
            for f2, w2, v2 in _fields(b, *v):
                if f2 == 1:  # tensor_type
                    for f3, w3, v3 in _fields(b, *v2):
                        if f3 == 2:  # shape
                            for f4, w4, v4 in _fields(b, *v3):
                                if f4 == 1:
                                    d = None
                                    for f5, w5, v5 in _fields(b, *v4):
                                        if f5 == 1: d = v5
                                        elif f5 == 2: d = b[v5[0]:v5[1]].decode()
                                    shape.append(d)
    return name, shape

def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, offset = {}, 0
    blobs = []
    for name, (kind, dims, raw) in tensors.items():
        header[name] = {"dtype": kind, "shape": dims, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw); blobs.append(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for blob in blobs:
            f.write(blob)

def export(onnx_path: Path, out_dir: Path) -> dict:
    b = onnx_path.read_bytes()
    graph = next(v for f, w, v in _fields(b, 0, len(b)) if f == 7)
    tensors, nodes, inputs, outputs = {}, [], [], []
    for f, w, v in _fields(b, *graph):
        if f == 5:
            name, dims, kind, raw = _tensor(b, v); tensors[name] = (kind, dims, raw)
        elif f == 1:
            node = {"op": None, "name": None, "inputs": [], "outputs": [], "attrs": {}}
            for f2, w2, v2 in _fields(b, *v):
                if f2 == 1: node["inputs"].append(b[v2[0]:v2[1]].decode())
                elif f2 == 2: node["outputs"].append(b[v2[0]:v2[1]].decode())
                elif f2 == 3: node["name"] = b[v2[0]:v2[1]].decode()
                elif f2 == 4: node["op"] = b[v2[0]:v2[1]].decode()
                elif f2 == 5:
                    an, av = _attr(b, v2)
                    if av is not None and av[0] == "t":
                        tname, tdims, tkind, traw = av[1]
                        key = f"{node['name'] or 'const'}.{an}"
                        tensors[key] = (tkind, tdims, traw); node["attrs"][an] = ["tensor", key]
                    elif av is not None:
                        node["attrs"][an] = [av[0], av[1]]
            nodes.append(node)
        elif f == 11:
            inputs.append(_value_info(b, v))
        elif f == 12:
            outputs.append(_value_info(b, v))
    init_names = set(tensors)
    inputs = [i for i in inputs if i[0] not in init_names]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_safetensors(out_dir / "weights.safetensors", tensors)
    meta = {"source": str(onnx_path), "inputs": [{"name": n, "shape": s} for n, s in inputs],
            "outputs": [{"name": n, "shape": s} for n, s in outputs], "nodes": nodes}
    (out_dir / "graph.json").write_text(json.dumps(meta, indent=0))
    return {"nodes": len(nodes), "tensors": len(tensors), "inputs": inputs, "outputs": outputs}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx", type=Path); ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    print(export(a.onnx, a.out_dir))

if __name__ == "__main__":
    main()
