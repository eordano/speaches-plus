"""Turn a fitted Kokoro adaptation into a runtime-store voice directory.

Input: the checkpoint written by the fitting run, a torch file holding
``{"style": Tensor[1,256], "decoder_style_layers": {torch_name: Tensor}}`` where the
torch names are relative to ``KModel.decoder`` (``decode.0.norm1.fc.weight`` ...).

Output: ``<store>/<name>/voice.json``, ``pack.npy`` ([1,1,256] f32) and, when decoder
layers are present, ``adapter.npz`` -- a zip of ``.npy`` entries named by the ONNX
initializer each tensor overrides (``decoder.decoder.decode.0.norm1.fc.weight``).
The ONNX names are read from the served model, so a mismatch fails here, not at
serve time.

    python -m tts.kokoro.adapter_export --model kokoro-v1.0.onnx --ckpt fit.pt \
        --name my_voice --store /var/lib/speaches/voices-store/voices \
        --speaker-id spk-0001 --eval eval.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import zipfile
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
ONNX_DECODER_PREFIX = "decoder.decoder."

def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i

def _fields(b: bytes, start: int, end: int):
    i = start
    while i < end:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield fn, v
        elif wt == 1:
            yield fn, b[i:i + 8]
            i += 8
        elif wt == 2:
            ln, i = _varint(b, i)
            yield fn, (i, i + ln)
            i += ln
        elif wt == 5:
            yield fn, b[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")

def onnx_initializer_shapes(model_path: Path) -> dict[str, list[int]]:
    """Initializer name -> dims, from the ONNX protobuf (ModelProto.graph.initializer)."""
    b = model_path.read_bytes()
    graph = next(v for fn, v in _fields(b, 0, len(b)) if fn == 7)
    out: dict[str, list[int]] = {}
    for fn, v in _fields(b, *graph):
        if fn != 5:
            continue
        dims: list[int] = []
        name = None
        for f2, v2 in _fields(b, *v):
            if f2 == 1:
                dims.append(v2)
            elif f2 == 8:
                name = b[v2[0]:v2[1]].decode()
        if name is not None:
            out[name] = dims
    return out

def map_to_onnx(layers: dict[str, np.ndarray], shapes: dict[str, list[int]]) -> dict[str, np.ndarray]:
    mapped: dict[str, np.ndarray] = {}
    for torch_name, arr in layers.items():
        onnx_name = ONNX_DECODER_PREFIX + torch_name
        if onnx_name not in shapes:
            raise KeyError(f"{torch_name}: no ONNX initializer {onnx_name}")
        if list(arr.shape) != shapes[onnx_name]:
            raise ValueError(f"{onnx_name}: shape {list(arr.shape)} != ONNX {shapes[onnx_name]}")
        mapped[onnx_name] = np.ascontiguousarray(arr, dtype="<f4")
    return mapped

def write_npz_zip(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Zip of raw .npy members (the same layout as Kokoro's voices.bin)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr)
            zf.writestr(f"{name}.npy", buf.getvalue())

def load_checkpoint(ckpt: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import torch

    d = torch.load(ckpt, map_location="cpu", weights_only=True)
    style = d["style"].float().reshape(1, 1, 256).numpy()
    layers = {k: v.float().numpy() for k, v in d.get("decoder_style_layers", {}).items()}
    return style, layers

def export(
    *,
    model: Path,
    ckpt: Path,
    name: str,
    store: Path,
    speaker_id: str | None,
    eval_path: Path | None,
    producer: dict | None,
) -> Path:
    style, layers = load_checkpoint(ckpt)
    voice_dir = store / name
    voice_dir.mkdir(parents=True, exist_ok=True)
    np.save(voice_dir / "pack.npy", style.astype("<f4"))
    kind = "kokoro-style"
    if layers:
        mapped = map_to_onnx(layers, onnx_initializer_shapes(model))
        write_npz_zip(voice_dir / "adapter.npz", mapped)
        kind = "kokoro-adapter"
    meta = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "kind": kind,
        "speaker_id": speaker_id,
        "producer": producer or {"tool": "tts.kokoro.adapter_export", "checkpoint": ckpt.name,
                                 "n_adapted_tensors": len(layers)},
        "eval": json.loads(eval_path.read_text()) if eval_path else None,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    tmp = voice_dir / "voice.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(voice_dir / "voice.json")
    return voice_dir

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="served kokoro-v1.0.onnx (initializer names)")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--store", required=True, type=Path, help="<NV_VOICES_DATA_DIR>/voices")
    ap.add_argument("--speaker-id")
    ap.add_argument("--eval", type=Path)
    a = ap.parse_args()
    out = export(model=a.model, ckpt=a.ckpt, name=a.name, store=a.store, speaker_id=a.speaker_id,
                 eval_path=a.eval, producer=None)
    print(out)

if __name__ == "__main__":
    main()
