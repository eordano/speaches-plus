#!/usr/bin/env python3
import io, json, os, struct, sys, wave, zipfile
import numpy as np

STYLE_ROW = 100

def stock_rows(voices_bin):
    out = {}
    with zipfile.ZipFile(voices_bin) as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".npy"):
                continue
            arr = np.load(io.BytesIO(z.read(name)))
            if arr.ndim != 3 or arr.shape[0] == 0:
                continue
            out[name[:-4]] = arr[min(STYLE_ROW, arr.shape[0] - 1), 0, :].astype(np.float32).tolist()
    return out

def wav_f32(path):
    w = wave.open(path)
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    raw = w.readframes(n)
    assert sw == 2, f"{path}: {sw*8}-bit samples unsupported"
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 24000:
        t = np.arange(int(len(x) * 24000 / sr)) * (sr / 24000.0)
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x

def main():
    if len(sys.argv) < 4:
        print("usage: kokoro_stage.py OUT_DIR VOICES_BIN CLIP.wav [CLIP.wav ...]  (env KOKORO_MAX_REF_SECONDS, default 12)", file=sys.stderr)
        sys.exit(2)
    out, voices_bin, clips = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(os.path.join(out, "refs"), exist_ok=True)
    rows = stock_rows(voices_bin)
    with open(os.path.join(out, "stock.json"), "w") as f:
        json.dump(rows, f)
    cap = int(float(os.environ.get("KOKORO_MAX_REF_SECONDS", "12")) * 24000)
    names = []
    for i, c in enumerate(clips):
        x = wav_f32(c)[:cap]
        name = f"refs/{i:02d}.f32"
        x.astype("<f4").tofile(os.path.join(out, name))
        names.append({"file": name, "seconds": round(len(x) / 24000.0, 2), "source": c})
    with open(os.path.join(out, "refs.json"), "w") as f:
        json.dump(names, f, indent=1)
    print(f"staged {len(rows)} stock rows and {len(names)} refs into {out}")

if __name__ == "__main__":
    main()
