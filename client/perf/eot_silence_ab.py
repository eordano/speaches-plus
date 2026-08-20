"""A/B a silence-end threshold change at the smart-turn layer.

Feeds labeled turns (pipecat smart-turn-data parquet shard) to smart-turn,
truncated at VAD speech-end plus each candidate silence duration, and reports
complete-detected / false-cutoff rates per candidate.  The feature pipeline
mirrors rust/src/eou/audio.rs (whisper log-mel, slaney filters, leading pad)
and the VAD framing mirrors rust/src/vad (512-sample window + 64-sample
context); validate any change to either by checking the whole-clip eval
reproduces the published smart-turn accuracy for the model in use.

Env:
  EOT_AB_SHARD        parquet shard (audio/endpoint_bool/language columns)
  EOT_AB_SMART_TURN   smart-turn onnx (mel [1,80,800] input)
  EOT_AB_SILERO       silero vad onnx
  EOT_AB_SILENCES_MS  comma list, default "350,200"
  EOT_AB_LANG         dataset language filter, default "eng"
"""

import io
import os
import sys

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf

SR = 16_000
N_FFT, HOP, N_MELS, TARGET = 400, 160, 80, 8 * SR
N_FRAMES = TARGET // HOP
VAD_WINDOW, VAD_CONTEXT = 512, 64

MEL_F_SP = 200.0 / 3.0
MEL_MIN_LOG_HZ = 1000.0
MIN_LOG_MEL = MEL_MIN_LOG_HZ / MEL_F_SP
LOGSTEP = np.log(6.4) / 27.0

def hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    safe = np.maximum(f, 1e-12)
    return np.where(
        f >= MEL_MIN_LOG_HZ, MIN_LOG_MEL + np.log(safe / MEL_MIN_LOG_HZ) / LOGSTEP, f / MEL_F_SP
    )

def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    return np.where(
        m >= MIN_LOG_MEL, MEL_MIN_LOG_HZ * np.exp((m - MIN_LOG_MEL) * LOGSTEP), MEL_F_SP * m
    )

def mel_filters():
    n_bins = N_FFT // 2 + 1
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2), N_MELS + 2)
    hz = mel_to_hz(mels)
    freqs = np.arange(n_bins) * SR / N_FFT
    fb = np.zeros((N_MELS, n_bins))
    for m in range(N_MELS):
        lo, ce, up = hz[m], hz[m + 1], hz[m + 2]
        enorm = 2.0 / max(up - lo, 1e-12)
        rise = (freqs - lo) / max(ce - lo, 1e-12)
        fall = (up - freqs) / max(up - ce, 1e-12)
        fb[m] = np.clip(np.minimum(rise, fall), 0, None) * enorm
    return fb.astype(np.float32)

HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
FB = mel_filters()

def log_mel(audio):
    pad = N_FFT // 2
    padded = np.concatenate([audio[pad:0:-1], audio, audio[-2 : -2 - pad : -1]])
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP][:N_FRAMES]
    spec = np.fft.rfft(frames * HANN, axis=1)
    power = np.abs(spec).astype(np.float64) ** 2
    mel = np.maximum(power @ FB.T, 1e-10)
    logm = np.log10(mel)
    logm = np.maximum(logm, logm.max() - 8.0)
    return ((logm + 4.0) / 4.0).astype(np.float32).T

def prepare(audio):
    a = audio[-TARGET:]
    if len(a) < TARGET:
        a = np.concatenate([np.zeros(TARGET - len(a), dtype=np.float32), a])
    return np.clip(a, -1.0, 1.0)

def decode(b):
    data, rate = sf.read(io.BytesIO(b), dtype="float32")
    assert rate == SR, f"rate {rate}"
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)

def main():
    shard = os.environ["EOT_AB_SHARD"]
    st = ort.InferenceSession(os.environ["EOT_AB_SMART_TURN"], providers=["CPUExecutionProvider"])
    st_in = st.get_inputs()[0].name
    vad = ort.InferenceSession(os.environ["EOT_AB_SILERO"], providers=["CPUExecutionProvider"])
    silences = [int(s) for s in os.environ.get("EOT_AB_SILENCES_MS", "350,200").split(",")]
    lang_filter = os.environ.get("EOT_AB_LANG", "eng")

    def speech_end(a):
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(SR, dtype=np.int64)
        ctx = np.zeros(VAD_CONTEXT, dtype=np.float32)
        last = 0
        for i in range(len(a) // VAD_WINDOW):
            frame = a[i * VAD_WINDOW : (i + 1) * VAD_WINDOW].astype(np.float32)
            out = vad.run(None, {"input": np.concatenate([ctx, frame])[None, :], "state": state, "sr": sr})
            state = out[1]
            ctx = frame[-VAD_CONTEXT:]
            if out[0].item() > 0.5:
                last = (i + 1) * VAD_WINDOW
        return last

    tbl = pq.read_table(shard, columns=["audio", "endpoint_bool", "language"])
    stats = {ms: dict(tp=0, comp=0, fc=0, inc=0) for ms in silences}
    whole = dict(tp=0, comp=0, fc=0, inc=0)
    used = 0
    for arec, label, lang in zip(
        tbl.column("audio").to_pylist(),
        tbl.column("endpoint_bool").to_pylist(),
        tbl.column("language").to_pylist(),
    ):
        if lang != lang_filter:
            continue
        try:
            a = decode(arec["bytes"])
        except Exception:
            continue
        se = speech_end(a)
        if se == 0:
            continue
        used += 1

        def score(clip):
            return float(st.run(None, {st_in: log_mel(prepare(clip))[None]})[0].reshape(-1)[0])

        def tally(s, p):
            if label:
                s["comp"] += 1
                s["tp"] += p > 0.5
            else:
                s["inc"] += 1
                s["fc"] += p > 0.5

        tally(whole, score(a))
        for ms in silences:
            d = ms * SR // 1000
            cut = min(se + d, len(a))
            clip = a[:cut]
            if cut < se + d:
                clip = np.concatenate([clip, np.zeros(se + d - cut, dtype=np.float32)])
            tally(stats[ms], score(clip))

    def row(name, s):
        tp = s["tp"] / max(s["comp"], 1) * 100
        fc = s["fc"] / max(s["inc"], 1) * 100
        print(
            f"{name:>14}: complete-detected {tp:5.1f}% ({s['tp']}/{s['comp']})  "
            f"false-cutoff {fc:5.1f}% ({s['fc']}/{s['inc']})"
        )

    print(f"samples used={used} lang={lang_filter}")
    row("whole-clip", whole)
    for ms in silences:
        row(f"{ms}ms", stats[ms])
    sane = whole["tp"] / max(whole["comp"], 1)
    if sane < 0.9:
        print(
            f"WARNING: whole-clip complete-detected {sane:.0%} is far below published "
            "smart-turn accuracy -- feature pipeline likely diverged from rust/src/eou/audio.rs",
            file=sys.stderr,
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
