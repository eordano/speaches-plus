"""Paper validations, share B (sequence/decoding half) -- Python side.

Each check() validates one concrete property from a source paper against OUR
implementation, with an independently written reference. Run directly:

    python3 test_papers_b.py            # prints a JSON verdict ledger

Covered here:
  - GPT-2 byte-level BPE byte<->rune table (Radford et al. 2019, encoder.py
    bytes_to_unicode) vs stt/bpe.py
  - Whisper log-mel front end (Radford et al. 2022, released audio.py;
    librosa Slaney mel scale) vs stt/mel.py + stt/constants.py
  - Powerset multi-class segmentation (Plaquet & Bredin, Interspeech 2023;
    pyannote.audio utils/powerset.py ordering) vs diarization/powerset.py
  - Kaldi-style fbank for the wespeaker speaker-embedding front end
    (Povey et al. 2011; torchaudio.compliance.kaldi) vs diarization/fbank.py
    -- includes a QUANTIFIED DEVIATION: triangle interpolation done in linear
    frequency instead of mel domain
  - Cosine speaker scoring (wespeaker; standard cosine backend) vs
    diarization/embedding.py + clustering.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS: list[dict] = []

def check(topic: str, prop: str, fn):
    try:
        detail = fn()
        RESULTS.append({"topic": topic, "property": prop, "verdict": "validated",
                        "detail": detail if isinstance(detail, str) else ""})
    except AssertionError as e:
        RESULTS.append({"topic": topic, "property": prop, "verdict": "refuted",
                        "detail": str(e)})
    except Exception as e:  # noqa: BLE001
        RESULTS.append({"topic": topic, "property": prop, "verdict": "inconclusive",
                        "detail": f"{type(e).__name__}: {e}"})

def reference_bytes_to_unicode() -> dict[int, str]:
    """Independent transcription of the released GPT-2 encoder.py."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def check_bpe_table():
    from stt import bpe
    ref = reference_bytes_to_unicode()
    assert len(bpe._BPE_BYTE_TO_RUNE) == 256, "table must cover all 256 bytes"
    for b in range(256):
        assert bpe._BPE_BYTE_TO_RUNE[b] == ref[b], (
            f"byte {b}: ours {bpe._BPE_BYTE_TO_RUNE[b]!r} != gpt2 {ref[b]!r}")
    assert len(set(bpe._BPE_BYTE_TO_RUNE.values())) == 256
    assert {bpe._BPE_RUNE_TO_BYTE[r] for r in bpe._BPE_BYTE_TO_RUNE.values()} == set(range(256))
    return "byte<->rune table identical to GPT-2 bytes_to_unicode, bijective on 256 bytes"

def check_bpe_roundtrip():
    from stt import bpe
    cases = ["hello world", "café naïve", "你好世界",
             "emoji \U0001f600\U0001f680", "\r\n\t \x00\x7f",
             "".join(chr(c) for c in range(32, 127))]
    for s in cases:
        assert bpe.decode_bpe(bpe.encode_bpe(s)) == s, f"round-trip failed on {s!r}"
    for b in range(256):
        rune = bpe._BPE_BYTE_TO_RUNE[b]
        assert bpe._BPE_RUNE_TO_BYTE[rune] == b
    return "encode∘decode identity on ASCII/CJK/emoji/controls + all 256 bytes"

def check_whisper_constants():
    from stt import constants as c
    assert c.WHISPER_NFFT == 400, "whisper uses n_fft=400 (25 ms @ 16 kHz)"
    assert c.WHISPER_HOP_LENGTH == 160, "hop=160 (10 ms)"
    assert c.WHISPER_SAMPLING_HZ == 16_000
    assert c.WHISPER_PAD_SAMPLES == 480_000, "30 s context"
    assert c.WHISPER_NB_FRAMES == 3_000, "3000 frames after dropping the stft tail frame"
    assert c.DEFAULT_N_MELS == 80 and c.LARGE_V3_N_MELS == 128
    assert c.WHISPER_TIMESTAMP_STEP_MS == 20, "timestamp tokens step 0.02 s"
    assert c.WHISPER_TIMESTAMP_TOKEN_COUNT == 1501, "0.00..30.00 inclusive = 1501 tokens"
    assert (30_000 // c.WHISPER_TIMESTAMP_STEP_MS) + 1 == c.WHISPER_TIMESTAMP_TOKEN_COUNT
    return "n_fft/hop/sr/pad/frames/mels/timestamp constants all match openai-whisper"

def librosa_slaney_mel_filters(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Independent vectorized transcription of librosa.filters.mel
    (htk=False, norm='slaney') -- the exact function whose output ships as
    whisper's assets/mel_filters.npz."""
    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)
        f_sp = 200.0 / 3
        mels = f / f_sp
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        log_t = f >= min_log_hz
        mels = np.where(log_t, min_log_mel + np.log(np.maximum(f, 1e-30) / min_log_hz) / logstep, mels)
        return mels

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        f_sp = 200.0 / 3
        freqs = f_sp * m
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        log_t = m >= min_log_mel
        return np.where(log_t, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)

    fmax = sr / 2.0
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(fmax), n_mels + 2))
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    lower = -ramps[:-2] / fdiff[:-1][:, None]
    upper = ramps[2:] / fdiff[1:][:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights

def check_whisper_mel_filters():
    from stt import mel
    for n_mels in (80, 128):
        ours = mel.build_mel_filters(n_mels)
        ref = librosa_slaney_mel_filters(n_mels, 400, 16_000)
        diff = float(np.abs(ours - ref).max())
        assert diff < 2e-6, f"n_mels={n_mels}: max filter weight diff {diff}"
    assert abs(mel.hz_to_mel(1000.0) - 15.0) < 1e-12, "1 kHz must map to mel 15 (200/3 Hz/mel)"
    assert abs(mel.mel_to_hz(15.0 + 27.0) - 6400.0) < 1e-6, "27 mel above breakpoint = x6.4 in Hz"
    for f in (0.0, 100.0, 999.9, 1000.0, 3456.7, 8000.0):
        assert abs(mel.mel_to_hz(mel.hz_to_mel(f)) - f) < 1e-6
    return "80/128-bin filterbank == librosa slaney (htk=False, norm=slaney) to <2e-6"

def check_whisper_log_mel_pipeline():
    from stt import mel
    rng = np.random.default_rng(1234)
    audio = (rng.standard_normal(480_000) * 0.1).astype(np.float32)
    ours = mel.MelExtractor(n_mels=80).log_mel(audio)

    n_fft, hop = 400, 160
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    padded = np.pad(audio.astype(np.float64), n_fft // 2, mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    frames = np.stack([padded[i * hop: i * hop + n_fft] * window for i in range(n_frames)])
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2)[:-1]
    ref_filters = librosa_slaney_mel_filters(80, n_fft, 16_000)
    melspec = ref_filters @ power.T
    log_spec = np.log10(np.maximum(melspec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    ref = ((log_spec + 4.0) / 4.0).astype(np.float32)

    assert ours.shape == ref.shape == (80, 3000), f"shape {ours.shape} vs {ref.shape}"
    diff = float(np.abs(ours - ref).max())
    assert diff < 5e-4, f"log-mel differs from whisper reference by {diff}"
    return f"full log-mel pipeline matches independent whisper audio.py transcription (max diff {diff:.2e})"

def reference_powerset_mapping(num_speakers: int, max_set_size: int) -> list[list[int]]:
    """pyannote.audio utils/powerset.py build_mapping ordering."""
    import itertools
    out = []
    for size in range(0, max_set_size + 1):
        for combo in itertools.combinations(range(num_speakers), size):
            out.append(list(combo))
    return out

def check_powerset_mapping():
    from diarization.powerset import PowersetDecoder
    for (s, k) in [(3, 2), (4, 2), (2, 2), (5, 3)]:
        dec = PowersetDecoder(s, k)
        ref = reference_powerset_mapping(s, k)
        assert dec.mapping == ref, f"({s},{k}): mapping order differs from pyannote"
        want_classes = sum(math.comb(s, i) for i in range(k + 1))
        assert dec.num_classes() == want_classes, (
            f"({s},{k}): {dec.num_classes()} classes, paper says {want_classes}")
    assert PowersetDecoder(3, 2).num_classes() == 7
    return "class count = sum_k C(S,k) and pyannote combination order, incl. the 7-class 3.0 topology"

def check_powerset_hard_decode():
    from diarization.powerset import PowersetDecoder
    from diarization.segmentation import SegmentationLogits
    dec = PowersetDecoder(3, 2)
    rng = np.random.default_rng(7)
    frames, classes = 50, dec.num_classes()
    data = rng.standard_normal((frames, classes)).astype(np.float32)
    logits = SegmentationLogits(frames=frames, classes=classes, data=data.reshape(-1))
    ml = dec.to_multilabel_hard(logits)
    ref_map = reference_powerset_mapping(3, 2)
    for f in range(frames):
        active = set(ref_map[int(np.argmax(data[f]))])
        got = {s for s in range(3) if ml.row(f)[s] == 1}
        assert got == active, f"frame {f}: {got} vs {active}"
    return "argmax powerset class -> multilabel matches the paper's hard decode on 50 random frames"

def check_povey_window_and_framing():
    from diarization import fbank as fb
    n = 400
    w = fb._povey_window(n)
    ref = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))) ** 0.85
    assert float(np.abs(w - ref).max()) < 1e-6, "povey window formula mismatch"
    assert fb.PRE_EMPHASIS == 0.97, "kaldi preemphasis default 0.97"
    assert fb._next_power_of_two(400) == 512, "kaldi round_to_power_of_two"
    f = fb.FBank(80, 400, 160)
    out = f.compute(np.zeros(16000, dtype=np.float32))
    assert out.size == 80 * (1 + (16000 - 400) // 160)
    assert abs(fb._hz_to_mel(700.0) - 1127.0 * math.log(2.0)) < 1e-9
    return "povey window, preemphasis 0.97, 512-fft, snip_edges framing, HTK mel all match kaldi"

def kaldi_mel_banks(num_bins: int, n_fft: int, sr: float, low: float, high: float) -> np.ndarray:
    """Independent transcription of torchaudio.compliance.kaldi.get_mel_banks:
    triangle interpolation in MEL domain (this is what Kaldi/wespeaker use)."""
    def mel(hz):
        return 1127.0 * np.log(1.0 + np.asanyarray(hz, dtype=np.float64) / 700.0)
    fft_bin_width = sr / n_fft
    mel_low, mel_high = mel(low), mel(high)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)
    bins = np.zeros((num_bins, n_fft // 2 + 1))
    mel_of_bin = mel(fft_bin_width * np.arange(n_fft // 2 + 1))
    for j in range(num_bins):
        left = mel_low + j * mel_delta
        center = left + mel_delta
        right = center + mel_delta
        up = (mel_of_bin - left) / (center - left)
        down = (right - mel_of_bin) / (right - center)
        bins[j] = np.maximum(0.0, np.minimum(up, down))
    return bins

def check_fbank_mel_interpolation_domain():
    """DELIBERATE REFUTATION ATTEMPT: Kaldi (and therefore wespeaker's
    torchaudio front end) interpolates the triangular filters linearly in MEL
    space; diarization/fbank.py _build_mel_filters interpolates linearly in
    Hz/bin space. Same band edges, different in-band weights."""
    from diarization import fbank as fb
    ours_sparse = fb._build_mel_filters(80, 512, 16000.0, 20.0, 7600.0)
    ours = np.zeros((80, 257))
    for m, taps in enumerate(ours_sparse):
        for k, w in taps:
            if k < 257:
                ours[m, k] = w
    ref = kaldi_mel_banks(80, 512, 16000.0, 20.0, 7600.0)
    max_diff = float(np.abs(ours - ref).max())
    rng = np.random.default_rng(3)
    power = (np.abs(np.fft.rfft(rng.standard_normal(512))) ** 2)
    e_ours = np.log(np.maximum(ours @ power, 1e-10))
    e_ref = np.log(np.maximum(ref @ power, 1e-10))
    log_diff = float(np.abs(e_ours - e_ref).max())
    assert max_diff < 1e-6, (
        f"filter weights are linear-in-Hz, Kaldi is linear-in-mel: max weight diff "
        f"{max_diff:.4f}, max log-energy diff on white noise {log_diff:.4f} "
        f"(fbank.py _build_mel_filters L85-L102 vs kaldi get_mel_banks)")
    return "unexpected: matches kaldi mel-domain interpolation"

def check_fbank_dc_offset_handling():
    """Kaldi default remove_dc_offset=True subtracts the frame mean before
    windowing; fbank.py never does. A pure DC bias should therefore leak into
    the features if the implementation deviates."""
    from diarization import fbank as fb
    f = fb.FBank(80, 400, 160)
    base = np.random.default_rng(11).standard_normal(4000).astype(np.float32) * 0.1
    a = f.compute(base)
    b = f.compute(base + 0.5)
    diff = float(np.abs(a - b).max())
    assert diff < 1e-3, (
        f"DC offset changes features by up to {diff:.3f} -- Kaldi's default "
        f"remove_dc_offset=True would make them identical (fbank.py has no "
        f"remove_dc_offset step)")
    return "unexpected: DC offset rejected"

def torchaudio_kaldi_fbank_reference(audio: np.ndarray, num_mels: int = 80,
                                     window: str = "povey") -> np.ndarray:
    """Independent numpy transcription of torchaudio.compliance.kaldi.fbank
    with the wespeaker runtime settings: frame 400/160, dither=0,
    remove_dc_offset=True, preemphasis 0.97, snip_edges=True, 512-point FFT,
    mel-domain triangles, low=20, high=0 (Nyquist). The deployed pyannote
    wrapper (config.yaml of pyannote/wespeaker-voxceleb-resnet34-LM) uses
    window_type=hamming; wespeaker's own CLI uses the kaldi default povey."""
    frame_length, frame_shift, n_fft = 400, 160, 512
    n = 1 + (len(audio) - frame_length) // frame_shift
    idx = np.arange(frame_length)
    if window == "povey":
        win = (0.5 - 0.5 * np.cos(2.0 * np.pi * idx / (frame_length - 1))) ** 0.85
    else:
        win = 0.54 - 0.46 * np.cos(2.0 * np.pi * idx / (frame_length - 1))
    feats = np.zeros((n, num_mels))
    banks = kaldi_mel_banks(num_mels, n_fft, 16000.0, 20.0, 8000.0)
    for i in range(n):
        frame = audio[i * frame_shift: i * frame_shift + frame_length].astype(np.float64)
        frame = frame - frame.mean()
        prev = np.concatenate([[frame[0]], frame[:-1]])
        frame = frame - 0.97 * prev
        frame = frame * win
        spec = np.fft.rfft(frame, n=n_fft)
        power = spec.real ** 2 + spec.imag ** 2
        feats[i] = np.log(np.maximum(banks @ power, 1.1921e-07))
    feats -= feats.mean(axis=0)
    return feats

def check_fbank_aggregate_vs_wespeaker_runtime():
    """DELIBERATE REFUTATION ATTEMPT: aggregate feature difference between
    diarization/fbank.py and a faithful torchaudio-kaldi reference at the
    wespeaker runtime settings (povey window). Any large gap means the ONNX
    embedding model is fed features unlike its training front end."""
    from diarization import fbank as fb
    rng = np.random.default_rng(42)
    t = np.arange(32000) / 16000.0
    audio = (0.3 * np.sin(2 * np.pi * 120 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
             + 0.1 * np.sin(2 * np.pi * 850 * t) + 0.02 * rng.standard_normal(len(t))
             + 0.01).astype(np.float32)
    ours = fb.FBank(80, 400, 160).compute(audio).reshape(-1, 80).astype(np.float64)
    ref = torchaudio_kaldi_fbank_reference(audio, 80, window="povey")
    assert ours.shape == ref.shape
    diff = np.abs(ours - ref)
    assert float(diff.max()) < 0.1, (
        f"post-CMN features deviate from the wespeaker/kaldi front end: "
        f"max {float(diff.max()):.3f}, mean {float(diff.mean()):.3f} "
        f"(causes: Hz-domain triangles, no remove_dc_offset, high_freq 7600 vs "
        f"Nyquist; pyannote's wrapper additionally uses hamming, not povey)")
    return "unexpected: matches wespeaker runtime front end"

def check_fbank_cmn():
    from diarization import fbank as fb
    f = fb.FBank(40, 400, 160)
    out = f.compute(np.random.default_rng(5).standard_normal(8000).astype(np.float32))
    feats = out.reshape(-1, 40)
    col_means = feats.mean(axis=0)
    assert float(np.abs(col_means).max()) < 1e-4, "wespeaker applies per-utterance CMN"
    return "cepstral/log-mel mean normalization (per utterance, per dim) applied as in wespeaker"

def check_cosine_backend():
    from diarization.embedding import _l2_normalize, cosine_sim
    rng = np.random.default_rng(21)
    for _ in range(50):
        a = rng.standard_normal(256).astype(np.float32)
        b = rng.standard_normal(256).astype(np.float32)
        na, nb = _l2_normalize(a), _l2_normalize(b)
        assert abs(float(np.linalg.norm(na)) - 1.0) < 1e-5
        want = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        got = cosine_sim(na, nb)
        assert abs(got - want) < 1e-5, f"cosine {got} vs {want}"
        assert abs(cosine_sim(na, na) - 1.0) < 1e-5
    return "score(a,b) = <a,b>/(|a||b|) via unit-normalized dot; self-similarity 1"

def check_online_clustering_invariants():
    from diarization.clustering import OnlineClusterer
    from diarization.embedding import _l2_normalize
    rng = np.random.default_rng(31)
    cl = OnlineClusterer(threshold=0.6, max_speakers=4)
    v = _l2_normalize(rng.standard_normal(64).astype(np.float32))
    cid, _ = cl.assign(v)
    cid2, sim2 = cl.assign(v)
    assert cid2 == cid and sim2 > 0.999, "identical embedding must re-match its cluster"
    for _ in range(20):
        cl.assign(_l2_normalize(v + 0.05 * rng.standard_normal(64).astype(np.float32)))
    for c in cl._centroids:
        assert abs(float(np.linalg.norm(c.vec)) - 1.0) < 1e-5, "centroid must stay unit-norm"
    w = np.zeros(64, dtype=np.float32)
    w[np.argmin(np.abs(v))] = 1.0
    w = _l2_normalize(w - float(np.dot(w, v)) * v)
    cidw, _ = cl.assign(w)
    assert cidw != cid, "below-threshold similarity must create a new speaker"
    return "threshold gating, EMA centroid unit-norm, new-speaker creation all hold"

def main():
    check("bpe", "GPT-2 bytes_to_unicode table identity + bijection", check_bpe_table)
    check("bpe", "byte-level BPE round-trip losslessness", check_bpe_roundtrip)
    check("whisper-mel", "front-end constants (n_fft/hop/frames/mels/timestamps)",
          check_whisper_constants)
    check("whisper-mel", "Slaney mel filterbank == librosa/whisper assets", check_whisper_mel_filters)
    check("whisper-mel", "full log-mel pipeline vs whisper audio.py", check_whisper_log_mel_pipeline)
    check("powerset", "class count and pyannote mapping order", check_powerset_mapping)
    check("powerset", "argmax hard decode to multilabel", check_powerset_hard_decode)
    check("fbank", "povey window / preemphasis / framing / HTK mel", check_povey_window_and_framing)
    check("fbank", "triangle interpolation domain (mel vs Hz)", check_fbank_mel_interpolation_domain)
    check("fbank", "remove_dc_offset (kaldi default)", check_fbank_dc_offset_handling)
    check("fbank", "aggregate features vs wespeaker/kaldi runtime front end",
          check_fbank_aggregate_vs_wespeaker_runtime)
    check("fbank", "per-utterance CMN", check_fbank_cmn)
    check("speaker-cosine", "cosine backend on unit embeddings", check_cosine_backend)
    check("speaker-cosine", "online clustering invariants", check_online_clustering_invariants)
    print(json.dumps(RESULTS, indent=2))
    bad = [r for r in RESULTS if r["verdict"] != "validated"]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} validated, "
          f"{len(bad)} refuted/inconclusive", file=sys.stderr)

if __name__ == "__main__":
    main()
