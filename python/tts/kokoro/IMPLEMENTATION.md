# tts/kokoro/npz.py -- Kokoro voices NPZ loader

## What it does

Faithful Python port of `speaches-plus/rust/src/tts/npz.rs`. Loads a single
`.bin` / `.npz` archive (a zip of `.npy` arrays, one per voice) and yields
`{voice_name: Voice}` where `Voice.row(i)` returns the i-th leading-dim slice.

## Python deviation: numpy.load instead of a hand-rolled .npy parser

The Rust port reimplements the `.npy` v1/v2/v3 header parser by hand because
Rust does not have numpy. Python does. We use `numpy.load(BytesIO(b))` to do
the parse instead -- numpy literally defines the .npy format, so this is the
right tool. We still validate dtype is `<f4` (float32, little-endian) to
match upstream's contract.

## Why a single-archive path exists

The existing `tts/kokoro/model.py` loads voices as one `.bin` per voice from
a directory (`KOKORO_VOICES_DIR`). The single-archive path supports the
speaches-plus deployment model where all voices ship as one zip -- a single
file is faster to download and atomic to update.

The wire-up at `tts/kokoro/__init__.py` only exports `Voice` and
`load_voices`; the existing per-file directory path in `model.py` remains
the default. A follow-up can teach `KokoroTTS.__init__` to detect a
file-shaped `voices_dir` and call `load_voices()` instead.

## Naming

Function and class names match upstream Rust verbatim: `Voice`,
`load_voices`, `parse_npy`, `Voice.row`.
