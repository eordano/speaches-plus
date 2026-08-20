# 050-diarization-multispeaker

Multi-speaker diarization fixture with recorded ground truth. It exists so
the `DIAR_HOP_RATIO` sweep has something that actually exercises speaker
discrimination — the previous diarization fixture was single-voice espeak
audio that collapses to one cluster, which makes any hop_ratio look fine.

`fixture.json` is descriptive only: this is not an endpoint fixture, so
`run_endpoint_fixture.py` classifies it as a declarative fixture and does not
drive it. The artifacts are consumed directly by the Rust sweep test.

## Input artifacts

`audio.wav` — 39.72 s, 16 kHz mono, 16-bit PCM (1.27 MB). Synthesized by
`rust/src/diarization/hop_sweep.rs::diar_build_multispeaker_fixture` from
the Kokoro-82M v1.0 ONNX voice bank, four distinct voices:

| label | Kokoro voice | character |
| --- | --- | --- |
| SPK_A | `af_heart` | US female |
| SPK_B | `am_michael` | US male |
| SPK_C | `bf_emma` | UK female |
| SPK_D | `bm_george` | UK male |

12 turns, lengths from 0.57 s to 7.46 s, with two deliberate overlaps
(350 ms and 900 ms). Turns are laid out by measuring each synthesized
utterance and offsetting it from the previous turn's end, so the ground
truth is the actual layout, not an estimate.

`ground_truth.json` — the recorded layout: per-turn speaker label, Kokoro
voice, start/end in ms, source text, plus the computed overlap intervals.

## Regenerating

```sh
K=$HOME/.cache/huggingface/hub/models--speaches-ai--Kokoro-82M-v1.0-ONNX/snapshots/*/
DIAR_FIXTURE_BUILD=1 \
DIAR_FIXTURE_KOKORO_MODEL=$K/model.onnx \
DIAR_FIXTURE_KOKORO_VOICES=$K/voices.bin \
  nix develop --command bash -c \
    'cd rust && cargo test -p speaches-plus --lib \
       diarization::hop_sweep::diar_build_multispeaker_fixture -- --ignored --nocapture'
```

`DIAR_FIXTURE_OUT` overrides the output directory.

## Consumed by

`rust/src/diarization/hop_sweep.rs::diar_hop_ratio_sweep`
(`DIAR_SWEEP=1`, `#[ignore]`). The sweep defaults to this fixture and
takes `DIAR_SWEEP_AUDIO` / `DIAR_SWEEP_TRUTH` to point at a real
recording instead, with no code change.

## Caveat

Synthetic TTS speakers separate far more cleanly than real recordings —
no channel noise, no room, no crosstalk bleed, stable per-voice timbre.
A clean sweep on this fixture is necessary but **not sufficient** to
justify moving the `DIAR_HOP_RATIO` default off 0.1.
