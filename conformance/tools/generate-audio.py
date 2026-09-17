#!/usr/bin/env python3
import os
import sys
import wave

import numpy as np

SR = 16000
OUT = os.environ.get("OUT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fixtures"
)

FIXTURES = {
    "030-voice-clone-custom-from-3sec/reference_440hz_3s_16k_mono.wav":
        (3.0, 0.5, [(0, 48000, 440.0)]),
    "040-align-en-words/audio.wav":
        (2.0, 0.4, [(1600, 8800, 440.0), (11200, 19200, 523.25), (20800, 29600, 659.25)]),
    "040-align-multi-lang-de-es-fr/audio_de.wav": (1.5, 0.4, [(2400, 22401, 392.0)]),
    "040-align-multi-lang-de-es-fr/audio_es.wav": (1.5, 0.4, [(2400, 22401, 440.0)]),
    "040-align-multi-lang-de-es-fr/audio_fr.wav": (1.5, 0.4, [(2400, 22401, 466.16)]),
    "040-align-zh-segments/audio.wav":
        (3.0, 0.4, [(3200, 19200, 392.0), (24000, 44800, 466.16)]),
}


def render(seconds, gain, bursts):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    pcm = np.zeros(n)
    for start, end, hz in bursts:
        pcm[start:end] = gain * 32767 * np.sin(2 * np.pi * hz * t[start:end])
    return pcm.astype("<i2")


def main():
    names = sys.argv[1:] or sorted(FIXTURES)
    for rel in names:
        seconds, gain, bursts = FIXTURES[rel]
        path = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(render(seconds, gain, bursts).tobytes())
        print("wrote", rel)


if __name__ == "__main__":
    main()
