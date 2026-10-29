#!/usr/bin/env python3
"""HF transformers oracle for the nv-models Voxtral Realtime port.

  nix shell nixpkgs#python3Packages.{torch,transformers,mistral-common,soundfile,numpy} \
    -c python3 rust/scripts/voxtral-reference.py --model-dir <snapshot> --wav <16k mono wav> \
    [--dump out.safetensors]

Prints the greedy token stream and text; --dump also stores the projector output
(`audio_embeds`), the log-mel features and the token ids for tensor-level parity.
"""
import argparse
import json

import numpy as np
import soundfile as sf
import torch

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--dump")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000, sr
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(args.model_dir, dtype=dtype).to(device)
    model.eval()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device, dtype=dtype)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False)
        prefix = inputs["input_ids"].shape[1]
        tokens = out[0, prefix:].tolist()
        text = processor.batch_decode(out[:, prefix:], skip_special_tokens=True)[0]
        print(json.dumps({"text": text, "tokens": tokens, "prefix_len": prefix}))
        if args.dump:
            from safetensors.torch import save_file

            feats = model.model.get_audio_features(input_features=inputs["input_features"])
            save_file(
                {
                    "input_features": inputs["input_features"][0].float().cpu().contiguous(),
                    "audio_embeds": feats.pooler_output[0].float().cpu().contiguous(),
                    "tokens": torch.tensor(tokens, dtype=torch.int32),
                    "input_ids": inputs["input_ids"][0].to(torch.int32).cpu(),
                },
                args.dump,
            )

if __name__ == "__main__":
    main()
