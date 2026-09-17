"""kokoro-v1_0.pth -> kokoro-v1_0.safetensors for the candle port (docs/book/07.3-kokoro-candle.md).

Torch parameter names are kept; weight_norm is folded (weight = g * v / ||v||) so candle loads
plain conv weights. With --ref, also dumps every intermediate of one sentence (phoneme string
+ 256-d style pack row) as the parity reference for `cargo test -p nv-tts kokoro`.

    python -m tts.kokoro.convert_checkpoint --kokoro-src <dir with kokoro/model.py> \
        --config config.json --pth kokoro-v1_0.pth --out kokoro-v1_0.safetensors \
        --ref ref.safetensors --phonemes "ðə bˈɪɡɪst ..." --pack voice.npy
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

def _import_kokoro(src: Path):
    class _NoLog:
        def __getattr__(self, name):
            return lambda *a, **k: None
    lg = types.ModuleType("loguru"); lg.logger = _NoLog(); sys.modules["loguru"] = lg
    pkg = types.ModuleType("kokoro"); pkg.__path__ = [str(src / "kokoro")]; sys.modules["kokoro"] = pkg  # skip __init__ (misaki)
    sys.path.insert(0, str(src))
    from kokoro.model import KModel  # noqa: E402
    return KModel

def convert(km, out: Path) -> int:
    import torch
    from safetensors.torch import save_file
    from torch.nn.utils import remove_weight_norm
    folded = 0
    for m in km.modules():
        if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            remove_weight_norm(m); folded += 1
    sd = {k: v.contiguous().float() for k, v in km.state_dict().items()
          if (v.dtype.is_floating_point or v.dtype == torch.int64) and not k.startswith("bert.embeddings.position_ids")}
    save_file(sd, str(out), metadata={"source": "hexgrad/Kokoro-82M kokoro-v1_0.pth", "weight_norm": "folded"})
    return folded

def dump_reference(km, phonemes: str, pack_row, out: Path) -> dict:
    import torch
    from safetensors.torch import save_file
    ids = [km.vocab[c] for c in phonemes if c in km.vocab]
    input_ids = torch.LongTensor([[0, *ids, 0]])
    ref_s = torch.as_tensor(pack_row).reshape(1, 256).float()
    d = {"input_ids": input_ids, "ref_s": ref_s}
    with torch.no_grad():
        n = input_ids.shape[-1]
        lengths = torch.full((1,), n, dtype=torch.long)
        mask = torch.gt(torch.arange(n).unsqueeze(0) + 1, lengths.unsqueeze(1))
        d["bert_dur"] = km.bert(input_ids, attention_mask=(~mask).int())
        d["d_en"] = km.bert_encoder(d["bert_dur"]).transpose(-1, -2)
        s = ref_s[:, 128:]
        d["d"] = km.predictor.text_encoder(d["d_en"], s, lengths, mask)
        x, _ = km.predictor.lstm(d["d"]); d["duration_logits"] = km.predictor.duration_proj(x)
        dur = torch.round(torch.sigmoid(d["duration_logits"]).sum(-1)).clamp(min=1).long().squeeze(); d["pred_dur"] = dur
        idx = torch.repeat_interleave(torch.arange(n), dur)
        aln = torch.zeros((n, idx.shape[0])); aln[idx, torch.arange(idx.shape[0])] = 1; aln = aln.unsqueeze(0)
        d["en"] = d["d"].transpose(-1, -2) @ aln
        d["F0_pred"], d["N_pred"] = km.predictor.F0Ntrain(d["en"], s)
        d["t_en"] = km.text_encoder(input_ids, lengths, mask); d["asr"] = d["t_en"] @ aln
        dec = km.decoder
        F0 = dec.F0_conv(d["F0_pred"].unsqueeze(1)); N = dec.N_conv(d["N_pred"].unsqueeze(1))
        xx = dec.encode(torch.cat([d["asr"], F0, N], 1), ref_s[:, :128]); d["dec_encode"] = xx
        asr_res = dec.asr_res(d["asr"]); res = True
        for block in dec.decode:
            if res:
                xx = torch.cat([xx, asr_res, F0, N], 1)
            xx = block(xx, ref_s[:, :128])
            if block.upsample_type != "none":
                res = False
        d["dec_pre_gen"] = xx
        d["audio"] = dec.generator(xx, ref_s[:, :128], d["F0_pred"]).reshape(-1)
        # noise-free variant: zero the random initial phase and every noise draw so the candle
        # generator can be gated exactly (production keeps the noise)
        rand, randn_like = torch.rand, torch.randn_like
        torch.rand = lambda *a, **k: torch.zeros(*a, **{x: y for x, y in k.items() if x != "generator"})
        torch.randn_like = lambda t, *a, **k: torch.zeros_like(t)
        try:
            d["audio_det"] = dec.generator(xx, ref_s[:, :128], d["F0_pred"]).reshape(-1)
            # the harmonic source spectrogram feeding the noise branch, for a mid-generator gate
            g = dec.generator
            f0u = g.f0_upsamp(d["F0_pred"][:, None]).transpose(1, 2)
            har_source, _, _ = g.m_source(f0u)
            d["har_source"] = har_source.transpose(1, 2).squeeze(1)  # (1, L)
            har_spec, har_phase = g.stft.transform(d["har_source"])
            d["har"] = torch.cat([har_spec, har_phase], dim=1)
        finally:
            torch.rand, torch.randn_like = rand, randn_like
    save_file({k: v.contiguous() for k, v in d.items()}, str(out))
    return {k: tuple(v.shape) for k, v in d.items()}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kokoro-src", required=True, type=Path, help="directory containing kokoro/ (pip download kokoro)")
    ap.add_argument("--config", required=True, type=Path); ap.add_argument("--pth", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ref", type=Path); ap.add_argument("--phonemes"); ap.add_argument("--pack", type=Path, help=".npy style pack; the row for the token count is used")
    a = ap.parse_args()
    import torch
    torch.manual_seed(0)
    KModel = _import_kokoro(a.kokoro_src)
    km = KModel(repo_id="hexgrad/Kokoro-82M", config=str(a.config), model=str(a.pth), disable_complex=True).eval()
    folded = convert(km, a.out)
    print(f"wrote {a.out} ({folded} weight_norm modules folded)")
    if a.ref:
        import numpy as np
        pack = np.load(a.pack); n_ids = sum(1 for c in a.phonemes if c in km.vocab)
        row = pack.reshape(pack.shape[0], -1)[min(n_ids, pack.shape[0] - 1)]
        print("reference:", dump_reference(km, a.phonemes, row, a.ref))

if __name__ == "__main__":
    main()
