#!/usr/bin/env python3
"""Reference oracle for the nv-models Voxtral TTS port (mistralai/Voxtral-4B-TTS-2603).

  nix shell --impure --inputs-from . --expr 'let p = (builtins.getFlake "nixpkgs").legacyPackages.x86_64-linux;
      in p.python3.withPackages (ps: [ ps.torch ps.mistral-common ps.safetensors ps.soundfile ps.numpy ps.pycountry ])' \
    -c python3 -P scripts/voxtral-tts-reference.py --model-dir <snapshot> --out rust/tests/data [--case en|es|all]

A plain-torch transcription of vllm-omni's VoxtralTTSAudioGenerationForConditionalGeneration,
FlowMatchingAudioTransformer and VoxtralTTSAudioTokenizer (vllm_omni/model_executor/models/voxtral_tts/,
commit e6a7e63d) over the consolidated checkpoint; the prompt comes from mistral_common's
encode_speech_request. No vllm is imported: nixpkgs' vllm neither evaluates on python 3.14 nor is cached
on 3.12, and vllm-omni is not packaged. Everything runs in bf16 as vllm serves it. Writes
voxtral_tts_<case>.{json,safetensors,wav} and voxtral_tts_voice_<voice>.safetensors.
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

CASES = {
    "en": "The quick brown fox jumps over the lazy dog.",
    "es": "Hola, ¿cómo estás hoy? Me alegra verte.",
}
BOS, AUDIO, BEGIN_AUDIO, REPEAT_AUDIO_TEXT, NEXT_AUDIO_TEXT = 1, 24, 25, 35, 36
N_AUDIO_SPECIAL, EMPTY_AUDIO, END_AUDIO = 2, 0, 1
SEMANTIC_SIZE, ACOUSTIC_LEVELS, N_ACOUSTIC = 8192, 21, 36
DIM, HEADS, KV_HEADS, HEAD_DIM, LAYERS = 3072, 32, 8, 128, 26
CODEC_DIM, CODEC_HEADS, PATCH, LATENT = 1024, 8, 240, 292
SAMPLE_RATE, SAMPLES_PER_FRAME = 24000, 1920
BF16 = torch.bfloat16

def rms_vllm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w

def rms_torch(x, w, eps):
    return F.rms_norm(x, (x.shape[-1],), w, eps)

def swiglu(x, w1, w2, w3):
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)

def rope_cis(positions, theta=1e6, dim=HEAD_DIM):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.outer(positions.float(), freqs)
    return torch.polar(torch.ones_like(ang), ang)

def apply_rope(x, cis):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * cis[:, None, :]).flatten(-2).type_as(x)

def gqa_attention(q, k, v, is_causal=False, attn_mask=None):
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=attn_mask)

class LanguageModel:
    def __init__(self, W):
        self.W = W
        self.cache = [None] * LAYERS

    def forward(self, x, pos0):
        W, T = self.W, x.shape[0]
        cis = rope_cis(torch.arange(pos0, pos0 + T))
        for i in range(LAYERS):
            p = f"layers.{i}."
            h = rms_vllm(x, W[p + "attention_norm.weight"], 1e-5)
            q = apply_rope(F.linear(h, W[p + "attention.wq.weight"]).view(T, HEADS, HEAD_DIM), cis)
            k = apply_rope(F.linear(h, W[p + "attention.wk.weight"]).view(T, KV_HEADS, HEAD_DIM), cis)
            v = F.linear(h, W[p + "attention.wv.weight"]).view(T, KV_HEADS, HEAD_DIM)
            if self.cache[i] is not None:
                k = torch.cat([self.cache[i][0], k])
                v = torch.cat([self.cache[i][1], v])
            self.cache[i] = (k, v)
            assert T == 1 or T == k.shape[0], "prefill runs from an empty cache so SDPA's causal mask aligns"
            o = gqa_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=T > 1)
            x = x + F.linear(o[0].transpose(0, 1).reshape(T, HEADS * HEAD_DIM), W[p + "attention.wo.weight"])
            h = rms_vllm(x, W[p + "ffn_norm.weight"], 1e-5)
            x = x + swiglu(h, W[p + "feed_forward.w1.weight"], W[p + "feed_forward.w2.weight"], W[p + "feed_forward.w3.weight"])
        return rms_vllm(x, W["norm.weight"], 1e-5)

class FlowMatchingAcoustic:
    def __init__(self, W, nfe, cfg_alpha):
        self.W, self.nfe = W, nfe
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(DIM // 2).float() / (DIM // 2))
        self.timesteps = torch.linspace(0, 1, nfe + 1).to(BF16)
        emb = torch.einsum("bi,j->bj", self.timesteps.view(-1, 1), inv_freq)
        self.t_emb = torch.cat((emb.cos(), emb.sin()), dim=-1).to(BF16)
        self.t_proj = F.linear(self.t_emb, W["acoustic_transformer.time_projection.weight"])
        self.dts = self.timesteps[1:] - self.timesteps[:-1]
        self.cfg_alpha = torch.tensor([[cfg_alpha]], dtype=BF16)

    def semantic_logits(self, h):
        return F.linear(h, self.W["acoustic_transformer.semantic_codebook_output.weight"]).float()

    def semantic_code(self, logits):
        logits = logits.clone()
        logits[:, EMPTY_AUDIO] = -float("inf")
        logits[:, N_AUDIO_SPECIAL + SEMANTIC_SIZE :] = -float("inf")
        return logits.argmax(dim=-1)

    def transformer(self, x):
        W = self.W
        B, S, _ = x.shape
        for i in range(3):
            p = f"acoustic_transformer.layers.{i}."
            h = rms_torch(x, W[p + "attention_norm.weight"], 1e-5)
            q = F.linear(h, W[p + "attention.wq.weight"]).view(B, S, HEADS, HEAD_DIM).transpose(1, 2)
            k = F.linear(h, W[p + "attention.wk.weight"]).view(B, S, KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = F.linear(h, W[p + "attention.wv.weight"]).view(B, S, KV_HEADS, HEAD_DIM).transpose(1, 2)
            o = gqa_attention(q, k, v).transpose(1, 2).reshape(B, S, HEADS * HEAD_DIM)
            x = x + F.linear(o, W[p + "attention.wo.weight"])
            h = rms_torch(x, W[p + "ffn_norm.weight"], 1e-5)
            x = x + swiglu(h, W[p + "feed_forward.w1.weight"], W[p + "feed_forward.w2.weight"], W[p + "feed_forward.w3.weight"])
        return rms_torch(x, W["acoustic_transformer.norm.weight"], 1e-5)

    def velocity(self, x_t, llm_proj, t_proj):
        tokens = torch.stack([F.linear(x_t.to(BF16), self.W["acoustic_transformer.input_projection.weight"]), t_proj, llm_proj], dim=1)
        return F.linear(self.transformer(tokens)[:, 0, :], self.W["acoustic_transformer.acoustic_codebook_output.weight"])

    def frame(self, h, semantic_code, noise, trace=None):
        B = h.shape[0]
        llm_proj = F.linear(torch.cat([h, torch.zeros_like(h)], dim=0), self.W["acoustic_transformer.llm_projection.weight"])
        alpha = self.cfg_alpha
        x = noise
        for i in range(self.nfe):
            t_proj = self.t_proj[i].unsqueeze(0).expand(B, -1)
            v = self.velocity(torch.cat([x, x]), llm_proj, torch.cat([t_proj, t_proj]))
            v_c, v_u = v[:B], v[B:]
            v_t = alpha * v_c + (1 - alpha) * v_u
            x = x + v_t * self.dts[i]
            if trace is not None:
                trace["v_cond"].append(v_c.float())
                trace["v_uncond"].append(v_u.float())
                trace["x"].append(x.float())
        x = torch.clamp(x, -1, 1)
        codes = (((x + 1) / 2) * (ACOUSTIC_LEVELS - 1)).round().long()
        codes[semantic_code == END_AUDIO] = EMPTY_AUDIO
        return codes + N_AUDIO_SPECIAL

def weight_norm(W, prefix):
    return torch._weight_norm(W[prefix + ".parametrizations.weight.original1"], W[prefix + ".parametrizations.weight.original0"], 0)

def pad1d(x, paddings, mode):
    left, right = paddings
    if mode != "reflect":
        return F.pad(x, paddings, mode)
    length, max_pad = x.shape[-1], max(left, right)
    extra = max_pad - length + 1 if length <= max_pad else 0
    if extra:
        x = F.pad(x, (0, extra))
    padded = F.pad(x, paddings, mode)
    return padded[..., : padded.shape[-1] - extra]

def causal_conv(x, w, kernel, stride, pad_mode):
    total = kernel - stride
    n_frames = (x.shape[-1] - kernel + total) / stride + 1
    target = (math.ceil(n_frames) - 1) * stride + (kernel - total)
    return F.conv1d(pad1d(x, (total, target - x.shape[-1]), pad_mode), w, stride=stride)

def causal_conv_transpose(x, w, kernel, stride):
    out = F.conv_transpose1d(x, w, stride=stride)
    return out[..., : out.shape[-1] - (kernel - stride)]

def alibi_slopes(n_heads):
    r = 2.0 ** (-8.0 / n_heads)
    return torch.tensor([r**i for i in range(n_heads)], dtype=torch.float32)

class CodecDecoder:
    BLOCKS, WINDOWS, UPSAMPLERS = (1, 3, 5, 7), (2, 4, 8, 16), (2, 4, 6)

    def __init__(self, W):
        self.W = W
        self.slopes = alibi_slopes(CODEC_HEADS)
        self.embedding = W["audio_tokenizer.quantizer.semantic_codebook.embedding_sum"] / W["audio_tokenizer.quantizer.semantic_codebook.cluster_usage"].clamp(min=1e-5)[:, None]

    def attention(self, x, p, window):
        W = self.W
        B, S, _ = x.shape
        q = rms_torch(F.linear(x, W[p + "wq.weight"]), W[p + "q_norm.weight"], 1e-6).view(B, S, CODEC_HEADS, HEAD_DIM).transpose(1, 2)
        k = rms_torch(F.linear(x, W[p + "wk.weight"]), W[p + "k_norm.weight"], 1e-6).view(B, S, CODEC_HEADS, HEAD_DIM).transpose(1, 2)
        v = F.linear(x, W[p + "wv.weight"]).view(B, S, CODEC_HEADS, HEAD_DIM).transpose(1, 2)
        pos = torch.arange(S)
        rel = (pos[None, :] - pos[:, None])[None]
        bias = self.slopes.to(x.dtype).view(CODEC_HEADS, 1, 1) * rel.to(x.dtype)
        bias = bias.masked_fill(rel > 0, float("-inf")).masked_fill(rel < -window, float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None])
        return F.linear(o.transpose(1, 2).reshape(B, S, CODEC_HEADS * HEAD_DIM), W[p + "wo.weight"])

    def layer(self, x, p, window):
        W = self.W
        x = x + W[p + "attention_scale"] * self.attention(rms_torch(x, W[p + "attention_norm.weight"], 1e-2), p + "attention.", window)
        h = rms_torch(x, W[p + "ffn_norm.weight"], 1e-2)
        return x + W[p + "ffn_scale"] * swiglu(h, W[p + "feed_forward.w1.weight"], W[p + "feed_forward.w2.weight"], W[p + "feed_forward.w3.weight"])

    def latent(self, codes):
        semantic = F.embedding(codes[:, 0], self.embedding)
        acoustic = (codes[:, 1:] * 2 / (ACOUSTIC_LEVELS - 1) - 1).to(BF16)
        return torch.cat([semantic, acoustic], dim=-1)[None].transpose(1, 2)

    def decode(self, codes):
        x = causal_conv(self.latent(codes), weight_norm(self.W, "audio_tokenizer.decoder_blocks.0.conv"), 3, 1, "replicate").transpose(1, 2)
        for i, (b, window) in enumerate(zip(self.BLOCKS, self.WINDOWS)):
            for layer in range(2):
                x = self.layer(x, f"audio_tokenizer.decoder_blocks.{b}.layers.{layer}.", window)
            if i < len(self.UPSAMPLERS):
                w = weight_norm(self.W, f"audio_tokenizer.decoder_blocks.{self.UPSAMPLERS[i]}.conv")
                x = causal_conv_transpose(x.transpose(1, 2), w, 4, 2).transpose(1, 2)
        x = causal_conv(x.transpose(1, 2), weight_norm(self.W, "audio_tokenizer.output_proj.conv"), 7, 1, "reflect")
        return x.transpose(1, 2).reshape(-1).float()

    def decode_chunked(self, codes, first, chunk, left_context):
        out, emitted = [], 0
        T = codes.shape[0]
        while emitted < T:
            size = first if emitted == 0 and T > chunk else chunk
            end = min(T, emitted + size) if emitted or T > chunk else T
            if emitted == 0 and T <= chunk:
                end = T
            start = max(0, end - left_context - (end - emitted))
            audio = self.decode(codes[start:end])
            out.append(audio[(emitted - start) * SAMPLES_PER_FRAME :])
            emitted = end
        return torch.cat(out)

def codec_input(frames):
    eoa = (frames[:, 0] == END_AUDIO).nonzero()
    cut = int(eoa[0]) if eoa.numel() else frames.shape[0]
    return frames[:cut] - N_AUDIO_SPECIAL

def build_prompt(model_dir, text, voice):
    from mistral_common.protocol.speech.request import SpeechRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    it = MistralTokenizer.from_file(os.path.join(model_dir, "tekken.json")).instruct_tokenizer
    text_ids = it.tokenizer.encode(text, bos=False, eos=False)
    n_voice = it.audio_encoder.audio_config.voice_num_audio_tokens[voice]
    expected = [BOS, BEGIN_AUDIO] + [AUDIO] * n_voice + [NEXT_AUDIO_TEXT] + text_ids + [REPEAT_AUDIO_TEXT, BEGIN_AUDIO]
    tokens = it.encode_speech_request(SpeechRequest(input=text, voice=voice)).tokens
    assert tokens == expected, "mistral_common's speech prompt layout differs from the documented one"
    return tokens, text_ids, n_voice

def load_weights(path):
    W = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            W[k] = f.get_tensor(k)
    return W

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default="all", choices=["all", *CASES])
    ap.add_argument("--voice", default="neutral_female")
    ap.add_argument("--nfe", type=int, default=7)
    ap.add_argument("--cfg-alpha", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames", type=int, default=250)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    t0 = time.time()
    W = load_weights(os.path.join(args.model_dir, "consolidated.safetensors"))
    voice = torch.load(os.path.join(args.model_dir, "voice_embedding", f"{args.voice}.pt"), map_location="cpu", weights_only=True)
    print(f"loaded {len(W)} tensors in {time.time() - t0:.1f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    save_file({"voice_embedding": voice.contiguous()}, os.path.join(args.out, f"voxtral_tts_voice_{args.voice}.safetensors"))
    codebook_sizes = [SEMANTIC_SIZE + N_AUDIO_SPECIAL] + [ACOUSTIC_LEVELS + N_AUDIO_SPECIAL] * N_ACOUSTIC
    offsets = torch.tensor(np.cumsum([0] + codebook_sizes[:-1]))
    table = W["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]
    flow = FlowMatchingAcoustic(W, args.nfe, args.cfg_alpha)
    codec = CodecDecoder(W)
    import mistral_common
    import safetensors

    for case in CASES if args.case == "all" else [args.case]:
        text = CASES[case]
        tokens, text_ids, n_voice = build_prompt(args.model_dir, text, args.voice)
        assert voice.shape == (n_voice, DIM), (voice.shape, n_voice)
        ids = torch.tensor(tokens)
        emb = W["mm_audio_embeddings.tok_embeddings.weight"][ids].clone()
        emb[ids == AUDIO] = voice
        lm = LanguageModel(W)
        gen = torch.Generator().manual_seed(args.seed)
        t1 = time.time()
        h = lm.forward(emb, 0)[-1:]
        prefill_s = time.time() - t1
        frames, noises, hidden, frame_embeds, trace, logits0 = [], [], {}, [], None, None
        for t in range(args.max_frames):
            logits = flow.semantic_logits(h)
            semantic = flow.semantic_code(logits)
            noise = torch.randn(1, N_ACOUSTIC, dtype=BF16, generator=gen)
            if t == 0:
                logits0 = logits[0].clone()
                trace = {"v_cond": [], "v_uncond": [], "x": []}
            acoustic = flow.frame(h, semantic, noise, trace if t == 0 else None)
            frame = torch.cat([semantic.view(1, 1), acoustic], dim=1)[0]
            frames.append(frame)
            noises.append(noise[0].float())
            if t < 3:
                hidden[t] = h[0].float()
            if semantic.item() == END_AUDIO:
                hidden["last"] = h[0].float()
                break
            e = table[frame + offsets].sum(dim=0, keepdim=True)
            if t == 0:
                frame_embeds.append(e[0].float())
            h = lm.forward(e, len(tokens) + t)[-1:]
        else:
            raise SystemExit(f"{case}: no END_AUDIO within {args.max_frames} frames")
        decode_s = time.time() - t1 - prefill_s
        frames = torch.stack(frames)
        codes = codec_input(frames)
        t2 = time.time()
        audio = codec.decode(codes)
        codec_s = time.time() - t2
        chunked = codec.decode_chunked(codes, first=5, chunk=25, left_context=25)
        chunk_diff = (chunked - audio).abs().max().item()
        wav = audio.clamp(-1, 1).numpy()
        stem = os.path.join(args.out, f"voxtral_tts_{case}")
        sf.write(stem + ".wav", wav, SAMPLE_RATE, subtype="PCM_16")
        tensors = {
            "prompt_ids": ids.to(torch.int32),
            "frames": frames.to(torch.int32),
            "codec_codes": codes.to(torch.int32),
            "noise": torch.stack(noises).contiguous(),
            "timesteps": flow.timesteps.float(),
            "semantic_logits_frame0": logits0,
            "flow_x_frame0": torch.cat([noises[0][None], torch.cat(trace["x"])]),
            "flow_v_cond_frame0": torch.cat(trace["v_cond"]),
            "flow_v_uncond_frame0": torch.cat(trace["v_uncond"]),
            "frame_embed_0": frame_embeds[0],
            "waveform": audio.contiguous(),
        }
        for k, v in hidden.items():
            tensors[f"lm_hidden_frame_{k}"] = v
        save_file(tensors, stem + ".safetensors")
        meta = {
            "case": case,
            "text": text,
            "voice": args.voice,
            "text_token_ids": text_ids,
            "prompt_token_ids": tokens,
            "n_voice_tokens": n_voice,
            "n_frames": int(frames.shape[0]),
            "n_codec_frames": int(codes.shape[0]),
            "semantic_codes": frames[:, 0].tolist(),
            "frames": frames.tolist(),
            "nfe": args.nfe,
            "cfg_alpha": args.cfg_alpha,
            "cfg_alpha_bf16": flow.cfg_alpha.item(),
            "timesteps_bf16": flow.timesteps.float().tolist(),
            "seed": args.seed,
            "sample_rate": SAMPLE_RATE,
            "n_samples": int(audio.shape[0]),
            "duration_s": audio.shape[0] / SAMPLE_RATE,
            "chunked_decode_max_abs_diff": chunk_diff,
            "chunked_decode_geometry": {"first_chunk_frames": 5, "chunk_frames": 25, "left_context_frames": 25},
            "timing_s": {"prefill": prefill_s, "frames": decode_s, "codec": codec_s},
            "oracle": {
                "script": "scripts/voxtral-tts-reference.py",
                "transcribed_from": "vllm-project/vllm-omni e6a7e63d vllm_omni/model_executor/models/voxtral_tts",
                "model": "mistralai/Voxtral-4B-TTS-2603@b81be46c3777f88621676791b512bb01dc1cb970",
                "device": "cpu",
                "dtype": "bfloat16",
                "torch": torch.__version__,
                "mistral_common": mistral_common.__version__,
                "safetensors": safetensors.__version__,
                "python": platform.python_version(),
                "command": " ".join(a.replace(os.path.expanduser("~"), "~") for a in sys.argv),
            },
        }
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        print(json.dumps({k: meta[k] for k in ("case", "n_frames", "duration_s", "chunked_decode_max_abs_diff", "timing_s")}), flush=True)

if __name__ == "__main__":
    main()
