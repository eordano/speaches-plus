"""HF transformers reference for the Qwen3.8-Flash-Next vision path, CPU only.

Writes <out>/images/*.png, <out>/ref.safetensors (HF PIL-processor pixel_values and grid per image, the
Qwen4ExpVisionModel merged embeddings in f32 and bf16, the rotary cos/sin of a two-image chat prompt),
<out>/raw.safetensors (the source RGB bytes) and <out>/meta.json (template text, input ids, HF get_rope_index
positions and rope delta). nv-omni tests/qwen4_exp_vision_hf_ref.rs and nv-models qwen3_mm_splice read them
through NV_Q4EXP_VISION_REF=<out>.

    python qwen4_exp_vision_hf_ref.py <out> <dejavu truetype dir>

Needs torch, transformers >= 5.17 (models/qwen4_exp), safetensors, numpy and Pillow; the snapshot path is the
first models--nvidia--Qwen3.8-Flash-Next-NVFP4 snapshot under NV_Q4EXP_SNAPSHOT_GLOB.
"""
import glob, json, os, sys, types
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(16)
SNAP = Path(sorted(glob.glob(os.environ.get("NV_Q4EXP_SNAPSHOT_GLOB", "/nix/store/*-hf-hub-cache-*/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/*")))[0])
OUT = Path(sys.argv[1]); (OUT / "images").mkdir(parents=True, exist_ok=True)
FONTS = Path(sys.argv[2])
font = ImageFont.truetype(str(FONTS / "DejaVuSans.ttf"), 30)
bold = ImageFont.truetype(str(FONTS / "DejaVuSans-Bold.ttf"), 30)

def ocr():
    img = Image.new("RGB", (900, 220), "white"); d = ImageDraw.Draw(img)
    for i, l in enumerate(["INVOICE", "Customer: Harbor Lights Ltd", "Total due: 1,284.50 EUR"]):
        d.text((40, 35 + 50 * i), l, fill="black", font=font)
    return img
def bars():
    data = {"Jan": 12, "Feb": 30, "Mar": 21, "Apr": 9}
    img = Image.new("RGB", (720, 520), "white"); d = ImageDraw.Draw(img); top = 30
    for i, (k, v) in enumerate(data.items()):
        x = 60 + 160 * i; h = int(340 * v / top)
        d.rectangle([x, 440 - h, x + 100, 440], fill=(70, 110, 180))
        d.text((x + 20, 400 - h), str(v), fill="black", font=bold); d.text((x + 10, 455), k, fill="black", font=font)
    return img
def ui():
    img = Image.new("RGB", (720, 464), (235, 235, 235)); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 720, 70], fill=(45, 55, 72)); d.text((30, 18), "Network Preferences", fill="white", font=bold)
    d.text((40, 110), "Connected to vpn-ams-2", fill="black", font=font)
    d.rectangle([430, 360, 690, 425], fill="white", outline=(60, 60, 60), width=2); d.text((450, 375), "Disconnect", fill="black", font=font)
    return img
def scene():
    w, h = 640, 400
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = np.stack([120 + 60 * y / h, 170 + 40 * y / h, 235 - 20 * y / h], -1)
    img = np.where((y > 260)[..., None], np.stack([60 + 0 * x, 140 - 40 * (y - 260) / 140, 50 + 0 * x], -1), sky)
    sun = ((x - 500) ** 2 + (y - 90) ** 2) < 45 ** 2
    img[sun] = [250, 220, 60]
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    d.rectangle([120, 190, 300, 300], fill=(180, 40, 40)); d.polygon([(110, 190), (210, 120), (310, 190)], fill=(90, 50, 30))
    d.rectangle([190, 240, 230, 300], fill=(70, 40, 20)); d.ellipse([380, 230, 440, 290], fill=(30, 110, 30))
    d.rectangle([405, 285, 415, 320], fill=(90, 60, 30))
    return img
def tiny():
    img = Image.new("RGB", (120, 90), (240, 240, 200)); d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 60, 60], fill=(200, 30, 30)); d.ellipse([70, 30, 110, 80], fill=(30, 30, 200))
    return img

images = {"ocr": ocr(), "bars": bars(), "ui": ui(), "scene": scene(), "tiny": tiny()}
for k, im in images.items():
    im.save(OUT / "images" / f"{k}.png")

from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

cfg = AutoConfig.from_pretrained(SNAP)
pp = json.load(open(SNAP / "processor_config.json"))["image_processor"]
proc = Qwen2VLImageProcessorPil(**{k: v for k, v in pp.items() if k != "image_processor_type"})

state = {}
idx = json.load(open(SNAP / "model.safetensors.index.json"))["weight_map"]
for f in sorted({v for k, v in idx.items() if k.startswith("model.visual.")}):
    with safe_open(str(SNAP / f), "pt") as s:
        for k in s.keys():
            if k.startswith("model.visual."):
                state[k[len("model.visual."):]] = s.get_tensor(k)
print("visual tensors", len(state), "params", sum(v.numel() for v in state.values()))

out = {}
meta = {"images": {}}
models = {}
for dt in (torch.float32, torch.bfloat16):
    vc = cfg.vision_config
    vc._attn_implementation = "sdpa"
    m = Qwen4ExpVisionModel._from_config(vc, dtype=dt).eval()
    missing = m.load_state_dict({k: v.to(dt) for k, v in state.items()}, strict=False)
    print(dt, "missing", missing.missing_keys, "unexpected", missing.unexpected_keys)
    models[dt] = m
for k, im in images.items():
    bf = proc(images=[im], return_tensors="pt")
    pv, g = bf["pixel_values"], bf["image_grid_thw"]
    out[f"pix.{k}"] = pv.float().contiguous()
    out[f"grid.{k}"] = g[0].to(torch.int64)
    with torch.no_grad():
        for dt, tag in ((torch.float32, "f32"), (torch.bfloat16, "bf16")):
            r = models[dt](pv.to(dt), grid_thw=g)
            out[f"emb_{tag}.{k}"] = r.pooler_output.float().contiguous()
    a, b = out[f"emb_f32.{k}"], out[f"emb_bf16.{k}"]
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    meta["images"][k] = {"size_wh": list(im.size), "grid_thw": g[0].tolist(), "tokens": int(a.shape[0]),
                         "hf_bf16_vs_f32_cos": cos, "hf_bf16_vs_f32_maxabs": (a - b).abs().max().item(),
                         "emb_absmax": a.abs().max().item()}
    print(k, meta["images"][k], flush=True)

tok = AutoTokenizer.from_pretrained(SNAP)
tmpl = (SNAP / "chat_template.jinja").read_text()
msgs = [{"role": "user", "content": [{"type": "text", "text": "Compare these."}, {"type": "image"}, {"type": "text", "text": "and"}, {"type": "image"}, {"type": "text", "text": "Which is bigger?"}]}]
text = tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True)
meta["template_text"] = text
pair = ["ui", "tiny"]
merge = cfg.vision_config.spatial_merge_size
exp = text
for k in pair:
    t, h, w = out[f"grid.{k}"].tolist()
    exp = exp.replace("<|image_pad|>", "<|placeholder|>" * (t * h * w // merge // merge), 1)
exp = exp.replace("<|placeholder|>", "<|image_pad|>")
ids = tok(exp, add_special_tokens=False)["input_ids"]
img_id = cfg.image_token_id
tt = torch.tensor([[1 if i == img_id else 0 for i in ids]])
fake = types.SimpleNamespace(config=cfg)
fake.get_vision_position_ids = types.MethodType(Qwen3_5MoeModel.get_vision_position_ids, fake)
pos, delta = Qwen3_5MoeModel.get_rope_index(fake, torch.tensor([ids]), tt, image_grid_thw=torch.stack([out[f"grid.{k}"] for k in pair]))
meta["layout"] = {"images": pair, "input_ids": ids, "position_ids": pos[:, 0].tolist(), "rope_delta": int(delta.item())}
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding
rot = Qwen4ExpTextRotaryEmbedding(cfg.text_config)
cos, sin = rot(torch.zeros(1, dtype=torch.float32), pos)
out["layout.cos"] = cos[0].float().contiguous()
out["layout.sin"] = sin[0].float().contiguous()
save_file(out, str(OUT / "ref.safetensors"))
json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
print("delta", meta["layout"]["rope_delta"], "n", len(ids))

raw = {f"raw.{k}": torch.from_numpy(np.asarray(im.convert("RGB")).copy()) for k, im in images.items()}
save_file(raw, str(OUT / "raw.safetensors"))
