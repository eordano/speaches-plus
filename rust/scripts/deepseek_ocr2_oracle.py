import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import load_file

def load_encoder_module(model_dir):
    spec = importlib.util.spec_from_file_location(
        "deepencoderv2", os.path.join(model_dir, "deepencoderv2.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def synth_image(w, h):
    y, x = np.mgrid[0:h, 0:w]
    arr = np.stack(
        [(7 * x + 13 * y) % 256, (3 * x + 29 * y) % 256, (11 * x + 5 * y + 128) % 256],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(arr, "RGB")

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    return processed_images, target_aspect_ratio

def to_norm_chw(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()

def build_custom_4d_mask(seq_len, n_image, dtype=torch.float32):
    min_dtype = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_dtype, dtype=dtype)
    image_positions = torch.arange(0, n_image)
    text_positions = torch.arange(n_image, seq_len)
    mask[image_positions[:, None], image_positions] = 0.0
    for i, tp in enumerate(text_positions):
        mask[tp, image_positions] = 0.0
        mask[tp, text_positions[: i + 1]] = 0.0
    return mask[None, None, :, :]

def build_qwen2(sd):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    cfg = Qwen2Config(
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
        max_position_embeddings=131072,
        vocab_size=151936,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        attention_dropout=0.0,
        hidden_act="silu",
    )
    cfg._attn_implementation = "eager"
    qm = Qwen2Model(cfg)
    prefix = "model.qwen2_model.model.model."
    qsd = {k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = qm.load_state_dict(qsd, strict=False)
    assert not unexpected, unexpected
    real_missing = [m for m in missing if "rotary" not in m and m != "embed_tokens.weight"]
    assert not real_missing, real_missing
    qm.eval()
    q768 = sd["model.qwen2_model.query_768.weight"].float()
    q1024 = sd["model.qwen2_model.query_1024.weight"].float()
    return qm, q768, q1024

def flow_forward(qm, q768, q1024, feat):
    x = feat.flatten(2).transpose(1, 2)
    b, n, _ = x.shape
    table = q768 if n == 144 else q1024
    q = table.unsqueeze(0).expand(b, -1, -1)
    xc = torch.cat([x, q], dim=1)
    mask = build_custom_4d_mask(2 * n, n)
    out = qm(inputs_embeds=xc, attention_mask={"full_attention": mask}, use_cache=False)
    return out.last_hidden_state[:, n:, :]

def dump(out_dir, name, tensor):
    arr = tensor.detach().to(torch.float32).cpu().numpy()
    arr.astype("<f4").tofile(os.path.join(out_dir, f"{name}.bin"))
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump({"shape": list(arr.shape)}, f)
    print(f"dumped {name} {list(arr.shape)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=800)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    mod = load_encoder_module(args.model_dir)
    sd = load_file(os.path.join(args.model_dir, "model-00001-of-000001.safetensors"))

    img = synth_image(args.width, args.height)
    tiles_raw, crop_grid = dynamic_preprocess(img)
    global_view = ImageOps.pad(img, (1024, 1024), color=(127, 127, 127))
    prep_global = to_norm_chw(global_view)
    prep_tiles = torch.stack([to_norm_chw(t) for t in tiles_raw])
    dump(args.out_dir, "prep_global", prep_global)
    dump(args.out_dir, "prep_tiles", prep_tiles)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "width": args.width,
                "height": args.height,
                "crop_grid": list(crop_grid),
                "num_tiles": len(tiles_raw),
            },
            f,
        )

    sam = mod.build_sam_vit_b()
    sam_prefix = "model.sam_model."
    sam_sd = {k[len(sam_prefix):]: v.float() for k, v in sd.items() if k.startswith(sam_prefix)}
    sam.load_state_dict(sam_sd, strict=True)
    sam.eval().float()

    qm, q768, q1024 = build_qwen2(sd)
    proj_w = sd["model.projector.layers.weight"].float()
    proj_b = sd["model.projector.layers.bias"].float()
    sep = sd["model.view_seperator"].float()

    sam_global = sam(prep_global.unsqueeze(0))
    dump(args.out_dir, "sam_global", sam_global)
    sam_tiles = sam(prep_tiles)
    dump(args.out_dir, "sam_tiles", sam_tiles)

    flow_global = flow_forward(qm, q768, q1024, sam_global)
    dump(args.out_dir, "flow_global", flow_global)
    flow_tiles = flow_forward(qm, q768, q1024, sam_tiles)
    dump(args.out_dir, "flow_tiles", flow_tiles)

    proj_global = F.linear(flow_global, proj_w, proj_b).reshape(-1, 1280)
    proj_tiles = F.linear(flow_tiles, proj_w, proj_b).reshape(-1, 1280)
    feats = torch.cat([proj_tiles, proj_global, sep[None, :]], dim=0)
    dump(args.out_dir, "features", feats)

if __name__ == "__main__":
    sys.exit(main())
