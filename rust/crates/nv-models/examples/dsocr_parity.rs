use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_models::deepseek_ocr::{
    DecoderPrecision, DeepSeekOcr2Pipeline, ResolutionMode, RgbImage, PROMPT_FREE_OCR,
    PROMPT_GROUNDING_MARKDOWN,
};

struct Dumper {
    dir: PathBuf,
    manifest: BTreeMap<String, Vec<usize>>,
}

impl Dumper {
    fn new(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest: BTreeMap::new(),
        })
    }

    fn put(&mut self, name: &str, t: &Tensor) -> Result<()> {
        let t = t.to_dtype(DType::F32)?.contiguous()?;
        let shape = t.dims().to_vec();
        let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in &v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        fs::write(self.dir.join(format!("{name}.f32")), &bytes)?;
        self.manifest.insert(name.to_string(), shape);
        Ok(())
    }

    fn put_slice(&mut self, name: &str, v: &[f32], shape: Vec<usize>) -> Result<()> {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        fs::write(self.dir.join(format!("{name}.f32")), &bytes)?;
        self.manifest.insert(name.to_string(), shape);
        Ok(())
    }

    fn finish(&self, extra: &str) -> Result<()> {
        let mut s = String::from("{\n");
        s.push_str("  \"tensors\": {\n");
        let n = self.manifest.len();
        for (i, (k, shape)) in self.manifest.iter().enumerate() {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            s.push_str(&format!("    \"{}\": [{}]", k, dims.join(", ")));
            s.push_str(if i + 1 == n { "\n" } else { ",\n" });
        }
        s.push_str("  },\n");
        s.push_str(extra);
        s.push_str("}\n");
        let mut f = fs::File::create(self.dir.join("manifest.json"))?;
        f.write_all(s.as_bytes())?;
        Ok(())
    }
}

use nv_models::deepseek_ocr::default_snapshot_dir as default_dir;

fn load_rgb(path: &Path) -> Result<RgbImage> {
    RgbImage::decode_file(path)
}

fn main() -> Result<()> {
    let mut image: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut markdown = false;
    let mut cpu = false;
    let mut gen = 0usize;
    let mut prompt_override: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(args.next().context("--out needs a value")?)),
            "--markdown" => markdown = true,
            "--cpu" => cpu = true,
            "--gen" => gen = args.next().context("--gen needs a value")?.parse()?,
            "--prompt" => prompt_override = Some(args.next().context("--prompt needs a value")?),
            other => image = Some(PathBuf::from(other)),
        }
    }
    let image = image.context("usage: dsocr_parity <image> --out <dir> [--gen N]")?;
    let out = out.context("--out is required")?;

    let dir = default_dir().context("DeepSeek-OCR-2 snapshot not found; set NV_DSOCR_DIR")?;
    let device = if cpu {
        Device::Cpu
    } else {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0)?
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cpu
    };
    eprintln!("loading DeepSeek-OCR-2 from {}", dir.display());
    let pipe = DeepSeekOcr2Pipeline::load(&dir, &device, DecoderPrecision::Bf16)?;

    let prompt: String = match prompt_override {
        Some(p) => p,
        None => {
            if markdown {
                PROMPT_GROUNDING_MARKDOWN.to_string()
            } else {
                PROMPT_FREE_OCR.to_string()
            }
        }
    };

    let img = load_rgb(&image)?;
    let prep = nv_models::deepseek_ocr::preprocess::prepare(&img, ResolutionMode::Gundam)?;
    let mut d = Dumper::new(&out)?;

    let gs = prep.global_size;
    d.put_slice("pix_global", &prep.global, vec![1, 3, gs, gs])?;
    if !prep.tiles.is_empty() {
        let ts = prep.tile_size;
        let mut flat = Vec::with_capacity(prep.tiles.len() * 3 * ts * ts);
        for t in &prep.tiles {
            flat.extend_from_slice(t);
        }
        d.put_slice("pix_tiles", &flat, vec![prep.tiles.len(), 3, ts, ts])?;
    }

    let vis = pipe.vision();
    let dtype = vis.dtype();
    let global = Tensor::from_slice(&prep.global, (1, 3, gs, gs), &Device::Cpu)?
        .to_device(vis.device())?
        .to_dtype(dtype)?;
    let sam_global = vis.sam_compress(&global)?;
    d.put("sam_global", &sam_global)?;
    let flow_global = vis.flow_features(&sam_global)?;
    d.put("flow_global", &flow_global)?;
    let proj_global = vis.project(&flow_global)?;
    d.put("proj_global", &proj_global)?;

    if !prep.tiles.is_empty() {
        let ts = prep.tile_size;
        let mut flat = Vec::with_capacity(prep.tiles.len() * 3 * ts * ts);
        for t in &prep.tiles {
            flat.extend_from_slice(t);
        }
        let tiles = Tensor::from_slice(&flat, (prep.tiles.len(), 3, ts, ts), &Device::Cpu)?
            .to_device(vis.device())?
            .to_dtype(dtype)?;
        let sam_tiles = vis.sam_compress(&tiles)?;
        d.put("sam_tiles", &sam_tiles)?;
        let flow_tiles = vis.flow_features(&sam_tiles)?;
        d.put("flow_tiles", &flow_tiles)?;
        let proj_tiles = vis.project(&flow_tiles)?;
        d.put("proj_tiles", &proj_tiles)?;
    }

    let feats = vis.encode_prepared(&prep)?;
    d.put("vision_feats", &feats)?;

    let tokens = nv_models::deepseek_ocr::build_prompt_tokens(
        |s| pipe.encode_text(s),
        &prompt,
        prep.vision_tokens(),
    )?;
    let dec = pipe.decoder();
    let mut cache = dec.new_kv_cache(tokens.len() + gen + 8)?;
    let embeds = dec.embed_tokens_with_vision(&tokens, Some(&feats))?;
    d.put("embeds", &embeds)?;
    let mut taps: Vec<Tensor> = Vec::new();
    let hidden = dec.forward_embeds_hidden_taps(&embeds, &mut cache, Some(&mut taps))?;
    for (i, t) in taps.iter().enumerate() {
        d.put(&format!("tap_{i}"), &t.squeeze(0)?)?;
    }
    let logits = dec.lm_head_forward(&hidden)?;
    let t = logits.dim(1)?;
    let last = logits
        .narrow(1, t - 1, 1)?
        .flatten_all()?
        .to_dtype(DType::F32)?;
    d.put("logits_last", &last)?;

    let lastv: Vec<f32> = last.to_vec1()?;
    let mut idx: Vec<usize> = (0..lastv.len()).collect();
    idx.sort_by(|&a, &b| lastv[b].total_cmp(&lastv[a]));

    let mut extra = String::new();
    extra.push_str(&format!(
        "  \"tokens\": [{}],\n",
        tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    extra.push_str(&format!("  \"tiles\": {},\n", prep.tiles.len()));
    extra.push_str(&format!("  \"vision_tokens\": {},\n", prep.vision_tokens()));
    extra.push_str(&format!(
        "  \"top10\": [{}],\n",
        idx.iter()
            .take(10)
            .map(|&i| format!("[{}, {:.6}]", i, lastv[i]))
            .collect::<Vec<_>>()
            .join(", ")
    ));

    let mut gen_tokens: Vec<u32> = Vec::new();
    if gen > 0 {
        let mut next = idx[0] as u32;
        gen_tokens.push(next);
        for _ in 1..gen {
            if next == nv_models::deepseek_ocr::EOS_TOKEN_ID {
                break;
            }
            let lg = dec.forward_tokens(&[next], None, &mut cache)?;
            let tt = lg.dim(1)?;
            let v: Vec<f32> = lg
                .narrow(1, tt - 1, 1)?
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1()?;
            let mut best = 0usize;
            for (i, x) in v.iter().enumerate() {
                if *x > v[best] {
                    best = i;
                }
            }
            next = best as u32;
            gen_tokens.push(next);
        }
        let text = pipe
            .tokenizer()
            .decode(&gen_tokens, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        eprintln!(
            "GEN_TEXT_HEAD {:?}",
            text.chars().take(800).collect::<String>()
        );
    }
    extra.push_str(&format!(
        "  \"gen_tokens\": [{}]\n",
        gen_tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    d.finish(&extra)?;
    eprintln!("wrote {}", out.display());
    eprintln!("prompt {:?}", prompt);
    eprintln!("tokens {} tiles {}", tokens.len(), prep.tiles.len());
    for &i in idx.iter().take(10) {
        let piece = pipe
            .tokenizer()
            .decode(&[i as u32], false)
            .unwrap_or_default();
        eprintln!("  {i}\t{:.4}\t{piece:?}", lastv[i]);
    }
    Ok(())
}
