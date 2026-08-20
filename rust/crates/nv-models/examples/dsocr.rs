use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use nv_models::deepseek_ocr::{
    DecoderPrecision, DeepSeekOcr2Pipeline, GenerateOptions, ResolutionMode, RgbImage,
    PROMPT_FREE_OCR, PROMPT_GROUNDING_MARKDOWN,
};

use nv_models::deepseek_ocr::default_snapshot_dir as default_dir;

fn load_rgb(path: &PathBuf) -> Result<RgbImage> {
    RgbImage::decode_file(path)
}

fn main() -> Result<()> {
    let mut images: Vec<PathBuf> = Vec::new();
    let mut markdown = false;
    let mut nvfp4 = false;
    let mut cpu = false;
    let mut max_new = 2048usize;
    let mut probe = false;
    let mut res_mode = ResolutionMode::Gundam;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--probe" => probe = true,
            "--markdown" => markdown = true,
            "--nvfp4" => nvfp4 = true,
            "--cpu" => cpu = true,
            "--base1024" => res_mode = ResolutionMode::Base1024,
            "--base768" => res_mode = ResolutionMode::Base768,
            "--max-new" => {
                max_new = args
                    .next()
                    .context("--max-new needs a value")?
                    .parse()
                    .context("--max-new value")?
            }
            other => images.push(PathBuf::from(other)),
        }
    }
    anyhow::ensure!(!images.is_empty(), "usage: dsocr [--markdown] [--nvfp4] [--cpu] [--base1024|--base768] [--max-new N] <image>...");

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
    let precision = if nvfp4 {
        DecoderPrecision::Nvfp4
    } else {
        DecoderPrecision::Bf16
    };
    eprintln!(
        "loading DeepSeek-OCR-2 from {} device={:?} precision={:?}",
        dir.display(),
        device,
        precision
    );
    let t0 = Instant::now();
    let pipe = DeepSeekOcr2Pipeline::load(&dir, &device, precision)?;
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let prompt = if markdown {
        PROMPT_GROUNDING_MARKDOWN
    } else {
        PROMPT_FREE_OCR
    };
    let opts = GenerateOptions {
        max_new_tokens: max_new,
        ..Default::default()
    };
    if std::env::var("NV_DSOCR_PHASES").is_ok() {
        for path in &images {
            let t = Instant::now();
            let img = load_rgb(path)?;
            let t_load = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let prep = nv_models::deepseek_ocr::preprocess::prepare(&img, res_mode)?;
            let t_prep = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let feats = pipe.vision().encode_prepared(&prep)?;
            let t_vis = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let toks = nv_models::deepseek_ocr::build_prompt_tokens(
                |s| pipe.encode_text(s),
                prompt,
                prep.vision_tokens(),
            )?;
            let t_tok = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut cache = pipe.decoder().new_kv_cache(toks.len() + max_new + 4)?;
            let _ = pipe
                .decoder()
                .forward_tokens(&toks, Some(&feats), &mut cache)?;
            let t_prefill = t.elapsed().as_secs_f64();
            eprintln!(
                "[phases] {} tiletok={} vtok={} ptok={} | decode_png {:.3}s prepare {:.3}s vision {:.3}s tokenize {:.3}s prefill {:.3}s",
                path.display(),
                prep.tile_tokens(),
                prep.vision_tokens(),
                toks.len(),
                t_load,
                t_prep,
                t_vis,
                t_tok,
                t_prefill
            );
        }
        return Ok(());
    }
    if probe {
        for path in &images {
            let img = load_rgb(path)?;
            let (tokens, feats) = {
                let prep = nv_models::deepseek_ocr::preprocess::prepare(&img, res_mode)?;
                let feats = pipe.vision().encode_prepared(&prep)?;
                let toks = nv_models::deepseek_ocr::build_prompt_tokens(
                    |s| pipe.encode_text(s),
                    prompt,
                    prep.vision_tokens(),
                )?;
                (toks, feats)
            };
            let mut cache = pipe.decoder().new_kv_cache(tokens.len() + 4)?;
            let logits = pipe
                .decoder()
                .forward_tokens(&tokens, Some(&feats), &mut cache)?;
            let t = logits.dim(1)?;
            let last: Vec<f32> = logits
                .narrow(1, t - 1, 1)?
                .flatten_all()?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1()?;
            let mut idx: Vec<usize> = (0..last.len()).collect();
            idx.sort_by(|&a, &b| last[b].total_cmp(&last[a]));
            println!("=== probe {} ===", path.display());
            for &i in idx.iter().take(10) {
                let piece = pipe
                    .tokenizer()
                    .decode(&[i as u32], false)
                    .unwrap_or_default();
                println!("{}\t{:.4}\t{:?}", i, last[i], piece);
            }
        }
        return Ok(());
    }
    for path in &images {
        let img = load_rgb(path)?;
        let t = Instant::now();
        let (tokens, prep) = pipe.generate_tokens(&img, prompt, res_mode, &opts)?;
        let dt = t.elapsed().as_secs_f64();
        let text = pipe
            .tokenizer()
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        eprintln!(
            "{}: {}x{} tiles={} vision_tokens={} out_tokens={} {:.2}s ({:.1} tok/s)",
            path.display(),
            img.w,
            img.h,
            prep.tiles.len(),
            prep.vision_tokens(),
            tokens.len(),
            dt,
            tokens.len() as f64 / dt
        );
        println!("=== {} ===", path.display());
        println!("{text}");
    }
    Ok(())
}
