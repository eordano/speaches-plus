use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use nv_models::deepseek_ocr::{
    preprocess, DecoderPrecision, DeepSeekOcr2Pipeline, ResolutionMode, RgbImage,
};

use nv_models::deepseek_ocr::default_snapshot_dir as default_dir;

fn load_rgb(path: &PathBuf) -> Result<RgbImage> {
    RgbImage::decode_file(path)
}

#[derive(Default, Clone)]
struct Acc {
    prep: f64,
    h2d: f64,
    sam: f64,
    comp: f64,
    flow: f64,
    proj: f64,
    total: f64,
    views: f64,
    pages: f64,
}

fn sync(dev: &Device) {
    let _ = dev.synchronize();
}

fn main() -> Result<()> {
    let mut images: Vec<PathBuf> = Vec::new();
    let mut rounds = 3usize;
    let mut res_mode = ResolutionMode::Gundam;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--base1024" => res_mode = ResolutionMode::Base1024,
            "--base768" => res_mode = ResolutionMode::Base768,
            "--rounds" => {
                rounds = args
                    .next()
                    .context("--rounds needs a value")?
                    .parse()
                    .context("--rounds value")?
            }
            other => images.push(PathBuf::from(other)),
        }
    }
    anyhow::ensure!(
        !images.is_empty(),
        "usage: dsocr_vision_prof [--rounds N] <image>..."
    );

    let dir = default_dir().context("DeepSeek-OCR-2 snapshot not found; set NV_DSOCR_DIR")?;
    let device = {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0)?
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cpu
    };
    eprintln!("loading DeepSeek-OCR-2 from {}", dir.display());
    let pipe = DeepSeekOcr2Pipeline::load(&dir, &device, DecoderPrecision::Bf16)?;
    let vis = pipe.vision();
    eprintln!("vis dtype {:?}", vis.dtype());

    let mut preps = Vec::new();
    for p in &images {
        let img = load_rgb(p)?;
        preps.push((p.clone(), img));
    }

    if std::env::var("NV_VIS_CHECK").is_ok() {
        let (_, img) = &preps[0];
        let prep = preprocess::prepare(img, res_mode)?;
        let px = vis.to_pixels(&prep.global, 1, prep.global_size)?;
        let a = vis
            .sam()
            .forward(&px)?
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?;
        std::env::set_var("NV_SAM_FUSED", "0");
        let b = vis
            .sam()
            .forward(&px)?
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?;
        let d = (&a - &b)?.abs()?;
        let maxd: f32 = d.max(0)?.to_scalar()?;
        let mean: f32 = a.abs()?.mean(0)?.to_scalar()?;
        println!(
            "fused-vs-eager SAM feature max|d|={maxd:.6} mean|a|={mean:.6} rel={:.3e}",
            maxd / mean
        );
        return Ok(());
    }

    let mut acc = Acc::default();
    for round in 0..rounds + 1 {
        let warm = round == 0;
        let mut a = Acc::default();
        for (path, img) in &preps {
            let t_page = Instant::now();
            let t = Instant::now();
            let prep = preprocess::prepare(img, res_mode)?;
            a.prep += t.elapsed().as_secs_f64();

            let mut groups: Vec<(usize, usize, Vec<f32>)> = Vec::new();
            if !prep.tiles.is_empty() {
                let ts = prep.tile_size;
                let mut flat = Vec::with_capacity(prep.tiles.len() * 3 * ts * ts);
                for tile in &prep.tiles {
                    flat.extend_from_slice(tile);
                }
                groups.push((prep.tiles.len(), ts, flat));
            }
            groups.push((1, prep.global_size, prep.global.clone()));

            let mut nviews = 0usize;
            for (b, s, data) in &groups {
                nviews += b;
                sync(&device);
                let t = Instant::now();
                let px: Tensor = vis.to_pixels(data, *b, *s)?;
                sync(&device);
                a.h2d += t.elapsed().as_secs_f64();

                let t = Instant::now();
                let feat = vis.sam().forward(&px)?;
                sync(&device);
                a.sam += t.elapsed().as_secs_f64();

                let t = Instant::now();
                let comp = vis.compressor_stage().forward(&feat)?;
                sync(&device);
                a.comp += t.elapsed().as_secs_f64();

                let t = Instant::now();
                let fl = vis.flow().forward(&comp)?;
                sync(&device);
                a.flow += t.elapsed().as_secs_f64();

                let t = Instant::now();
                let _pr = vis.project(&fl)?;
                sync(&device);
                a.proj += t.elapsed().as_secs_f64();
            }
            a.views += nviews as f64;
            a.pages += 1.0;
            a.total += t_page.elapsed().as_secs_f64();
            if warm {
                eprintln!("warmup {} views={}", path.display(), nviews);
            }
        }
        if warm {
            nv_models::deepseek_ocr::sam::sam_prof_reset();
            continue;
        }
        acc.prep += a.prep;
        acc.h2d += a.h2d;
        acc.sam += a.sam;
        acc.comp += a.comp;
        acc.flow += a.flow;
        acc.proj += a.proj;
        acc.total += a.total;
        acc.views += a.views;
        acc.pages += a.pages;
        eprintln!(
            "round {round}: total {:.1} ms/page sam {:.1} comp {:.1} flow {:.1}",
            a.total / a.pages * 1e3,
            a.sam / a.pages * 1e3,
            a.comp / a.pages * 1e3,
            a.flow / a.pages * 1e3
        );
    }

    let p = acc.pages;
    let ms = |v: f64| v / p * 1e3;
    let gpu = acc.sam + acc.comp + acc.flow + acc.proj;
    println!("pages {p} views/page {:.2}", acc.views / p);
    println!("| stage | ms/page | share of tower |");
    println!("|---|---|---|");
    let rows = [
        ("preprocess (CPU)", acc.prep),
        ("H2D + dtype", acc.h2d),
        ("SAM ViT-B", acc.sam),
        ("compressor (16x conv)", acc.comp),
        ("visual flow (Qwen2 24L)", acc.flow),
        ("projector", acc.proj),
    ];
    for (name, v) in rows {
        println!("| {name} | {:.1} | {:.1}% |", ms(v), v / gpu * 100.0);
    }
    println!("| TOWER (gpu stages) | {:.1} | 100% |", ms(gpu));
    println!("| end-to-end vision | {:.1} | -- |", ms(acc.total));
    println!("views/s (tower only) {:.2}", acc.views / gpu);
    let sub = nv_models::deepseek_ocr::sam::sam_prof_report(p);
    if !sub.trim().is_empty() {
        println!("\n{sub}");
    }
    Ok(())
}
