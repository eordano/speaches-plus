use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use nv_models::dots_ocr::{
    DotsMode, DotsOcrPipeline, GenerateOptions, PixelBudget, PromptStyle, RgbImage,
};

fn default_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_DOTS_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

fn load_rgb(path: &PathBuf) -> Result<RgbImage> {
    RgbImage::decode_file(path)
}

fn main() -> Result<()> {
    let mut images: Vec<PathBuf> = Vec::new();
    let mut mode = DotsMode::LayoutAll;
    let mut cpu = false;
    let mut max_new = 16384usize;
    let mut emit_json = false;
    let mut out_dir: Option<PathBuf> = None;
    let mut max_pixels: Option<usize> = None;
    let mut style: Option<PromptStyle> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--layout-only" => mode = DotsMode::LayoutOnly,
            "--plain" => mode = DotsMode::PlainOcr,
            "--cpu" => cpu = true,
            "--json" => emit_json = true,
            "--user-turn" => style = Some(PromptStyle::UserTurn),
            "--max-new" => {
                max_new = args
                    .next()
                    .context("--max-new needs a value")?
                    .parse()
                    .context("--max-new value")?
            }
            "--max-pixels" => {
                max_pixels = Some(
                    args.next()
                        .context("--max-pixels needs a value")?
                        .parse()
                        .context("--max-pixels value")?,
                )
            }
            "--out" => out_dir = Some(PathBuf::from(args.next().context("--out needs a dir")?)),
            other => images.push(PathBuf::from(other)),
        }
    }
    anyhow::ensure!(
        !images.is_empty(),
        "usage: dots_ocr [--layout-only|--plain] [--cpu] [--json] [--max-new N] [--max-pixels N] [--out DIR] <image>..."
    );

    let dir = default_dir().context("dots.ocr snapshot not found; set NV_DOTS_DIR")?;
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
    eprintln!("[dots] loading {} on {:?}", dir.display(), device);
    let t0 = Instant::now();
    let mut pipe = DotsOcrPipeline::load(&dir, &device)?;
    if let Some(mp) = max_pixels {
        let mut b = PixelBudget::from_env();
        b.max_pixels = mp.max(b.min_pixels);
        pipe.set_pixel_budget(b);
    }
    if let Some(s) = style {
        pipe.set_prompt_style(s);
    }
    eprintln!(
        "[dots] loaded in {:.2}s style={:?} budget={:?}",
        t0.elapsed().as_secs_f64(),
        pipe.prompt_style(),
        pipe.pixel_budget()
    );

    let opts = GenerateOptions {
        max_new_tokens: max_new,
        ..Default::default()
    };

    if let Some(d) = &out_dir {
        std::fs::create_dir_all(d)?;
    }
    for path in &images {
        let img = load_rgb(path)?;
        let t = Instant::now();
        let res = pipe.recognize(&img, mode, &opts)?;
        let secs = t.elapsed().as_secs_f64();
        eprintln!(
            "[dots] {} {}x{} grid={}x{} prompt={} gen={} looped={} {:.3}s ({:.1} tok/s)",
            path.display(),
            img.w,
            img.h,
            res.grid.0,
            res.grid.1,
            res.prompt_tokens,
            res.generated_tokens,
            res.looped,
            secs,
            res.generated_tokens as f64 / secs.max(1e-9)
        );
        let body = if emit_json {
            serde_json::to_string_pretty(&res.page.elements)?
        } else {
            res.text.clone()
        };
        match &out_dir {
            Some(d) => {
                let stem = path.file_stem().unwrap_or_default().to_string_lossy();
                let ext = if emit_json { "json" } else { "txt" };
                let dst = d.join(format!("{stem}.{ext}"));
                std::fs::write(&dst, &body)?;
                std::fs::write(d.join(format!("{stem}.raw.txt")), &res.raw)?;
            }
            None => println!("{body}"),
        }
    }
    Ok(())
}
