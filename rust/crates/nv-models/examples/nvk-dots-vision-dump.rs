use anyhow::{ensure, Context, Result};
use candle_core::Device;
use clap::Parser;
use nv_models::dots_ocr::{
    build_prompt_tokens, prepare, DotsMode, DotsVisionConfig, DotsVisionTowerWgpu, PixelBudget,
    PromptStyle, RgbImage, ENDOFASSISTANT_TOKEN_ID,
};
use nv_train::dots::write_dump_sample;
use std::io::Write;
use std::path::PathBuf;

pub const DUMP_REPLAYS_THE_SERVING_PATH_SO_TRAIN_AND_SERVE_SEE_THE_SAME_INPUTS: &str =
    "each sample runs the exact serving pipeline pieces in order (smart-resize + patchify via \
     prepare, the wgpu vision tower, build_prompt_tokens with the same mode prompt and \
     PromptStyle::from_env) and the label is appended after the assistant token, so the \
     trainer's spliced sequence is byte-identical to what DotsQ3dDecoder::generate prefills; \
     drift here would train against inputs the served model never sees";

#[derive(Parser, Debug)]
#[command(
    name = "nvk-dots-vision-dump",
    about = "Precompute dots.ocr QLoRA training samples: reads a JSONL of {image, text, bbox?} \
             rows, replays the serving preprocessing + wgpu vision tower + prompt assembly on \
             the rednote-hilab vision/tokenizer snapshot, and writes one safetensors per row \
             (ids, rows, vision_start, completion_start) for nvk-dots-lora-train"
)]
struct Cli {
    #[arg(long)]
    dots_dir: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "plain")]
    mode: String,
    #[arg(long, default_value_t = ENDOFASSISTANT_TOKEN_ID)]
    eos_id: u32,
    #[arg(long)]
    max_pixels: Option<usize>,
    #[arg(long, default_value_t = 8)]
    bbox_margin_px: u32,
}

#[derive(serde::Deserialize)]
struct Row {
    image: String,
    text: String,
    bbox: Option<[f64; 4]>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mode = match cli.mode.as_str() {
        "plain" => DotsMode::PlainOcr,
        "layout-all" => DotsMode::LayoutAll,
        "layout-only" => DotsMode::LayoutOnly,
        other => anyhow::bail!("unknown --mode {other} (plain|layout-all|layout-only)"),
    };
    let mut budget = PixelBudget::from_env();
    if let Some(mp) = cli.max_pixels {
        budget.max_pixels = mp.max(budget.min_pixels);
    }

    let tokenizer = nv_tokenizer::load_tokenizer(&cli.dots_dir.join("tokenizer.json"))?;
    let encode = |s: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    };
    let loader = nv_weights::WeightLoader::open_dir(&cli.dots_dir, &Device::Cpu)?;
    let tower =
        DotsVisionTowerWgpu::from_loader(&loader, "vision_tower.", DotsVisionConfig::default())?;
    let hidden = DotsVisionConfig::default().hidden_size;
    let style = PromptStyle::from_env();

    std::fs::create_dir_all(&cli.out).with_context(|| format!("create {}", cli.out.display()))?;
    let data = std::fs::read_to_string(&cli.data)
        .with_context(|| format!("read {}", cli.data.display()))?;
    let mut index = std::fs::File::create(cli.out.join("index.jsonl"))?;

    let mut written = 0usize;
    for (li, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: Row =
            serde_json::from_str(line).with_context(|| format!("parse jsonl line {}", li + 1))?;
        let img = RgbImage::decode_file(std::path::Path::new(&row.image))
            .with_context(|| format!("open image {}", row.image))?;
        let img = match row.bbox {
            Some([x1, y1, x2, y2]) => {
                let m = cli.bbox_margin_px as f64;
                let (w, h) = (img.w as f64, img.h as f64);
                let cx1 = (x1 - m).max(0.0).min(w - 1.0) as usize;
                let cy1 = (y1 - m).max(0.0).min(h - 1.0) as usize;
                let cx2 = (x2 + m).max(0.0).min(w) as usize;
                let cy2 = (y2 + m).max(0.0).min(h) as usize;
                ensure!(
                    cx2 > cx1 && cy2 > cy1,
                    "line {}: degenerate bbox {:?}",
                    li + 1,
                    row.bbox
                );
                img.crop(cx1, cy1, cx2, cy2)
            }
            None => img,
        };
        let prep = prepare(&img, budget)?;
        let rows = tower.encode(&prep)?;
        let n_vis = prep.num_vision_tokens();
        ensure!(
            rows.len() == n_vis * hidden,
            "line {}: tower returned {} floats, expected {}",
            li + 1,
            rows.len(),
            n_vis * hidden
        );
        let (mut ids, vision_start) = build_prompt_tokens(encode, mode.prompt(), n_vis, style)?;
        let completion_start = ids.len();
        ids.extend(encode(&row.text)?);
        ids.push(cli.eos_id);

        let file = format!("{:05}.safetensors", li + 1);
        write_dump_sample(
            &cli.out.join(&file),
            &ids,
            &rows,
            hidden,
            vision_start,
            completion_start,
        )?;
        writeln!(
            index,
            "{}",
            serde_json::json!({
                "file": file,
                "image": row.image,
                "bbox": row.bbox,
                "n_vis": n_vis,
                "len": ids.len(),
                "completion_start": completion_start,
            })
        )?;
        written += 1;
        eprintln!(
            "[dots-vision-dump] {}/{} {} n_vis {} len {}",
            written,
            data.lines().filter(|l| !l.trim().is_empty()).count(),
            file,
            n_vis,
            ids.len()
        );
    }
    ensure!(
        written > 0,
        "no samples written from {}",
        cli.data.display()
    );
    eprintln!(
        "[dots-vision-dump] wrote {written} samples to {}",
        cli.out.display()
    );
    Ok(())
}
