#![cfg(feature = "browser")]

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::Device;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_ocr_web::train::e4b::{
    dump_window_samples, e4b_train_options, window_head_shard_bytes, window_layers_shard_bytes,
    E4bDims, E4bTrainSample, E4B_BASE_MODEL_NAME,
};
use nv_ocr_web::train::TrainOptions;
use nv_weights::WeightLoader;

pub const LAYERS_SHARD: &str = "e4b-window-layers.safetensors";
pub const HEAD_SHARD: &str = "e4b-window-head.safetensors";
pub const WINDOW_JSON: &str = "e4b-window.json";
pub const INDEX_JSONL: &str = "index.jsonl";

pub const DUMP_ROWS_ARE_GEMMA_CHAT_TURNS_AND_THE_MODEL_TURN_IS_THE_COMPLETION: &str =
    "each JSONL row {user, model} becomes <bos><start_of_turn>user\\n{user}<end_of_turn>\\n\
     <start_of_turn>model\\n followed by {model}<end_of_turn>\\n, with completion_start at the \
     model turn -- the same rows the e4b_train_gate suite trains natively";

const USAGE: &str = "nvk-e4b-window-dump --e4b-dir <snapshot> --data <rows.jsonl> --out <dir> \
                     [--grad-layers N] [--samples-only]";

struct Cli {
    e4b_dir: PathBuf,
    data: PathBuf,
    out: PathBuf,
    grad_layers: usize,
    samples_only: bool,
}

fn parse_cli() -> Result<Cli> {
    let mut e4b_dir = None;
    let mut data = None;
    let mut out = None;
    let mut grad_layers = TrainOptions::default().grad_layers;
    let mut samples_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || {
            args.next()
                .with_context(|| format!("{a} needs a value\n{USAGE}"))
        };
        match a.as_str() {
            "--e4b-dir" => e4b_dir = Some(PathBuf::from(val()?)),
            "--data" => data = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--grad-layers" => grad_layers = val()?.parse().context("--grad-layers")?,
            "--samples-only" => samples_only = true,
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    Ok(Cli {
        e4b_dir: e4b_dir.with_context(|| format!("--e4b-dir is required\n{USAGE}"))?,
        data: data.with_context(|| format!("--data is required\n{USAGE}"))?,
        out: out.with_context(|| format!("--out is required\n{USAGE}"))?,
        grad_layers,
        samples_only,
    })
}

pub fn chat_sample(tok: &tokenizers::Tokenizer, user: &str, model: &str) -> Result<E4bTrainSample> {
    let _ = DUMP_ROWS_ARE_GEMMA_CHAT_TURNS_AND_THE_MODEL_TURN_IS_THE_COMPLETION;
    let bos = tok
        .token_to_id("<bos>")
        .context("gemma tokenizer carries <bos>")?;
    let enc = |s: &str| -> Result<Vec<u32>> {
        Ok(tok
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("encode {s:?}: {e}"))?
            .get_ids()
            .to_vec())
    };
    let mut ids = vec![bos];
    ids.extend(enc(&format!(
        "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
    ))?);
    let completion_start = ids.len();
    ids.extend(enc(&format!("{model}<end_of_turn>\n"))?);
    Ok(E4bTrainSample {
        ids,
        completion_start,
    })
}

fn read_rows(path: &Path, tok: &tokenizers::Tokenizer) -> Result<Vec<E4bTrainSample>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("{}:{}", path.display(), n + 1))?;
        let field = |k: &str| -> Result<&str> {
            v.get(k)
                .and_then(|x| x.as_str())
                .with_context(|| format!("{}:{}: row needs a string {k}", path.display(), n + 1))
        };
        out.push(chat_sample(tok, field("user")?, field("model")?)?);
    }
    if out.is_empty() {
        bail!("{}: no rows", path.display());
    }
    Ok(out)
}

fn main() -> Result<()> {
    let cli = parse_cli()?;
    let config_json = std::fs::read_to_string(cli.e4b_dir.join("config.json"))
        .with_context(|| format!("read {}/config.json", cli.e4b_dir.display()))?;
    let tok = tokenizers::Tokenizer::from_file(cli.e4b_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
    let samples = read_rows(&cli.data, &tok)?;
    let loader = WeightLoader::open_dir(&cli.e4b_dir, &Device::Cpu).context("open E4B snapshot")?;
    let d = E4bDims::from_config_json(&config_json)?;
    let opts = TrainOptions {
        grad_layers: cli.grad_layers,
        ..e4b_train_options()
    };
    let window_start = d.layers - cli.grad_layers;
    std::fs::create_dir_all(&cli.out).with_context(|| format!("create {}", cli.out.display()))?;
    std::fs::write(cli.out.join("config.json"), &config_json)?;
    if !cli.samples_only {
        let t0 = std::time::Instant::now();
        let layers = window_layers_shard_bytes(&loader, &d, window_start)?;
        std::fs::write(cli.out.join(LAYERS_SHARD), &layers)?;
        let head = window_head_shard_bytes(&loader, &d)?;
        std::fs::write(cli.out.join(HEAD_SHARD), &head)?;
        eprintln!(
            "[e4b-window-dump] shards: layers {} bytes, head {} bytes in {:.1}s",
            layers.len(),
            head.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    let ctx = WgpuContext::shared().context("wgpu adapter")?;
    let t0 = std::time::Instant::now();
    let dumped = dump_window_samples(ctx, &loader, &config_json, &samples, &opts)?;
    let mut index = String::new();
    let mut bytes_total = 0usize;
    for (i, s) in dumped.iter().enumerate() {
        let file = format!("{:05}.safetensors", i + 1);
        let bytes = s.to_safetensors_bytes(&d)?;
        bytes_total += bytes.len();
        std::fs::write(cli.out.join(&file), &bytes)?;
        index.push_str(
            &serde_json::json!({
                "file": file,
                "len": s.ids.len(),
                "completion_start": s.completion_start,
                "kv_sources": s.kv.keys().collect::<Vec<_>>(),
            })
            .to_string(),
        );
        index.push('\n');
    }
    std::fs::write(cli.out.join(INDEX_JSONL), index)?;
    std::fs::write(
        cli.out.join(WINDOW_JSON),
        serde_json::json!({
            "base_model": E4B_BASE_MODEL_NAME,
            "window_start": window_start,
            "grad_layers": cli.grad_layers,
            "layers_shard": LAYERS_SHARD,
            "head_shard": HEAD_SHARD,
            "samples": dumped.len(),
        })
        .to_string(),
    )?;
    eprintln!(
        "[e4b-window-dump] {} samples ({} bytes) at window_start {window_start} in {:.1}s -> {}",
        dumped.len(),
        bytes_total,
        t0.elapsed().as_secs_f64(),
        cli.out.display()
    );
    Ok(())
}
