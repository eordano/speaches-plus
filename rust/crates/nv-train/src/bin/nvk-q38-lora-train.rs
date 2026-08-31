use anyhow::Result;
use candle_core::{DType, Device};
use clap::Parser;
use nv_train::qwen38::{run, Q38TrainArgs};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "nvk-q38-lora-train",
    about = "Train a PEFT LoRA adapter on the Qwen3.8 dense arm through an exact candle-autograd \
             forward of the GDN + full-attention trunk, host-dequantizing the fp8/nvfp4 \
             checkpoint. Layers below the trainable window are frozen and streamed once, so the \
             adapter it writes covers the top --grad-layers layers and merges with nvk-lora-merge \
             unchanged"
)]
struct Cli {
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long, default_value_t = 65536)]
    corpus_bytes: usize,
    #[arg(long, default_value_t = 256)]
    seq_len: usize,
    #[arg(long, default_value_t = 4)]
    examples: usize,
    #[arg(long, default_value_t = 4)]
    grad_layers: usize,
    #[arg(long, default_value_t = 16)]
    rank: usize,
    #[arg(long)]
    alpha: Option<f64>,
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,
    #[arg(long, default_value_t = 12)]
    steps: usize,
    #[arg(long)]
    targets: Option<String>,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = if cli.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };
    let targets = match &cli.targets {
        Some(s) => s
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect(),
        None => Q38TrainArgs::default_targets(),
    };
    let args = Q38TrainArgs {
        base: cli.base,
        out: cli.out,
        corpus: cli.corpus,
        corpus_bytes: cli.corpus_bytes,
        seq_len: cli.seq_len,
        examples: cli.examples,
        grad_layers: cli.grad_layers,
        rank: cli.rank,
        alpha: cli.alpha.unwrap_or(cli.rank as f64),
        lr: cli.lr,
        steps: cli.steps,
        targets,
        seed: cli.seed,
        dtype: if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        },
    };
    let summary = run(&args, &device)?;
    println!(
        "device={} resident_base={} window_start={} window_kinds={:?} modules={} vars={} \
         examples={} seq_len={} prefix_s={:.1} window_build_s={:.1}",
        summary.device,
        summary.resident_base,
        summary.window_start,
        summary.window_kinds,
        summary.modules.len(),
        summary.trainable_vars,
        summary.examples,
        summary.seq_len,
        summary.prefix_seconds,
        summary.window_build_seconds
    );
    println!(
        "losses={:?} monotone={} drop={:.6} grads_finite={} delta_amax_min={:.3e} \
         delta_amax_max={:.3e} tok_per_s={:.1} adapter={}",
        summary.losses,
        summary.losses_are_monotonically_decreasing(),
        summary.total_drop(),
        summary.all_grads_finite,
        summary.min_module_delta_amax,
        summary.max_module_delta_amax,
        summary.tokens_per_second(),
        summary.adapter_dir.display()
    );
    Ok(())
}
