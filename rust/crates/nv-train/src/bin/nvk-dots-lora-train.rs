use anyhow::Result;
use candle_core::{DType, Device};
use clap::Parser;
use nv_train::dots::{dots_default_targets, run, DotsTrainArgs};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "nvk-dots-lora-train",
    about = "QLoRA for the dots.ocr text decoder: the frozen base is the mlx-community 4-bit \
             affine snapshot dequantized once at load, fp32 LoRA A/B adapters train on the top \
             --grad-layers layers through an exact candle-autograd forward, vision rows come \
             precomputed from nvk-dots-vision-dump so the tower never loads, and the adapter \
             saves in PEFT layout under the checkpoint's own language_model.model.* module names"
)]
struct Cli {
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    eval_data: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 8)]
    grad_layers: usize,
    #[arg(long, default_value_t = 16)]
    rank: usize,
    #[arg(long)]
    alpha: Option<f64>,
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,
    #[arg(long, default_value_t = 1.0)]
    b_lr_ratio: f64,
    #[arg(long, default_value_t = 40)]
    steps: usize,
    #[arg(long)]
    targets: Option<String>,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    bf16: bool,
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

fn pick_device(cpu: bool) -> Device {
    if cpu {
        return Device::Cpu;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = pick_device(cli.cpu);
    let targets = match &cli.targets {
        Some(s) => s
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect(),
        None => dots_default_targets(),
    };
    let args = DotsTrainArgs {
        base: cli.base,
        data: cli.data,
        eval_data: cli.eval_data,
        out: cli.out,
        grad_layers: cli.grad_layers,
        rank: cli.rank,
        alpha: cli.alpha.unwrap_or(cli.rank as f64 * 2.0),
        lr: cli.lr,
        b_lr_ratio: cli.b_lr_ratio,
        steps: cli.steps,
        targets,
        seed: cli.seed,
        dtype: if cli.bf16 { DType::BF16 } else { DType::F32 },
    };
    let summary = run(&args, &device)?;
    println!(
        "{}",
        serde_json::json!({
            "losses": summary.losses,
            "step_seconds": summary.step_seconds,
            "prefix_seconds": summary.prefix_seconds,
            "window_build_seconds": summary.window_build_seconds,
            "window_start": summary.window_start,
            "modules": summary.modules.len(),
            "trainable_vars": summary.trainable_vars,
            "num_examples": summary.num_examples,
            "num_eval_examples": summary.num_eval_examples,
            "eval_loss_base": summary.eval_loss_base(),
            "eval_loss_adapter": summary.eval_loss_adapter(),
            "eval_ce_base": summary.eval_ce_base,
            "eval_ce_adapter": summary.eval_ce_adapter,
            "min_module_delta_amax": summary.min_module_delta_amax,
            "max_module_delta_amax": summary.max_module_delta_amax,
            "all_grads_finite": summary.all_grads_finite,
            "device": summary.device,
            "adapter_dir": summary.adapter_dir,
        })
    );
    Ok(())
}
