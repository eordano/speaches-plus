use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use nv_train::laya::{run, LayaTrainArgs, ENCODER_TARGETS};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    about = "QLoRA for a Laya decision checkpoint: adapters on the encoder linears over a 4-bit or 8-bit round-tripped base, or the stored 16-bit weights"
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
    #[arg(long)]
    merged_out: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    rank: usize,
    #[arg(long)]
    alpha: Option<f64>,
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,
    #[arg(long, default_value_t = 100)]
    steps: usize,
    #[arg(long, default_value_t = 16)]
    batch: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, value_delimiter = ',', default_values_t = ENCODER_TARGETS.map(String::from))]
    target: Vec<String>,
    #[arg(long, default_value_t = nv_train::laya::BASE_BITS_DEFAULT_IS_THE_STORED_WEIGHTS_BECAUSE_NOTHING_THIS_SMALL_NEEDS_A_QUANTIZED_BASE)]
    base_bits: u32,
    #[arg(long, default_value_t = false)]
    fixed_option_order: bool,
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
    let summary = run(
        &LayaTrainArgs {
            base: cli.base,
            data: cli.data,
            eval_data: cli.eval_data,
            out: cli.out,
            merged_out: cli.merged_out,
            rank: cli.rank,
            alpha: cli.alpha,
            lr: cli.lr,
            steps: cli.steps,
            batch: cli.batch,
            seed: cli.seed,
            targets: cli.target,
            base_bits: cli.base_bits,
            shuffle_options: !cli.fixed_option_order,
        },
        &device,
    )?;
    let eval =
        |e: &nv_train::laya::Eval| serde_json::json!({"loss": e.loss, "accuracy": e.accuracy});
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "device": summary.device,
            "train_items": summary.train_items,
            "eval_items": summary.eval_items,
            "first_loss": summary.losses.first(),
            "last_loss": summary.losses.last(),
            "losses": summary.losses,
            "seconds_per_step": summary.seconds_per_step,
            "eval_dense_base": summary.eval_dense_base.as_ref().map(eval),
            "eval_trained_base": eval(&summary.eval_trained_base),
            "eval_adapter": eval(&summary.eval_adapter),
            "every_gradient_was_finite": summary.every_gradient_was_finite,
            "smallest_lora_b_amax": summary.smallest_lora_b_amax,
            "adapter_modules": summary.adapter_modules,
        }))?
    );
    Ok(())
}
