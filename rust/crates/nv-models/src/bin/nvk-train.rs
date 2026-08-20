use anyhow::Result;
use clap::Parser;
use nv_models::train_runner::{run, TrainArgs};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "nvk-train",
    about = "Train a servable LoRA (PEFT) adapter on Gemma4Moe"
)]
struct Cli {
    #[arg(long)]
    base: PathBuf,

    #[arg(long)]
    data: PathBuf,

    #[arg(long)]
    out: PathBuf,

    #[arg(long, default_value_t = 8)]
    rank: usize,

    #[arg(long)]
    alpha: Option<f64>,

    #[arg(long, default_value = "q,k,v,o,gate,up,down")]
    target: String,

    #[arg(long, default_value_t = 100)]
    steps: usize,

    #[arg(long, default_value_t = 0.05)]
    lr: f64,

    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let targets: Vec<String> = cli
        .target
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let args = TrainArgs {
        base: cli.base,
        data: cli.data,
        out: cli.out.clone(),
        rank: cli.rank,
        alpha: cli.alpha.unwrap_or(cli.rank as f64),
        targets,
        steps: cli.steps,
        lr: cli.lr,
        seed: cli.seed,
    };

    let s = run(&args)?;

    println!("BASE_DTYPE {}", s.base_dtype);
    println!("NVFP4_BASE {}", if s.nvfp4_base { 1 } else { 0 });
    println!("DENSE_BASE {}", if s.dense_base { 1 } else { 0 });
    println!("DEVICE {}", s.device);
    println!("CHECKPOINTED {}", if s.checkpointed { 1 } else { 0 });
    println!("LMHEAD_CHUNK {}", s.lmhead_chunk);
    println!("LAYERS_BUILT {}", s.layers_built);
    println!("CONFIG_LAYERS {}", s.config_layers);
    println!("NUM_EXAMPLES {}", s.num_examples);
    println!("TRAINABLE_VARS {}", s.trainable_vars);
    println!("DETERMINISTIC {}", if s.deterministic { 1 } else { 0 });
    for (i, l) in s.losses.iter().enumerate() {
        println!("LOSS_STEP {i} {l:.6e}");
    }
    if let (Some(first), Some(last)) = (s.losses.first(), s.losses.last()) {
        println!("LOSS_FIRST {first:.6e}");
        println!("LOSS_LAST {last:.6e}");
    }
    println!("SERVING_EQUIV_MAXABS {:.3e}", s.serving_equiv_maxabs);
    println!("MODULES {}", s.modules.join(","));
    println!("ADAPTER_PATH {}", s.adapter_path.display());
    println!("CONFIG_PATH {}", s.config_path.display());
    println!("OK");
    Ok(())
}
