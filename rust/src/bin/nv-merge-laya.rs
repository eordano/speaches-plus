use anyhow::Result;
use clap::Parser;
use speaches_plus::oapi::systemone::FINE_TUNED_ID_LEAF_INSIDE_A_MERGED_CHECKPOINT_IS_HOW_A_RESTART_REDISCOVERS_IT as SERVED_ID_LEAF;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "nv-merge-laya",
    about = "Fold a saved Laya LoRA adapter (adapter_config.json, adapter_model.safetensors) into a base checkpoint's stored weights, so a host that receives only the adapter writes the merged checkpoint /v1/systemone loads"
)]
struct Cli {
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    adapter: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    served_id: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let modules =
        nv_train::laya::merge_saved_adapter_onto_stored_weights(&cli.base, &cli.adapter, &cli.out)?;
    if let Some(id) = &cli.served_id {
        std::fs::write(cli.out.join(SERVED_ID_LEAF), id)?;
    }
    println!(
        "{}",
        serde_json::json!({"merged_modules": modules, "out": cli.out, "served_id": cli.served_id})
    );
    Ok(())
}
