use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use clap::{Parser, Subcommand};
use nv_train::{save_peft_raw, RawLora};
use nv_weights::lora_merge::{list_mergeable_modules, merge_lora_adapter_into_quantized_snapshot};
use std::path::PathBuf;

struct SplitmixNormal(u64);

impl SplitmixNormal {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit_f32(&mut self) -> f32 {
        (((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32).max(f32::MIN_POSITIVE)
    }

    fn next_normal(&mut self, std: f32) -> f32 {
        let u1 = self.next_unit_f32();
        let u2 = self.next_unit_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std
    }

    fn tensor(&mut self, std: f32, shape: (usize, usize)) -> Result<Tensor> {
        let vals: Vec<f32> = (0..shape.0 * shape.1)
            .map(|_| self.next_normal(std))
            .collect();
        Ok(Tensor::from_vec(vals, shape, &Device::Cpu)?)
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "nvk-lora-merge",
    about = "Fold a PEFT LoRA adapter into a quantized Qwen3.8 snapshot (nvfp4/fp8/bf16 storage \
             preserved byte-layout-identical, so serving kernels and speed are exactly the base \
             model's), or synthesize a random low-rank adapter shaped like a real fine-tune"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    Synth {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 16)]
        rank: usize,
        #[arg(long)]
        alpha: Option<f64>,
        #[arg(
            long,
            default_value = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        )]
        targets: String,
        #[arg(long, default_value_t = 5e-3)]
        b_std: f64,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    Merge {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        adapter: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Synth {
            base,
            out,
            rank,
            alpha,
            targets,
            b_std,
            seed,
        } => {
            let leaves: Vec<String> = targets
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let mods = list_mergeable_modules(&base)?;
            let mut rng = SplitmixNormal(seed ^ 0x10_9A_2C_55);
            let mut entries: Vec<RawLora> = Vec::new();
            for m in mods.iter().filter(|m| {
                m.module.starts_with("model.language_model.layers.")
                    && leaves.iter().any(|l| m.module.ends_with(&format!(".{l}")))
            }) {
                let a_std = (1.0 / (m.in_features as f64).sqrt()) as f32;
                let a = rng.tensor(a_std, (rank, m.in_features))?;
                let b = rng.tensor(b_std as f32, (m.out_features, rank))?;
                entries.push(RawLora {
                    module: m.module.clone(),
                    a,
                    b,
                });
            }
            if entries.is_empty() {
                bail!(
                    "no modules matched targets {leaves:?} under {}",
                    base.display()
                );
            }
            let alpha = alpha.unwrap_or(rank as f64);
            save_peft_raw(
                &out,
                &entries,
                rank,
                alpha,
                0.0,
                &base.display().to_string(),
            )?;
            println!(
                "synth adapter: modules={} rank={rank} alpha={alpha} b_std={b_std} out={}",
                entries.len(),
                out.display()
            );
            Ok(())
        }
        Cmd::Merge { base, adapter, out } => {
            let t0 = std::time::Instant::now();
            let report = merge_lora_adapter_into_quantized_snapshot(&base, &adapter, &out)
                .with_context(|| "merge failed; the staged out dir may be partial")?;
            println!(
                "merged {} modules (nvfp4_packed={} fp8_rowscale={} bf16={}) in {:.1}s; worst \
                 requant abs-err/amax={:.4}; out={}",
                report.modules.len(),
                report.count("nvfp4_packed"),
                report.count("fp8_rowscale"),
                report.count("bf16"),
                t0.elapsed().as_secs_f64(),
                report.worst(),
                report.out_model_file.display()
            );
            for m in &report.modules {
                if m.worst_abs_err_over_amax > 0.2 {
                    println!(
                        "  WARN {} ({}) requant abs-err/amax {:.4}",
                        m.module, m.kind, m.worst_abs_err_over_amax
                    );
                }
            }
            Ok(())
        }
    }
}
